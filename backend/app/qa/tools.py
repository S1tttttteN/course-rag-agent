"""定义课程 Agent 的工具和本轮工具共享的上下文。"""

import asyncio
import json
from dataclasses import dataclass
from typing import Annotated, Any

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import InjectedToolArg
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from app.db.models import Document, DocumentVersion
from app.db.session import SessionLocal
from app.retrieval.web import (
    search_web as web_search,
    read_webpage as web_read,
    public_url,
)
from app.services import runs
from app.services.documents import preview, require_course, require_document

MAX_EVIDENCE_CHARS = 20000
MAX_SOURCES = 24


class BudgetExceeded(Exception):
    """工具调用数或引用容量耗尽时抛出的异常。"""

    pass


class RuntimeToolInput(BaseModel):
    """校验模型参数并接收由 LangGraph 注入的运行时。"""

    model_config = ConfigDict(extra="forbid")
    # 显式参数模型接收 ToolNode 注入的值，并将其从模型可见的参数中移除。
    runtime: Annotated[object, InjectedToolArg] = Field(exclude=True)


class SearchInput(RuntimeToolInput):
    """课程资料检索和网络搜索工具的参数模型。

    Attributes:
        query: 待查阅的主题或关键词，长度为 1 至 500 个字符。
    """

    query: str = Field(
        min_length=1,
        max_length=500,
        description="本轮需要查阅的具体主题；补查时换用更具体的关键词",
    )


class ListInput(RuntimeToolInput):
    """课程资料目录工具的分页参数。

    Attributes:
        offset: 目录起始偏移量，默认为 0，每次最多返回 10 份资料。
    """

    offset: int = Field(
        default=0, ge=0, le=10000, description="文件目录偏移；每次最多返回10份"
    )


class WebReadInput(RuntimeToolInput):
    """网页阅读工具的参数模型。

    Attributes:
        url: 本轮网络搜索返回的网页地址，长度为 8 至 2048 个字符。
    """

    url: str = Field(
        min_length=8, max_length=2048, description="本轮search_web返回的网页URL"
    )


class ReadInput(RuntimeToolInput):
    """课程资料阅读工具的分页参数。

    Attributes:
        document_id: 本轮目录或检索结果返回的资料 ID。
        version_id: 本轮目录或检索结果返回的版本 ID。
        offset: 从零开始的片段偏移量，默认为 0。
        limit: 本次读取的连续片段数，取值为 1 至 4，默认为 3。
    """

    document_id: str = Field(
        min_length=36, max_length=36, description="目录或搜索结果中的资料ID"
    )
    offset: int = Field(
        default=0, ge=0, le=100000, description="从0开始的片段偏移，不是PDF页码"
    )
    limit: int = Field(default=3, ge=1, le=4, description="本次连续读取的片段数")
    version_id: str = Field(
        min_length=36,
        max_length=36,
        description="目录或搜索结果中的版本ID；版本变化时需重新列目录",
    )


def catalog(course_id, offset):
    """分页查询课程资料并标记各版本的可读状态。

    Args:
        course_id: 课程 ID。
        offset: 结果起始偏移量。

    Returns:
        最多 10 条目录记录及可选的下一页偏移量。

    Raises:
        CourseNotFound: 指定课程不存在。
    """
    with SessionLocal() as session:
        require_course(session, course_id)
        rows = list(
            session.scalars(
                select(Document)
                .where(Document.course_id == course_id)
                .order_by(Document.created_at, Document.id)
                .offset(offset)
                .limit(11)
            )
        )
        items = []
        for document in rows[:10]:
            version = (
                session.get(DocumentVersion, document.active_version_id)
                if document.active_version_id
                else None
            )
            items.append(
                {
                    "document_id": document.id,
                    "name": document.name,
                    "version_id": version.id if version else None,
                    "chunks": version.chunk_count if version else 0,
                    "readable": bool(version and version.status == "ready"),
                }
            )
        return {
            "documents": items,
            "next_offset": offset + 10 if len(rows) > 10 else None,
        }


def read_records(course_id, document_id, version_id, offset, limit):
    """读取指定可用版本的一页连续片段。

    Args:
        course_id: 课程 ID。
        document_id: 资料 ID。
        version_id: 资料版本 ID。
        offset: 结果起始偏移量。
        limit: 最多返回的条数。

    Returns:
        片段记录、可选的下一页偏移量及总片段数。

    Raises:
        CourseNotFound: 指定课程不存在。
        DocumentError: 资料不存在或归属不符，或没有可预览版本。
        ValueError: 指定版本已非当前可用版本。
    """
    with SessionLocal() as session:
        document = require_document(session, course_id, document_id)
        if document.active_version_id != version_id:
            raise ValueError("version changed")
        result = preview(session, course_id, document_id, offset, limit)
        records = [
            {**chunk, "document_id": document_id, "version_id": version_id}
            for chunk in result["chunks"]
        ]
        end = offset + len(records)
        return records, end if end < result["total"] else None, result["total"]


@dataclass
class CourseToolContext:
    """保存本轮工具配置、共享状态和已登记引用。

    Attributes:
        manager: 执行阻塞检索和工具操作的运行管理器。
        run_id: 当前问答运行 ID。
        course_id: 工具访问范围内的课程 ID。
        lock: 串行执行工具调用的异步锁。
        calls: 已经消耗的工具调用次数。
        sources: 按片段 ID 索引的已登记引用。
        catalog_items: 按资料 ID 索引的已查阅目录。
        web_urls: 本轮搜索返回并登记的网页地址。
    """

    manager: Any
    run_id: str
    course_id: str

    def __post_init__(self):
        """初始化仅供本轮使用的工具状态。"""
        self.lock = asyncio.Lock()
        self.calls = 0
        self.sources = {}
        self.catalog_items = {}
        self.web_urls = set()

    def register(self, records):
        """保存课程原文快照并为当前候选提供引用编号。

        Args:
            records: 由课程检索或阅读工具取得的片段记录。

        Returns:
            本次候选中成功登记的引用及对应原文。

        Raises:
            BudgetExceeded: 新增候选将超过引用条数或原文字数预算。
        """
        fresh = {row["id"]: row for row in records if row["id"] not in self.sources}
        if len(self.sources) + len(fresh) > MAX_SOURCES:
            raise BudgetExceeded()
        # Use authoritative snapshot text, not model-supplied text. New records come from trusted tools.
        if (
            sum(len(row["text"]) for row in self.sources.values())
            + sum(len(row["text"]) for row in fresh.values())
            > MAX_EVIDENCE_CHARS
        ):
            raise BudgetExceeded()
        registered = runs.snapshot(self.run_id, records)
        self.sources = {row["chunk_id"]: row for row in registered}
        wanted = {row["id"] for row in records}
        return [
            {
                key: row[key]
                for key in (
                    "number",
                    "kind",
                    "url",
                    "document_id",
                    "version_id",
                    "source",
                    "text",
                    "page",
                )
            }
            for row in registered
            if row["chunk_id"] in wanted
        ]

    def register_web(self, records):
        """保存网络来源快照并返回本次登记的引用。

        Args:
            records: 由网络搜索或网页阅读工具取得的来源记录。

        Returns:
            对应本批来源地址和类型的已登记引用。

        Raises:
            BudgetExceeded: 该批来源将超过引用条数或原文字数预算。
        """
        if (
            len(self.sources) + len(records) > MAX_SOURCES
            or sum(len(row["text"]) for row in self.sources.values())
            + sum(len(row["text"]) for row in records)
            > MAX_EVIDENCE_CHARS
        ):
            raise BudgetExceeded()
        registered = runs.snapshot_web(self.run_id, records)
        self.sources = {row["chunk_id"]: row for row in registered}
        wanted = {(row["url"], row["kind"]) for row in records}
        return [
            {key: row[key] for key in ("number", "kind", "url", "source", "text")}
            for row in registered
            if (row["url"], row["kind"]) in wanted
        ]


@tool(args_schema=SearchInput)
async def search_course(query: str, runtime: ToolRuntime[CourseToolContext]) -> str:
    """搜索当前课程已就绪的资料。返回带稳定引用编号的原文；必要时可换词再次搜索。

    Args:
        query: 要在当前课程资料中检索的主题或关键词。

    Returns:
        包含已编号来源和检索摘要的 JSON 字符串。

    Raises:
        BudgetExceeded: 候选来源超过本轮引用预算。
    """
    context = runtime.context
    result = await context.manager.retrieve(context.course_id, query)
    sources = context.register(result["results"])
    return json.dumps(
        {"sources": sources, "retrieval": result["trace"]}, ensure_ascii=False
    )


@tool(args_schema=ListInput)
async def list_documents(runtime: ToolRuntime[CourseToolContext], offset: int = 0) -> str:
    """分页列出当前课程资料目录、可读状态及资料/版本ID。目录不是知识内容，阅读需调用read_document。

    Args:
        offset: 从零开始的目录偏移量，默认为 0。

    Returns:
        包含至多 10 份资料及下一页偏移量的 JSON 字符串。

    Raises:
        CourseNotFound: 当前课程已不存在。
    """
    context = runtime.context
    result = await context.manager.offload(catalog, context.course_id, offset)
    context.catalog_items.update(
        {row["document_id"]: row for row in result["documents"]}
    )
    return json.dumps(result, ensure_ascii=False)


@tool(args_schema=ReadInput)
async def read_document(
    document_id: str,
    version_id: str,
    runtime: ToolRuntime[CourseToolContext],
    offset: int = 0,
    limit: int = 3,
) -> str:
    """按连续片段分页阅读当前课程资料。使用返回的next_offset继续阅读；PDF来源另带原页码。

    Args:
        document_id: 目录或检索结果中的资料 ID。
        version_id: 目录或检索结果中的可用版本 ID。
        offset: 从零开始的片段偏移量，默认为 0。
        limit: 本次读取的连续片段数，默认为 3。

    Returns:
        包含已编号片段、下一页偏移量和总片段数的 JSON 字符串。

    Raises:
        CourseNotFound: 当前课程已不存在。
        DocumentError: 资料不存在、归属不符或没有可预览版本。
        ValueError: 指定版本已非当前可用版本。
        BudgetExceeded: 片段超过本轮引用预算。
    """
    context = runtime.context
    records, following, total = await context.manager.offload(
        read_records, context.course_id, document_id, version_id, offset, limit
    )
    return json.dumps(
        {
            "sources": context.register(records),
            "next_offset": following,
            "total_chunks": total,
        },
        ensure_ascii=False,
    )


@tool(args_schema=SearchInput)
async def search_web(query: str, runtime: ToolRuntime[CourseToolContext]) -> str:
    """搜索公开网络，补充课程资料未覆盖的知识、最新信息或外部参考。返回带引用编号的搜索摘要和URL，摘要不等于已读正文。

    Args:
        query: 要搜索的公开主题或关键词。

    Returns:
        包含已编号搜索摘要及其用途说明的 JSON 字符串。

    Raises:
        BudgetExceeded: 摘要超过本轮引用预算。
    """
    context = runtime.context
    records = await context.manager.offload(web_search, query)
    context.web_urls.update(row["url"] for row in records)
    return json.dumps(
        {
            "sources": context.register_web(records),
            "note": "这些是搜索摘要；重要细节可继续read_webpage核对。",
        },
        ensure_ascii=False,
    )


@tool(args_schema=WebReadInput)
async def read_webpage(url: str, runtime: ToolRuntime[CourseToolContext]) -> str:
    """读取本轮网络搜索结果中的公开网页正文摘录，补充摘要未包含的细节。

    Args:
        url: 本轮搜索返回的公开网页地址。

    Returns:
        包含已编号正文摘录及其用途说明的 JSON 字符串。

    Raises:
        ValueError: 地址未出现在本轮搜索结果中，或网页不符合读取要求。
        BudgetExceeded: 正文摘录超过本轮引用预算。
    """
    context = runtime.context
    normalized = public_url(url)
    if normalized not in context.web_urls:
        raise ValueError("Search first, then read a returned URL")
    record = await context.manager.offload(web_read, normalized)
    return json.dumps(
        {
            "sources": context.register_web([record]),
            "note": "网页正文截取，非完整站点或全文。",
        },
        ensure_ascii=False,
    )
