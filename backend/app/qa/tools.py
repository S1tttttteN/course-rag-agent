"""定义课程 Agent 的工具和本轮工具共享的上下文。"""

import asyncio
import json
from dataclasses import dataclass
from typing import Annotated, Any

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import InjectedToolArg
from pydantic import BaseModel, ConfigDict, Field

from app.retrieval.web import (
    search_web as web_search,
    read_webpage as web_read,
    public_url,
)
from app.services import runs
from app.services.documents import catalog, read_version_page


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

    def register(self, records, *, web=False):
        """保存来源快照并返回本批可供模型引用的字段。"""
        sources, matched = runs.snapshot(self.run_id, records, web=web)
        self.sources = {row["chunk_id"]: row for row in sources}
        fields = ("number", "kind", "url", "source", "text")
        if not web:
            fields += ("document_id", "version_id", "page")
        return [{key: row[key] for key in fields} for row in matched]


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
        DocumentError: 资料不存在、归属不符或版本已变化。
        BudgetExceeded: 片段超过本轮引用预算。
    """
    context = runtime.context
    page = await context.manager.offload(
        read_version_page, context.course_id, document_id, version_id, offset, limit
    )
    records = [
        {**chunk, "document_id": document_id, "version_id": version_id}
        for chunk in page["chunks"]
    ]
    end = offset + len(records)
    return json.dumps(
        {
            "sources": context.register(records),
            "next_offset": end if end < page["total"] else None,
            "total_chunks": page["total"],
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
            "sources": context.register(records, web=True),
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
            "sources": context.register([record], web=True),
            "note": "网页正文截取，非完整站点或全文。",
        },
        ensure_ascii=False,
    )
