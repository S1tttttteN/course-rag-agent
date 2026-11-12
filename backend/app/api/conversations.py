"""对话与问答的 HTTP 入口，接收提问并通过 SSE 回放已持久化的运行事件。"""

import asyncio
import json
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.models import (
    CHAT_MODEL,
    CHAT_PROVIDER,
    CHAT_PROVIDERS,
    TOP_K,
    OLLAMA_BASE_URL,
    chat_provider_conf,
    _get_api_key,
)
from app.db.models import Citation, Run, RunEvent
from app.db.session import SessionLocal, get_session
from app.retrieval.reranker import is_available
from app.services import conversations

router = APIRouter(tags=["conversations"])
SessionDep = Annotated[Session, Depends(get_session)]
Title = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)
]


class ConversationInput(BaseModel):
    """创建对话的请求体。

    Attributes:
        title: 对话标题，省略时使用「新对话」。去除首尾空白后须为 1 至 80 个字符。
    """

    model_config = ConfigDict(extra="forbid")
    title: Title = "新对话"


class ConversationRename(BaseModel):
    """重命名对话的请求体。

    Attributes:
        title: 必填的新标题，去除首尾空白后须为 1 至 80 个字符。
    """

    model_config = ConfigDict(extra="forbid")
    title: Title


class QuestionInput(BaseModel):
    """发起问答运行的请求体。

    Attributes:
        question: 问题正文，去除首尾空白后须为 1 至 4000 个字符。
        top_k: 检索候选条数，取值为 1 至 10，默认为 4。
        semantic: 是否启用语义检索，默认启用。
        rerank: 是否对检索结果重排序，默认启用。
        mode: 问答模式，可选 `rag` 或 `agent`，默认为 `rag`。
        web_search: 是否允许本轮 Agent 联网，默认关闭。
        answer_review: 是否在发布本轮正文前核对证据，默认关闭。
    """

    model_config = ConfigDict(extra="forbid")
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
    ]
    top_k: int = Field(default=4, ge=1, le=10)
    semantic: bool = True
    rerank: bool = True
    mode: Literal["rag", "agent"] = "rag"
    web_search: bool = Field(default=False, strict=True)
    answer_review: bool = Field(default=False, strict=True)


@router.get("/chat/config")
def chat_config():
    """返回前端使用的对话模型配置和功能开关。

    使用 Ollama 时，还会探测本地服务是否已安装当前配置的模型。

    Returns:
        包含供应商、模型名称、配置状态及提示、默认检索条数、重排序可用性、
        问答模式、答案审查和联网搜索开关的字典。
    """
    conf = CHAT_PROVIDERS.get(CHAT_PROVIDER, {})
    configured, note = True, ""
    try:
        conf = chat_provider_conf()
        _get_api_key(conf, CHAT_PROVIDER, "对话模型")
    except (RuntimeError, ValueError) as error:
        configured, note = False, str(error)
    if configured and CHAT_PROVIDER == "ollama":
        import httpx

        try:
            response = httpx.get(
                OLLAMA_BASE_URL.rstrip("/") + "/api/tags", timeout=2, trust_env=False
            )
            response.raise_for_status()
            names = {item["name"] for item in response.json().get("models", [])}
            configured = (CHAT_MODEL or conf["default_model"]) in names
            if not configured:
                note = "Ollama未安装当前配置的模型，请核对本机模型名称。"
        except Exception:
            configured, note = False, "无法连接本机Ollama，请启动Ollama后重试。"
    return {
        "provider": CHAT_PROVIDER,
        "model": CHAT_MODEL or conf.get("default_model", ""),
        "configured": configured,
        "note": note,
        "top_k": min(10, max(1, TOP_K)),
        "rerank_available": is_available(),
        "mode": "agent",
        "answer_review": False,
        "web_search": False,
    }


@router.get("/courses/{course_id}/conversations")
def list_conversations(course_id: UUID, session: SessionDep):
    """列出指定课程的对话。

    Args:
        course_id: 课程 ID。
        session: 用于查询课程和对话的数据库会话。

    Returns:
        由对话摘要字典组成的列表。

    Raises:
        CourseNotFound: 指定课程不存在。
    """
    return conversations.list_conversations(session, str(course_id))


@router.post("/courses/{course_id}/conversations", status_code=201)
def create_conversation(course_id: UUID, body: ConversationInput, session: SessionDep):
    """在指定课程下创建对话。

    Args:
        course_id: 课程 ID。
        body: 包含初始标题的请求体。
        session: 用于保存对话的数据库会话。

    Returns:
        新对话的摘要字典。

    Raises:
        CourseNotFound: 指定课程不存在。
    """
    return conversations.create(session, str(course_id), body.title)


@router.get("/courses/{course_id}/conversations/{conversation_id}")
def conversation_detail(course_id: UUID, conversation_id: UUID, session: SessionDep):
    """获取对话详情及用于续接事件流的快照。

    Args:
        course_id: 课程 ID。
        conversation_id: 对话 ID。
        session: 用于读取对话历史的数据库会话。

    Returns:
        包含对话信息、历史消息和当前活动运行的字典。助手消息附带引用和运行信息，
        运行信息包含最后一条事件的 ID。

    Raises:
        CourseNotFound: 指定课程不存在。
        conversations.ConversationError: 对话不存在或不属于指定课程。
    """
    return conversations.detail(session, str(course_id), str(conversation_id))


@router.patch("/courses/{course_id}/conversations/{conversation_id}")
def rename_conversation(
    course_id: UUID,
    conversation_id: UUID,
    body: ConversationRename,
    session: SessionDep,
):
    """更新对话标题和最后更新时间。

    Args:
        course_id: 课程 ID。
        conversation_id: 对话 ID。
        body: 包含新标题的请求体。
        session: 用于更新对话的数据库会话。

    Returns:
        更新后的对话摘要字典。

    Raises:
        CourseNotFound: 指定课程不存在。
        conversations.ConversationError: 对话不存在或不属于指定课程。
    """
    return conversations.rename(
        session, str(course_id), str(conversation_id), body.title
    )


@router.delete("/courses/{course_id}/conversations/{conversation_id}", status_code=204)
async def delete_conversation(
    course_id: UUID, conversation_id: UUID, request: Request, session: SessionDep
):
    """取消对话内的活动运行并删除对话。

    Args:
        course_id: 课程 ID。
        conversation_id: 对话 ID。
        request: 用于访问应用运行管理器的 HTTP 请求。
        session: 用于校验和删除对话的数据库会话。

    Returns:
        状态码为 204 的空响应。

    Raises:
        CourseNotFound: 指定课程不存在。
        conversations.ConversationError: 对话不存在或不属于指定课程。
    """
    conversations.require_conversation(session, str(course_id), str(conversation_id))
    request.app.state.runs.cancel_scope(str(course_id), str(conversation_id))
    conversations.remove(session, str(course_id), str(conversation_id))
    return Response(status_code=204)


@router.post(
    "/courses/{course_id}/conversations/{conversation_id}/runs", status_code=202
)
async def create_run(
    course_id: UUID,
    conversation_id: UUID,
    body: QuestionInput,
    request: Request,
    session: SessionDep,
):
    """持久化问答运行后启动后台回答任务。

    Args:
        course_id: 课程 ID。
        conversation_id: 接收提问的对话 ID。
        body: 包含问题和检索选项的请求体。
        request: 用于检查任务容量和启动后台任务的 HTTP 请求。
        session: 用于保存运行、消息和初始事件的数据库会话。

    Returns:
        包含运行 ID、排队状态和最后一条事件 ID 的运行信息字典。

    Raises:
        CourseNotFound: 指定课程不存在。
        conversations.ConversationError: 对话不存在或不属于指定课程（404）、
            对话已有活动运行（409），或后台任务数已达 6 个（429）。
    """
    if len(request.app.state.runs.tasks) >= 6:
        raise conversations.ConversationError("等待中的回答较多，请稍后再试。", 429)
    result = conversations.create_run(
        session,
        str(course_id),
        str(conversation_id),
        body.question,
        body.top_k,
        body.semantic,
        body.rerank,
        body.mode,
        web_search=body.web_search,
        answer_review=body.answer_review,
    )
    # 运行记录提交后再启动任务，前端可用返回的 ID 单独订阅进度。
    request.app.state.runs.start(result["id"])
    return result


@router.post(
    "/courses/{course_id}/conversations/{conversation_id}/runs/{run_id}/cancel"
)
async def cancel_run(
    course_id: UUID,
    conversation_id: UUID,
    run_id: UUID,
    request: Request,
    session: SessionDep,
):
    """请求取消指定运行并重新读取其状态。

    Args:
        course_id: 课程 ID。
        conversation_id: 对话 ID。
        run_id: 待取消的运行 ID。
        request: 用于访问应用运行管理器的 HTTP 请求。
        session: 用于校验运行归属和重新读取运行状态的数据库会话。

    Returns:
        提交取消请求后读取的运行信息字典。已结束的运行保留原有状态。

    Raises:
        CourseNotFound: 指定课程不存在。
        conversations.ConversationError: 对话或运行不存在，或其归属与请求路径不符。
    """
    conversations.require_run(
        session, str(course_id), str(conversation_id), str(run_id)
    )
    request.app.state.runs.cancel(str(run_id))
    session.expire_all()
    return conversations.run_info(session, session.get(Run, str(run_id)))


@router.get(
    "/courses/{course_id}/conversations/{conversation_id}/citations/{citation_id}"
)
def read_citation(
    course_id: UUID, conversation_id: UUID, citation_id: UUID, session: SessionDep
):
    """读取指定对话中的引用详情。

    Args:
        course_id: 课程 ID。
        conversation_id: 对话 ID。
        citation_id: 引用 ID。
        session: 用于校验引用归属和读取来源状态的数据库会话。

    Returns:
        包含引用原文、来源信息和来源当前状态的字典。

    Raises:
        CourseNotFound: 指定课程不存在。
        conversations.ConversationError: 对话、引用或对应的运行不存在，
            或其归属与请求路径不符。
    """
    conversations.require_conversation(session, str(course_id), str(conversation_id))
    citation = session.get(Citation, str(citation_id))
    if citation is None:
        raise conversations.ConversationError("参考片段不存在。", 404)
    conversations.require_run(
        session, str(course_id), str(conversation_id), citation.run_id
    )
    return conversations.citation_info(session, citation)


@router.get("/courses/{course_id}/conversations/{conversation_id}/runs/{run_id}/events")
async def run_events(
    course_id: UUID,
    conversation_id: UUID,
    run_id: UUID,
    request: Request,
    after: Annotated[int, Query(ge=0)] = 0,
):
    """回放指定运行的 SSE 事件并持续推送后续事件。

    Args:
        course_id: 课程 ID。
        conversation_id: 对话 ID。
        run_id: 要订阅的运行 ID。
        request: 用于读取 `Last-Event-ID` 请求头和检测客户端断开的 HTTP 请求。
        after: 已处理的事件 ID，默认为 0。有效的 `Last-Event-ID` 与此值取较大者，
            事件流只发送 ID 大于该游标的持久化事件。

    Returns:
        媒体类型为 `text/event-stream` 的流式响应。

    Raises:
        CourseNotFound: 指定课程不存在。
        conversations.ConversationError: 对话或运行不存在，或其归属与请求路径不符。
    """
    with SessionLocal() as session:
        conversations.require_run(
            session, str(course_id), str(conversation_id), str(run_id)
        )
    # 页面加载用 after 接续详情快照，浏览器自动重连用 Last-Event-ID 接续已收到的事件。
    try:
        after = max(after, int(request.headers.get("last-event-id", "0")))
    except ValueError:
        pass

    async def events():
        """按事件 ID 轮询已持久化的记录，直到运行结束或客户端断开。

        Yields:
            SSE 格式的字符串，内容为持久化事件、空闲时的保活注释，
            或终止状态下补发的 `done` 事件。
        """
        cursor, ticks = after, 0
        while not await request.is_disconnected():
            # 每次轮询结束即释放会话，避免在 SSE 长连接上持有数据库事务。
            with SessionLocal() as database:
                run = database.get(Run, str(run_id))
                state = run.status if run else "deleted"
                rows = [
                    (item.id, item.kind, item.data)
                    for item in database.scalars(
                        select(RunEvent)
                        .where(
                            RunEvent.run_id == str(run_id),
                            RunEvent.id > cursor,
                        )
                        .order_by(RunEvent.id)
                        .limit(100)
                    )
                ]
            for event_id, kind, data in rows:
                cursor = event_id
                yield f"id: {event_id}\nevent: {kind}\ndata: {data}\n\n"
                if kind == "done":
                    return
            if state not in conversations.ACTIVE and not rows:
                yield f"event: done\ndata: {json.dumps({'status': state})}\n\n"
                return
            if len(rows) == 100:
                continue
            ticks += 1
            if ticks % 100 == 0:
                yield ": keepalive\n\n"
            await asyncio.sleep(0.15)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
