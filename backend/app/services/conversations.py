"""管理课程内的对话、消息和问答运行，供 HTTP 接口和后台任务共享。"""

import json
from uuid import uuid4

from sqlalchemy import func, select

from app.core.locks import catalog_lock
from app.db.models import (
    Citation,
    Conversation,
    Document,
    Message,
    Run,
    RunEvent,
    utc_now,
)
from app.repositories.conversations import conversation_messages, course_conversations
from app.services.documents import require_course

ACTIVE = {"queued", "running", "cancelling"}


class ConversationError(Exception):
    """携带客户端错误信息的对话业务异常。

    Attributes:
        detail: 客户端可见的错误说明。
        status: 对应的 HTTP 状态码。
    """

    def __init__(self, detail, status=400):
        """保存对话业务错误说明和状态码。

        Args:
            detail: 面向客户端的错误说明。
            status: 对应的 HTTP 状态码。
        """
        self.detail, self.status = detail, status


def require_conversation(session, course_id, conversation_id):
    """确认指定对话存在且属于当前课程。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        course_id: 课程 ID。
        conversation_id: 对话 ID。

    Returns:
        通过归属校验的对话记录。

    Raises:
        CourseNotFound: 指定课程不存在。
        ConversationError: 对话不存在或不属于该课程。
    """
    require_course(session, course_id)
    conversation = session.get(Conversation, conversation_id)
    if conversation is None or conversation.course_id != course_id:
        raise ConversationError("对话不存在或不属于当前课程。", 404)
    return conversation


def require_run(session, course_id, conversation_id, run_id):
    """确认指定运行属于路径中的课程和对话。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        course_id: 课程 ID。
        conversation_id: 对话 ID。
        run_id: 问答运行 ID。

    Returns:
        通过归属校验的运行记录。

    Raises:
        CourseNotFound: 指定课程不存在。
        ConversationError: 对话或运行不存在，或归属不符。
    """
    require_conversation(session, course_id, conversation_id)
    run = session.get(Run, run_id)
    if run is None or run.conversation_id != conversation_id:
        raise ConversationError("本次回答不存在或不属于当前对话。", 404)
    return run


def conversation_info(conversation):
    """提取可返回给客户端的对话摘要。

    Args:
        conversation: 要转换的对话 ORM 记录。

    Returns:
        包含对话 ID、课程 ID、标题和时间戳的字典。
    """
    return {
        key: getattr(conversation, key)
        for key in ("id", "course_id", "title", "created_at", "updated_at")
    }


def list_conversations(session, course_id):
    """查询课程下的对话摘要。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        course_id: 课程 ID。

    Returns:
        按更新时间降序排列的对话摘要列表。

    Raises:
        CourseNotFound: 指定课程不存在。
    """
    require_course(session, course_id)
    return [
        conversation_info(item) for item in course_conversations(session, course_id)
    ]


def create(session, course_id, title):
    """新建对话并提交。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        course_id: 课程 ID。
        title: 对话标题。

    Returns:
        新对话的摘要字典。

    Raises:
        CourseNotFound: 指定课程不存在。
    """
    with catalog_lock:
        require_course(session, course_id)
        item = Conversation(course_id=course_id, title=title)
        session.add(item)
        session.commit()
        return conversation_info(item)


def rename(session, course_id, conversation_id, title):
    """保存对话的新标题和更新时间。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        course_id: 课程 ID。
        conversation_id: 对话 ID。
        title: 待保存的新标题。

    Returns:
        更新后的对话摘要。

    Raises:
        CourseNotFound: 指定课程不存在。
        ConversationError: 对话不存在或归属不符。
    """
    with catalog_lock:
        item = require_conversation(session, course_id, conversation_id)
        item.title, item.updated_at = title, utc_now()
        session.commit()
        return conversation_info(item)


def remove(session, course_id, conversation_id):
    """校验归属后删除对话并提交。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        course_id: 课程 ID。
        conversation_id: 对话 ID。

    Raises:
        CourseNotFound: 指定课程不存在。
        ConversationError: 对话不存在或归属不符。
    """
    with catalog_lock:
        session.delete(require_conversation(session, course_id, conversation_id))
        session.commit()


def citation_info(session, citation):
    """提取引用快照并识别当前来源状态。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        citation: 要转换的引用 ORM 记录。

    Returns:
        包含引用内容和来源状态的字典。
    """
    document = session.get(Document, citation.document_id)
    state = (
        "external"
        if citation.kind != "course"
        else "deleted"
        if document is None
        else "current"
        if document.active_version_id == citation.version_id
        else "updated"
    )
    return {
        **{
            key: getattr(citation, key)
            for key in (
                "id",
                "number",
                "kind",
                "url",
                "retrieved_at",
                "document_id",
                "version_id",
                "chunk_id",
                "source",
                "version",
                "text",
                "page",
                "start_index",
            )
        },
        "source_state": state,
    }


def citations_for_run(session, run_id):
    """读取一轮问答保存的全部引用。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        run_id: 问答运行 ID。

    Returns:
        按引用编号排序的引用详情列表。
    """
    return [
        citation_info(session, citation)
        for citation in session.scalars(
            select(Citation).where(Citation.run_id == run_id).order_by(Citation.number)
        )
    ]


def run_info(session, run):
    """生成运行摘要和 SSE 续接游标。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        run: 要转换的运行 ORM 记录。

    Returns:
        包含运行字段、反序列化后的执行摘要及最后事件 ID 的字典。
    """
    cursor = (
        session.scalar(select(func.max(RunEvent.id)).where(RunEvent.run_id == run.id))
        or 0
    )
    return {
        **{
            key: getattr(run, key)
            for key in (
                "id",
                "conversation_id",
                "question",
                "query",
                "mode",
                "status",
                "phase",
                "error",
                "created_at",
            )
        },
        "trace": json.loads(run.trace),
        "last_event_id": cursor,
    }


def detail(session, course_id, conversation_id):
    # 与运行写入共用一把锁，确保消息正文与 SSE 起始游标来自同一个状态。
    """在目录锁内读取一致的对话快照。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        course_id: 课程 ID。
        conversation_id: 对话 ID。

    Returns:
        包含对话摘要、历史消息及可选活动运行的字典。

    Raises:
        CourseNotFound: 指定课程不存在。
        ConversationError: 对话不存在或归属不符。
    """
    with catalog_lock:
        conversation = require_conversation(session, course_id, conversation_id)
        runs = {
            run.id: run
            for run in session.scalars(
                select(Run).where(Run.conversation_id == conversation_id)
            )
        }
        messages = [
            {
                "id": message.id,
                "run_id": message.run_id,
                "role": message.role,
                "content": message.content,
                "status": message.status,
                "created_at": message.created_at,
                "citations": citations_for_run(session, message.run_id)
                if message.role == "assistant"
                else [],
                "run": run_info(session, runs[message.run_id])
                if message.role == "assistant"
                else None,
            }
            for message in conversation_messages(session, conversation_id)
        ]
        active = next(
            (run_info(session, run) for run in runs.values() if run.status in ACTIVE),
            None,
        )
        return {
            **conversation_info(conversation),
            "messages": messages,
            "active_run": active,
        }


def event(session, run_id, kind, data):
    """将事件加入调用方的事务，使其随对应的业务状态一起提交。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        run_id: 问答运行 ID。
        kind: 要持久化的 SSE 事件类型。
        data: 可序列化为 JSON 的事件负载。
    """
    session.add(
        RunEvent(run_id=run_id, kind=kind, data=json.dumps(data, ensure_ascii=False))
    )


def create_run(
    session, course_id, conversation_id, question, top_k, semantic, rerank, mode="rag",
    *, web_search=False, answer_review=False,
):
    """提交问答运行、两条消息和初始事件。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        course_id: 课程 ID。
        conversation_id: 对话 ID。
        question: 用户提交的问题正文。
        top_k: 最多返回的候选片段数。
        semantic: 是否启用向量检索。
        rerank: 是否启用候选片段重排序。
        mode: 问答模式，取值为 rag 或 agent。
        web_search: 是否允许 Agent 使用网络工具。
        answer_review: 是否在发布正文前核对证据。

    Returns:
        新运行的摘要，包含队列状态和最后事件 ID。

    Raises:
        CourseNotFound: 指定课程不存在。
        ConversationError: 对话不存在或归属不符（404），或已有活动运行（409）。
    """
    with catalog_lock:
        conversation = require_conversation(session, course_id, conversation_id)
        existing = session.scalar(
            select(Run.id).where(
                Run.conversation_id == conversation_id, Run.status.in_(ACTIVE)
            )
        )
        if existing:
            raise ConversationError("这段对话正在生成回答，请等待完成或先停止。", 409)
        web_search = mode == "agent" and web_search
        run = Run(
            id=str(uuid4()),
            conversation_id=conversation_id,
            question=question,
            top_k=top_k,
            semantic=int(semantic),
            rerank=int(rerank),
            mode=mode,
            web_search=web_search,
            answer_review=answer_review,
            trace=json.dumps({"web_search": web_search, "answer_review": answer_review}),
        )
        session.add(run)
        session.flush()
        session.add(
            Message(
                conversation_id=conversation_id,
                run_id=run.id,
                role="user",
                content=question,
                status="completed",
            )
        )
        session.add(
            Message(
                conversation_id=conversation_id,
                run_id=run.id,
                role="assistant",
                content="",
                status="pending",
            )
        )
        if conversation.title == "新对话":
            conversation.title = question[:40]
        conversation.updated_at = utc_now()
        event(session, run.id, "status", {"status": "queued", "phase": "等待回答"})
        session.commit()
        return run_info(session, run)
