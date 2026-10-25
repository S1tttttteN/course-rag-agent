import json
from uuid import uuid4

from sqlalchemy import func, select

from app.core.locks import catalog_lock
from app.db.models import Citation, Conversation, Document, Message, Run, RunEvent, utc_now
from app.repositories.conversations import conversation_messages, course_conversations
from app.services.documents import require_course

ACTIVE = {"queued", "running", "cancelling"}


class ConversationError(Exception):
    def __init__(self, detail, status=400):
        self.detail, self.status = detail, status


def require_conversation(session, course_id, conversation_id):
    require_course(session, course_id)
    conversation = session.get(Conversation, conversation_id)
    if conversation is None or conversation.course_id != course_id:
        raise ConversationError("对话不存在或不属于当前课程。", 404)
    return conversation


def require_run(session, course_id, conversation_id, run_id):
    require_conversation(session, course_id, conversation_id)
    run = session.get(Run, run_id)
    if run is None or run.conversation_id != conversation_id:
        raise ConversationError("本次回答不存在或不属于当前对话。", 404)
    return run


def conversation_info(conversation):
    return {key: getattr(conversation, key) for key in ("id", "course_id", "title", "created_at", "updated_at")}


def list_conversations(session, course_id):
    require_course(session, course_id)
    return [conversation_info(item) for item in course_conversations(session, course_id)]


def create(session, course_id, title):
    with catalog_lock:
        require_course(session, course_id)
        item = Conversation(course_id=course_id, title=title)
        session.add(item)
        session.commit()
        return conversation_info(item)


def rename(session, course_id, conversation_id, title):
    with catalog_lock:
        item = require_conversation(session, course_id, conversation_id)
        item.title, item.updated_at = title, utc_now()
        session.commit()
        return conversation_info(item)


def remove(session, course_id, conversation_id):
    with catalog_lock:
        session.delete(require_conversation(session, course_id, conversation_id))
        session.commit()


def citation_info(session, citation):
    document = session.get(Document, citation.document_id)
    state = "external" if citation.kind != 'course' else "deleted" if document is None else "current" if document.active_version_id == citation.version_id else "updated"
    return {**{key: getattr(citation, key) for key in (
        "id", "number", "kind", "url", "retrieved_at", "document_id", "version_id", "chunk_id", "source", "version", "text", "page", "start_index",
    )}, "source_state": state}


def citations_for_run(session, run_id):
    return [citation_info(session, citation) for citation in session.scalars(select(Citation).where(Citation.run_id == run_id).order_by(Citation.number))]


def run_info(session, run):
    cursor = session.scalar(select(func.max(RunEvent.id)).where(RunEvent.run_id == run.id)) or 0
    return {**{key: getattr(run, key) for key in ("id", "conversation_id", "question", "query", "mode", "status", "phase", "error", "created_at")},
            "trace": json.loads(run.trace), "last_event_id": cursor}


def detail(session, course_id, conversation_id):
    # All run writes use this lock so body snapshots and event cursors agree.
    with catalog_lock:
        conversation = require_conversation(session, course_id, conversation_id)
        runs = {run.id: run for run in session.scalars(select(Run).where(Run.conversation_id == conversation_id))}
        messages = [{
            "id": message.id, "run_id": message.run_id, "role": message.role,
            "content": message.content, "status": message.status, "created_at": message.created_at,
            "citations": citations_for_run(session, message.run_id) if message.role == "assistant" else [],
            "run": run_info(session, runs[message.run_id]) if message.role == "assistant" else None,
        } for message in conversation_messages(session, conversation_id)]
        active = next((run_info(session, run) for run in runs.values() if run.status in ACTIVE), None)
        return {**conversation_info(conversation), "messages": messages, "active_run": active}


def event(session, run_id, kind, data):
    session.add(RunEvent(run_id=run_id, kind=kind, data=json.dumps(data, ensure_ascii=False)))


def create_run(session, course_id, conversation_id, question, top_k, semantic, rerank, mode="rag"):
    with catalog_lock:
        conversation = require_conversation(session, course_id, conversation_id)
        existing = session.scalar(select(Run.id).where(Run.conversation_id == conversation_id, Run.status.in_(ACTIVE)))
        if existing:
            raise ConversationError("这段对话正在生成回答，请等待完成或先停止。", 409)
        run = Run(id=str(uuid4()), conversation_id=conversation_id, question=question, top_k=top_k, semantic=int(semantic), rerank=int(rerank), mode=mode)
        session.add(run)
        session.flush()
        session.add(Message(conversation_id=conversation_id, run_id=run.id, role="user", content=question, status="completed"))
        session.add(Message(conversation_id=conversation_id, run_id=run.id, role="assistant", content="", status="pending"))
        if conversation.title == "新对话":
            conversation.title = question[:40]
        conversation.updated_at = utc_now()
        event(session, run.id, "status", {"status": "queued", "phase": "等待回答"})
        session.commit()
        return run_info(session, run)
