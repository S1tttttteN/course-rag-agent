import asyncio
import json
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.models import CHAT_MODEL, CHAT_PROVIDER, CHAT_PROVIDERS, TOP_K, ANSWER_REVIEW, WEB_SEARCH_ENABLED, OLLAMA_BASE_URL, chat_provider_conf, _get_api_key
from app.db.models import Citation, Run, RunEvent
from app.db.session import SessionLocal, get_session
from app.retrieval.reranker import is_available
from app.services import conversations

router = APIRouter(tags=["conversations"])
SessionDep = Annotated[Session, Depends(get_session)]
Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]


class ConversationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: Title = "新对话"


class ConversationRename(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: Title


class QuestionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
    top_k: int = Field(default=4, ge=1, le=10)
    semantic: bool = True
    rerank: bool = True
    mode: Literal["rag", "agent"] = "rag"


@router.get("/chat/config")
def chat_config():
    conf = CHAT_PROVIDERS.get(CHAT_PROVIDER, {})
    configured, note = True, ""
    try:
        conf = chat_provider_conf()
        _get_api_key(conf, CHAT_PROVIDER, "对话模型")
    except (RuntimeError, ValueError) as error:
        configured, note = False, str(error)
    if configured and CHAT_PROVIDER == 'ollama':
        import httpx
        try:
            response = httpx.get(OLLAMA_BASE_URL.rstrip('/') + '/api/tags', timeout=2, trust_env=False)
            response.raise_for_status()
            names = {item['name'] for item in response.json().get('models', [])}
            configured = (CHAT_MODEL or conf['default_model']) in names
            if not configured:
                note = 'Ollama未安装当前配置的模型，请核对本机模型名称。'
        except Exception:
            configured, note = False, '无法连接本机Ollama，请启动Ollama后重试。'
    return {"provider": CHAT_PROVIDER, "model": CHAT_MODEL or conf.get("default_model", ""),
            "configured": configured, "note": note, "top_k": min(10, max(1, TOP_K)), "rerank_available": is_available(), "mode": "agent", "answer_review": ANSWER_REVIEW, "web_search": WEB_SEARCH_ENABLED}


@router.get("/courses/{course_id}/conversations")
def list_conversations(course_id: UUID, session: SessionDep):
    return conversations.list_conversations(session, str(course_id))


@router.post("/courses/{course_id}/conversations", status_code=201)
def create_conversation(course_id: UUID, body: ConversationInput, session: SessionDep):
    return conversations.create(session, str(course_id), body.title)


@router.get("/courses/{course_id}/conversations/{conversation_id}")
def conversation_detail(course_id: UUID, conversation_id: UUID, session: SessionDep):
    return conversations.detail(session, str(course_id), str(conversation_id))


@router.patch("/courses/{course_id}/conversations/{conversation_id}")
def rename_conversation(course_id: UUID, conversation_id: UUID, body: ConversationRename, session: SessionDep):
    return conversations.rename(session, str(course_id), str(conversation_id), body.title)


@router.delete("/courses/{course_id}/conversations/{conversation_id}", status_code=204)
async def delete_conversation(course_id: UUID, conversation_id: UUID, request: Request, session: SessionDep):
    conversations.require_conversation(session, str(course_id), str(conversation_id))
    request.app.state.runs.cancel_scope(str(course_id), str(conversation_id))
    conversations.remove(session, str(course_id), str(conversation_id))
    return Response(status_code=204)


@router.post("/courses/{course_id}/conversations/{conversation_id}/runs", status_code=202)
async def create_run(course_id: UUID, conversation_id: UUID, body: QuestionInput, request: Request, session: SessionDep):
    if len(request.app.state.runs.tasks) >= 6:
        raise conversations.ConversationError("等待中的回答较多，请稍后再试。", 429)
    result = conversations.create_run(session, str(course_id), str(conversation_id), body.question, body.top_k, body.semantic, body.rerank, body.mode)
    request.app.state.runs.start(result["id"])
    return result


@router.post("/courses/{course_id}/conversations/{conversation_id}/runs/{run_id}/cancel")
async def cancel_run(course_id: UUID, conversation_id: UUID, run_id: UUID, request: Request, session: SessionDep):
    conversations.require_run(session, str(course_id), str(conversation_id), str(run_id))
    request.app.state.runs.cancel(str(run_id))
    session.expire_all()
    return conversations.run_info(session, session.get(Run, str(run_id)))


@router.get("/courses/{course_id}/conversations/{conversation_id}/citations/{citation_id}")
def read_citation(course_id: UUID, conversation_id: UUID, citation_id: UUID, session: SessionDep):
    conversations.require_conversation(session, str(course_id), str(conversation_id))
    citation = session.get(Citation, str(citation_id))
    if citation is None:
        raise conversations.ConversationError("参考片段不存在。", 404)
    conversations.require_run(session, str(course_id), str(conversation_id), citation.run_id)
    return conversations.citation_info(session, citation)


@router.get("/courses/{course_id}/conversations/{conversation_id}/runs/{run_id}/events")
async def run_events(course_id: UUID, conversation_id: UUID, run_id: UUID, request: Request,
                     after: Annotated[int, Query(ge=0)] = 0):
    with SessionLocal() as session:
        conversations.require_run(session, str(course_id), str(conversation_id), str(run_id))
    try:
        after = max(after, int(request.headers.get("last-event-id", "0")))
    except ValueError:
        pass

    async def events():
        cursor, ticks = after, 0
        while not await request.is_disconnected():
            with SessionLocal() as database:
                run = database.get(Run, str(run_id))
                state = run.status if run else "deleted"
                rows = [(item.id, item.kind, item.data) for item in database.scalars(select(RunEvent).where(
                    RunEvent.run_id == str(run_id), RunEvent.id > cursor,
                ).order_by(RunEvent.id).limit(100))]
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
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
