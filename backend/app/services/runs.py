"""Transactional run state and evidence publication, independent of HTTP."""
import json
import re
import hashlib
from datetime import datetime, timezone

from sqlalchemy import select

from app.core.locks import catalog_lock
from app.db.models import Chunk, Citation, Conversation, Document, DocumentVersion, Message, Run, utc_now
from app.db.session import SessionLocal
from app.services.conversations import ACTIVE, citations_for_run, event


def tool_update(run_id, record):
    """Publish a durable, bounded summary, never model reasoning or raw tool output."""
    with catalog_lock, SessionLocal() as session:
        run = session.get(Run, run_id)
        if run is None or run.status not in ACTIVE:
            return False
        trace = json.loads(run.trace)
        tools = trace.setdefault("tools", [])
        previous = next((item for item in tools if item["id"] == record["id"]), None)
        if previous is None:
            tools.append(record)
        else:
            previous.update(record)
        run.trace = json.dumps(trace, ensure_ascii=False)
        event(session, run_id, "tool", record)
        session.commit()
        return True


def status(run_id, phase, trace=None, query=None):
    with catalog_lock, SessionLocal() as session:
        run = session.get(Run, run_id)
        if run is None or run.status not in {"queued", "running"}:
            return False
        run.status, run.phase = "running", phase
        if trace is not None:
            run.trace = json.dumps({**json.loads(run.trace), **trace}, ensure_ascii=False)
        if query is not None:
            run.query = query
        event(session, run_id, "status", {"status": run.status, "phase": phase})
        session.commit()
        return True


def patch_trace(run_id, values):
    with catalog_lock, SessionLocal() as session:
        run = session.get(Run, run_id)
        if run is not None and run.status in ACTIVE:
            run.trace = json.dumps({**json.loads(run.trace), **values}, ensure_ascii=False)
            session.commit()


def snapshot(run_id, records):
    with catalog_lock, SessionLocal() as session:
        run = session.get(Run, run_id)
        if run is None or run.status not in {"queued", "running"}:
            return []
        conversation = session.get(Conversation, run.conversation_id)
        seen = {citation.chunk_id for citation in session.scalars(select(Citation).where(Citation.run_id == run_id))}
        for row in records:
            chunk = session.get(Chunk, row["id"])
            document = session.get(Document, row["document_id"])
            if chunk is None or document is None or document.course_id != conversation.course_id:
                continue
            if document.active_version_id != chunk.version_id or chunk.id in seen:
                continue
            version = session.get(DocumentVersion, chunk.version_id)
            if version is None or version.document_id != document.id or version.status != "ready":
                continue
            seen.add(chunk.id)
            session.add(Citation(
                run_id=run_id, number=len(seen), document_id=document.id,
                version_id=version.id, chunk_id=chunk.id, source=version.filename, version=version.number,
                text=chunk.text, page=chunk.page, start_index=chunk.start_index,
            ))
        session.flush()
        sources = citations_for_run(session, run_id)
        event(session, run_id, "citations", {"citations": sources})
        session.commit()
        return sources


def snapshot_web(run_id, records):
    """Snapshot actual search/fetch output. Never accept citations supplied by the model."""
    with catalog_lock, SessionLocal() as session:
        run = session.get(Run, run_id)
        if run is None or run.status not in {"queued", "running"}:
            return []
        existing = list(session.scalars(select(Citation).where(Citation.run_id == run_id)))
        seen = {item.chunk_id for item in existing}
        for row in records:
            key = 'web:' + hashlib.sha256((row['url'] + row['kind']).encode()).hexdigest()[:48]
            if key in seen:
                continue
            seen.add(key)
            session.add(Citation(run_id=run_id, number=len(seen), kind=row['kind'], url=row['url'],
                retrieved_at=utc_now(), document_id='', version_id='', chunk_id=key,
                source=row['title'][:255], version=0, text=row['text'], page=None, start_index=0))
        session.flush()
        sources = citations_for_run(session, run_id)
        event(session, run_id, 'citations', {'citations': sources})
        session.commit()
        return sources


def append(run_id, text):
    if not text:
        return True
    with catalog_lock, SessionLocal() as session:
        run = session.get(Run, run_id)
        if run is None or run.status not in ACTIVE:
            return False
        message = session.scalar(select(Message).where(Message.run_id == run_id, Message.role == "assistant"))
        message.content += text
        message.status = "running"
        event(session, run_id, "delta", {"text": text})
        session.commit()
        return True


def finish(run_id, state, error=""):
    with catalog_lock, SessionLocal() as session:
        run = session.get(Run, run_id)
        if run is None or run.status not in ACTIVE:
            return
        if run.status == "cancelling" and state == "completed":
            state = "cancelled"
        run.status = state
        run.phase = {"completed": "回答完成", "cancelled": "已停止", "interrupted": "服务中断", "failed": "回答失败"}[state]
        run.error = error
        message = session.scalar(select(Message).where(Message.run_id == run_id, Message.role == "assistant"))
        message.status = state
        known = {source["number"] for source in citations_for_run(session, run_id)}
        unknown = sorted({int(number) for number in re.findall(r"(?<![\w])\[(\d{1,4})\]", message.content)} - known)
        trace = json.loads(run.trace)
        trace["elapsed_seconds"] = round((datetime.now(timezone.utc) - datetime.fromisoformat(run.created_at)).total_seconds(), 3)
        if trace.get("evidence_check", {}).get("status") == "checking":
            trace["evidence_check"]["status"] = "interrupted"
        for tool in trace.get("tools", []):
            if tool["status"] == "running":
                tool["status"] = "interrupted"
                tool["summary"] = "执行已中断；未发布新结果。"
                event(session, run_id, "tool", tool)
        if unknown:
            trace["unknown_citations"] = unknown
        run.trace = json.dumps(trace, ensure_ascii=False)
        conversation = session.get(Conversation, run.conversation_id)
        conversation.updated_at = utc_now()
        event(session, run_id, "done", {"status": state, "phase": run.phase, "error": error})
        session.commit()


def recover():
    with SessionLocal() as session:
        ids = list(session.scalars(select(Run.id).where(Run.status.in_(ACTIVE))))
    for run_id in ids:
        finish(run_id, "interrupted", "服务重启，回答已中断。已保存部分内容，请重新发送问题。")
