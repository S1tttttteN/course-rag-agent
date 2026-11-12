"""维护问答状态和引用，将消息变更与供 SSE 回放的事件一起提交。"""

import json
import re
import hashlib
from datetime import datetime, timezone

from sqlalchemy import select

from app.core.locks import catalog_lock
from app.db.models import (
    Chunk,
    Citation,
    Conversation,
    Document,
    DocumentVersion,
    Message,
    Run,
    utc_now,
)
from app.db.session import SessionLocal
from app.services.conversations import ACTIVE, citations_for_run, event


def tool_update(run_id, record):
    """持久化调用方提供的工具摘要，供界面展示调用进度。

    Args:
        run_id: 问答运行 ID。
        record: 包含调用 ID 和当前状态的工具摘要。

    Returns:
        已保存更新时为 True，运行缺失或已终止时为 False。
    """
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
    """将排队中或运行中的问答标为运行中并记录阶段。

    Args:
        run_id: 问答运行 ID。
        phase: 客户端显示的阶段说明。
        trace: 要浅合并到执行摘要的可选字典。
        query: 要保存的可选实际检索查询。

    Returns:
        成功保存时为 True，运行缺失或状态不允许更新时为 False。
    """
    with catalog_lock, SessionLocal() as session:
        run = session.get(Run, run_id)
        if run is None or run.status not in {"queued", "running"}:
            return False
        run.status, run.phase = "running", phase
        if trace is not None:
            run.trace = json.dumps(
                {**json.loads(run.trace), **trace}, ensure_ascii=False
            )
        if query is not None:
            run.query = query
        event(session, run_id, "status", {"status": run.status, "phase": phase})
        session.commit()
        return True


def patch_trace(run_id, values):
    """将摘要字段浅合并到活动运行。

    Args:
        run_id: 问答运行 ID。
        values: 要写入或覆盖的执行摘要字段。
    """
    with catalog_lock, SessionLocal() as session:
        run = session.get(Run, run_id)
        if run is not None and run.status in ACTIVE:
            run.trace = json.dumps(
                {**json.loads(run.trace), **values}, ensure_ascii=False
            )
            session.commit()


def snapshot(run_id, records):
    """复核候选片段仍属于当前课程的可用版本，保存引用快照并返回本轮引用。

    Args:
        run_id: 问答运行 ID。
        records: 待复核的课程片段候选记录。

    Returns:
        本轮保存的引用列表，运行不可写入时为空。
    """
    with catalog_lock, SessionLocal() as session:
        run = session.get(Run, run_id)
        if run is None or run.status not in {"queued", "running"}:
            return []
        conversation = session.get(Conversation, run.conversation_id)
        seen = {
            citation.chunk_id
            for citation in session.scalars(
                select(Citation).where(Citation.run_id == run_id)
            )
        }
        for row in records:
            chunk = session.get(Chunk, row["id"])
            document = session.get(Document, row["document_id"])
            if (
                chunk is None
                or document is None
                or document.course_id != conversation.course_id
            ):
                continue
            if document.active_version_id != chunk.version_id or chunk.id in seen:
                continue
            version = session.get(DocumentVersion, chunk.version_id)
            if (
                version is None
                or version.document_id != document.id
                or version.status != "ready"
            ):
                continue
            seen.add(chunk.id)
            session.add(
                Citation(
                    run_id=run_id,
                    number=len(seen),
                    document_id=document.id,
                    version_id=version.id,
                    chunk_id=chunk.id,
                    source=version.filename,
                    version=version.number,
                    text=chunk.text,
                    page=chunk.page,
                    start_index=chunk.start_index,
                )
            )
        session.flush()
        sources = citations_for_run(session, run_id)
        event(session, run_id, "citations", {"citations": sources})
        session.commit()
        return sources


def snapshot_web(run_id, records):
    """把搜索或抓取工具实际返回的内容保存为本轮引用快照。

    Args:
        run_id: 问答运行 ID。
        records: 包含地址、类型、标题和文本的网络来源记录。

    Returns:
        本轮保存的引用列表，运行不可写入时为空。
    """
    with catalog_lock, SessionLocal() as session:
        run = session.get(Run, run_id)
        if run is None or run.status not in {"queued", "running"}:
            return []
        existing = list(
            session.scalars(select(Citation).where(Citation.run_id == run_id))
        )
        seen = {item.chunk_id for item in existing}
        for row in records:
            key = (
                "web:"
                + hashlib.sha256((row["url"] + row["kind"]).encode()).hexdigest()[:48]
            )
            if key in seen:
                continue
            seen.add(key)
            session.add(
                Citation(
                    run_id=run_id,
                    number=len(seen),
                    kind=row["kind"],
                    url=row["url"],
                    retrieved_at=utc_now(),
                    document_id="",
                    version_id="",
                    chunk_id=key,
                    source=row["title"][:255],
                    version=0,
                    text=row["text"],
                    page=None,
                    start_index=0,
                )
            )
        session.flush()
        sources = citations_for_run(session, run_id)
        event(session, run_id, "citations", {"citations": sources})
        session.commit()
        return sources


def append(run_id, text):
    """保存回答正文增量及对应的 SSE 事件。

    Args:
        run_id: 问答运行 ID。
        text: 要追加到助手消息的已发布文本。

    Returns:
        增量已保存或为空时为 True，运行缺失或已终止时为 False。
    """
    if not text:
        return True
    with catalog_lock, SessionLocal() as session:
        run = session.get(Run, run_id)
        if run is None or run.status not in ACTIVE:
            return False
        message = session.scalar(
            select(Message).where(Message.run_id == run_id, Message.role == "assistant")
        )
        message.content += text
        message.status = "running"
        event(session, run_id, "delta", {"text": text})
        session.commit()
        return True


def finish(run_id, state, error=""):
    """提交运行和消息的终态与结束事件，重复结束已终止的运行时直接返回。

    Args:
        run_id: 问答运行 ID。
        state: 预期的终态，支持 completed、cancelled、interrupted 和 failed。
        error: 要记录的失败或中断说明。
    """
    with catalog_lock, SessionLocal() as session:
        run = session.get(Run, run_id)
        if run is None or run.status not in ACTIVE:
            return
        if run.status == "cancelling" and state == "completed":
            state = "cancelled"
        run.status = state
        run.phase = {
            "completed": "回答完成",
            "cancelled": "已停止",
            "interrupted": "服务中断",
            "failed": "回答失败",
        }[state]
        run.error = error
        message = session.scalar(
            select(Message).where(Message.run_id == run_id, Message.role == "assistant")
        )
        message.status = state
        known = {source["number"] for source in citations_for_run(session, run_id)}
        unknown = sorted(
            {
                int(number)
                for number in re.findall(r"(?<![\w])\[(\d{1,4})\]", message.content)
            }
            - known
        )
        trace = json.loads(run.trace)
        trace["elapsed_seconds"] = round(
            (
                datetime.now(timezone.utc) - datetime.fromisoformat(run.created_at)
            ).total_seconds(),
            3,
        )
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
        event(
            session,
            run_id,
            "done",
            {"status": state, "phase": run.phase, "error": error},
        )
        session.commit()


def recover():
    """将重启前未结束的问答标记为中断，并保留已发布内容。"""
    with SessionLocal() as session:
        ids = list(session.scalars(select(Run.id).where(Run.status.in_(ACTIVE))))
    for run_id in ids:
        finish(
            run_id,
            "interrupted",
            "服务重启，回答已中断。已保存部分内容，请重新发送问题。",
        )
