"""在单个后台线程中建索引和清理资料，使用 SQLite 中的版本状态作为持久化队列。"""

import logging
import shutil
from threading import Event, Thread

from openai import APIStatusError
from sqlalchemy import delete, select, update

from app.core.config import UPLOAD_DIR
from app.core.locks import catalog_lock
from app.core.models import CHUNK_OVERLAP, CHUNK_SIZE, embedding_profile, get_embeddings
from app.db.models import Chunk, CleanupJob, Document, DocumentVersion, utc_now
from app.db.session import SessionLocal
from app.retrieval import index
from app.retrieval.loader import load_and_split
from app.services.documents import raw_path

log = logging.getLogger(__name__)


def public_error(error: Exception, step: str) -> str:
    """将索引失败转换为可向客户端展示的说明。

    Args:
        error: 当前操作产生的异常。
        step: 发生错误时的处理阶段。

    Returns:
        精简后的公开错误说明。
    """
    if isinstance(error, ValueError) and step == "解析正文":
        # 只向界面透出已知的文件校验提示，避免暴露模型服务的响应正文。
        text = str(error)
        if any(
            text.startswith(prefix)
            for prefix in (
                "PDF 已加密",
                "正文超过",
                "Word 解压",
                "无法识别",
                "没有可读取",
                "资料片段",
                "不支持的",
            )
        ):
            return text
    if isinstance(error, RuntimeError) and "需要在 .env 中配置" in str(error):
        return str(error)
    if step == "生成向量" and isinstance(error, APIStatusError):
        if error.status_code == 402:
            return "生成向量失败（HTTP 402）。嵌入服务余额不足或额度不可用，请检查账户后重试。"
        return f"生成向量失败（HTTP {error.status_code}）。请检查嵌入模型配置与服务状态后重试。"
    return (
        f"{step}失败（{type(error).__name__}）。请检查文件、嵌入模型配置与网络后重试。"
    )


def process_version(version_id: str, stopped: Event):
    """处理一个待索引版本，写入全部向量后再提交原文片段并发布版本。

    发布前失败时记录失败状态并尝试清理本版本向量。停机中断的任务由启动恢复流程重新排队。

    Args:
        version_id: 资料版本 ID。
        stopped: 工作线程的停止信号，用于中止处理中版本。
    """
    with catalog_lock, SessionLocal() as session:
        version = session.get(DocumentVersion, version_id)
        if version is None or version.status != "queued":
            return
        document = session.get(Document, version.document_id)
        course_id, document_id, filename = (
            document.course_id,
            document.id,
            version.filename,
        )
        version.status, version.step, version.error = "processing", "解析正文", ""
        session.commit()
    store = None
    published = False
    step = "解析正文"
    try:
        chunks = load_and_split(
            raw_path(course_id, document_id, version_id, filename),
            filename,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
        )
        if stopped.is_set():
            return
        profile = embedding_profile()
        step = "生成向量"
        with catalog_lock, SessionLocal() as session:
            version = session.get(DocumentVersion, version_id)
            if version is None:
                return
            version.step, version.chunk_count, version.embedding_profile = (
                step,
                len(chunks),
                profile,
            )
            session.commit()
        embeddings = get_embeddings()
        store = index.collection(course_id, profile)
        store.delete(where={"version_id": version_id})
        for offset in range(0, len(chunks), 32):
            if stopped.is_set():
                return
            with SessionLocal() as session:
                if session.get(DocumentVersion, version_id) is None:
                    return
            batch = chunks[offset : offset + 32]
            vectors = embeddings.embed_documents(
                [chunk.page_content for chunk in batch]
            )
            if len(vectors) != len(batch):
                raise ValueError("嵌入返回数量与资料片段数量不一致")
            store.upsert(
                ids=[
                    f"{version_id}:{offset + position}"
                    for position in range(len(batch))
                ],
                embeddings=vectors,
                metadatas=[
                    {
                        "version_id": version_id,
                        "document_id": document_id,
                        "course_id": course_id,
                    }
                    for _ in batch
                ],
            )
            with SessionLocal() as session:
                session.execute(
                    update(DocumentVersion)
                    .where(DocumentVersion.id == version_id)
                    .values(
                        step=f"生成向量 {min(offset + 32, len(chunks))}/{len(chunks)}"
                    )
                )
                session.commit()
        step = "发布索引"
        # 向量写齐后才切换可用版本，避免检索读到部分索引，也让替换失败时仍能使用旧版本。
        with catalog_lock, SessionLocal() as session:
            version = session.get(DocumentVersion, version_id)
            document = session.get(Document, document_id)
            if stopped.is_set() or version is None or document is None:
                return
            session.execute(delete(Chunk).where(Chunk.version_id == version_id))
            session.add_all(
                [
                    Chunk(
                        id=f"{version_id}:{position}",
                        version_id=version_id,
                        ordinal=position,
                        text=chunk.page_content,
                        page=chunk.metadata.get("page"),
                        start_index=chunk.metadata.get("start_index", 0),
                    )
                    for position, chunk in enumerate(chunks)
                ]
            )
            document.active_version_id, document.updated_at = version_id, utc_now()
            version.status, version.step, version.error = "ready", "可用于检索", ""
            session.commit()
            published = True
        # 切换后旧向量已不可见，可以在发布事务结束后回收。
        for item in index.client().list_collections():
            name = item.name if hasattr(item, "name") else item
            if name.startswith(f"course_{course_id.replace('-', '')}_"):
                index.client().get_collection(name, embedding_function=None).delete(
                    where={
                        "$and": [
                            {"document_id": document_id},
                            {"version_id": {"$ne": version_id}},
                        ]
                    }
                )
    except Exception as error:
        log.warning(
            "Indexing version %s failed during %s (%s)",
            version_id,
            step,
            type(error).__name__,
        )
        if not published:
            with catalog_lock, SessionLocal() as session:
                version = session.get(DocumentVersion, version_id)
                if version:
                    version.status, version.step, version.error = (
                        "failed",
                        "处理失败",
                        public_error(error, step),
                    )
                    session.commit()
    finally:
        if not published and store is not None:
            try:
                store.delete(where={"version_id": version_id})
            except Exception:
                log.warning(
                    "Unpublished vectors require startup reconciliation: %s", version_id
                )


def cleanup_job(job_id: str):
    """回收已删除资料的向量和原件，失败时保留任务供后台重试。

    Args:
        job_id: 清理任务 ID。
    """
    with SessionLocal() as session:
        job = session.get(CleanupJob, job_id)
        if job is None:
            return
        try:
            index.remove_document(job.course_id, job.document_id)
            path = (
                (UPLOAD_DIR / job.course_id / job.document_id)
                if job.document_id
                else (UPLOAD_DIR / job.course_id)
            )
            resolved = path.resolve()
            if UPLOAD_DIR.resolve() not in resolved.parents:
                raise ValueError("清理路径超出资料目录")
            if resolved.exists():
                shutil.rmtree(resolved)
            session.delete(job)
        except Exception as error:
            job.attempts += 1
            job.error = f"清理暂未完成（{type(error).__name__}），将在后台重试。"
            log.warning("Cleanup requires retry: %s", job_id)
        session.commit()


def recover():
    """重新排队上次停机时仍在处理的版本，并清理未发布的向量与孤立原件。"""
    with catalog_lock, SessionLocal() as session:
        session.execute(
            update(DocumentVersion)
            .where(DocumentVersion.status == "processing")
            .values(status="queued", step="服务重启，等待恢复")
        )
        active = set(
            session.scalars(
                select(Document.active_version_id).where(
                    Document.active_version_id.is_not(None)
                )
            )
        )
        session.commit()
    index.reconcile(active)
    # 上传可能在原件写入后、版本提交前中断，此时文件没有对应的版本记录。
    with catalog_lock, SessionLocal() as session:
        valid = {version.id for version in session.scalars(select(DocumentVersion))}
        if UPLOAD_DIR.exists():
            for path in UPLOAD_DIR.glob("*/*/*"):
                if path.is_file() and path.stem not in valid:
                    path.unlink()


class IndexWorker:
    """启动时先校对存储，再轮询清理任务和待索引版本。

    Attributes:
        stopped: 工作线程的停止信号。
        wake: 唤醒轮询线程的信号。
        thread: 执行校对、清理和索引的守护线程。
    """

    def __init__(self):
        """初始化后台线程和控制信号。"""
        self.stopped = Event()
        self.wake = Event()
        self.thread = Thread(target=self.run, name="course-indexer", daemon=True)

    def start(self):
        """启动索引守护线程。"""
        self.thread.start()

    def run(self):
        """校对存储，然后循环处理清理任务和最早排队的版本。"""
        try:
            recover()
        except Exception:
            log.warning(
                "Startup reconciliation failed; trying again before processing jobs"
            )
            while not self.stopped.wait(5):
                try:
                    recover()
                    break
                except Exception:
                    continue
            if self.stopped.is_set():
                return
        while not self.stopped.is_set():
            try:
                with SessionLocal() as session:
                    jobs = list(session.scalars(select(CleanupJob.id).limit(10)))
                    version_id = session.scalar(
                        select(DocumentVersion.id)
                        .where(DocumentVersion.status == "queued")
                        .order_by(DocumentVersion.created_at)
                        .limit(1)
                    )
                for job_id in jobs:
                    cleanup_job(job_id)
                if version_id:
                    process_version(version_id, self.stopped)
                    continue
            except Exception:
                log.exception("Index worker will retry after an unexpected failure")
            self.wake.wait(2)
            self.wake.clear()

    def stop(self):
        """发出停止信号并唤醒等待中的线程。"""
        self.stopped.set()
        self.wake.set()
