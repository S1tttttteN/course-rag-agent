"""Single controlled worker; durable version statuses are the queue."""
import logging
import shutil
from threading import Event, Thread

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
    if isinstance(error, ValueError) and step == "解析正文":
        # Only loader-authored validation messages, never provider response bodies.
        text = str(error)
        if any(text.startswith(prefix) for prefix in ("PDF 已加密", "正文超过", "Word 解压", "无法识别", "没有可读取", "资料片段", "不支持的")):
            return text
    if isinstance(error, RuntimeError) and "需要在 .env 中配置" in str(error):
        return str(error)
    return f"{step}失败（{type(error).__name__}）。请检查文件、嵌入模型配置与网络后重试。"


def process_version(version_id: str, stopped: Event):
    with catalog_lock, SessionLocal() as session:
        version = session.get(DocumentVersion, version_id)
        if version is None or version.status != "queued":
            return
        document = session.get(Document, version.document_id)
        course_id, document_id, filename = document.course_id, document.id, version.filename
        version.status, version.step, version.error = "processing", "解析正文", ""
        session.commit()
    store = None
    published = False
    step = "解析正文"
    try:
        chunks = load_and_split(raw_path(course_id, document_id, version_id, filename), filename, CHUNK_SIZE, CHUNK_OVERLAP)
        if stopped.is_set():
            return
        profile = embedding_profile()
        step = "生成向量"
        with catalog_lock, SessionLocal() as session:
            version = session.get(DocumentVersion, version_id)
            if version is None:
                return
            version.step, version.chunk_count, version.embedding_profile = step, len(chunks), profile
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
            batch = chunks[offset:offset + 32]
            vectors = embeddings.embed_documents([chunk.page_content for chunk in batch])
            if len(vectors) != len(batch):
                raise ValueError("嵌入返回数量与资料片段数量不一致")
            store.upsert(
                ids=[f"{version_id}:{offset + position}" for position in range(len(batch))],
                embeddings=vectors,
                metadatas=[{"version_id": version_id, "document_id": document_id, "course_id": course_id} for _ in batch],
            )
            with SessionLocal() as session:
                session.execute(update(DocumentVersion).where(DocumentVersion.id == version_id).values(step=f"生成向量 {min(offset + 32, len(chunks))}/{len(chunks)}"))
                session.commit()
        step = "发布索引"
        with catalog_lock, SessionLocal() as session:
            version = session.get(DocumentVersion, version_id)
            document = session.get(Document, document_id)
            if stopped.is_set() or version is None or document is None:
                return
            session.execute(delete(Chunk).where(Chunk.version_id == version_id))
            session.add_all([Chunk(
                id=f"{version_id}:{position}", version_id=version_id, ordinal=position,
                text=chunk.page_content, page=chunk.metadata.get("page"), start_index=chunk.metadata.get("start_index", 0),
            ) for position, chunk in enumerate(chunks)])
            document.active_version_id, document.updated_at = version_id, utc_now()
            version.status, version.step, version.error = "ready", "可用于检索", ""
            session.commit()
            published = True
        # Retired vectors are invisible immediately; reclaim after publication.
        for item in index.client().list_collections():
            name = item.name if hasattr(item, "name") else item
            if name.startswith(f"course_{course_id.replace('-', '')}_"):
                index.client().get_collection(name, embedding_function=None).delete(where={"$and": [
                    {"document_id": document_id}, {"version_id": {"$ne": version_id}},
                ]})
    except Exception as error:
        log.warning("Indexing version %s failed during %s (%s)", version_id, step, type(error).__name__)
        if not published:
            with catalog_lock, SessionLocal() as session:
                version = session.get(DocumentVersion, version_id)
                if version:
                    version.status, version.step, version.error = "failed", "处理失败", public_error(error, step)
                    session.commit()
    finally:
        if not published and store is not None:
            try:
                store.delete(where={"version_id": version_id})
            except Exception:
                log.warning("Unpublished vectors require startup reconciliation: %s", version_id)


def cleanup_job(job_id: str):
    with SessionLocal() as session:
        job = session.get(CleanupJob, job_id)
        if job is None:
            return
        try:
            index.remove_document(job.course_id, job.document_id)
            path = (UPLOAD_DIR / job.course_id / job.document_id) if job.document_id else (UPLOAD_DIR / job.course_id)
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
    with catalog_lock, SessionLocal() as session:
        session.execute(update(DocumentVersion).where(DocumentVersion.status == "processing").values(status="queued", step="服务重启，等待恢复"))
        active = set(session.scalars(select(Document.active_version_id).where(Document.active_version_id.is_not(None))))
        session.commit()
    index.reconcile(active)
    # Remove upload files written before an interrupted transaction committed.
    with catalog_lock, SessionLocal() as session:
        valid = {version.id for version in session.scalars(select(DocumentVersion))}
        if UPLOAD_DIR.exists():
            for path in UPLOAD_DIR.glob("*/*/*"):
                if path.is_file() and path.stem not in valid:
                    path.unlink()


class IndexWorker:
    def __init__(self):
        self.stopped = Event()
        self.wake = Event()
        self.thread = Thread(target=self.run, name="course-indexer", daemon=True)

    def start(self):
        self.thread.start()

    def run(self):
        try:
            recover()
        except Exception:
            log.warning("Startup reconciliation failed; trying again before processing jobs")
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
                    version_id = session.scalar(select(DocumentVersion.id).where(DocumentVersion.status == "queued").order_by(DocumentVersion.created_at).limit(1))
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
        self.stopped.set()
        self.wake.set()
