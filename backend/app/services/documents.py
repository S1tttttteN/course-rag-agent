"""Versioned uploads. Publication happens only in the indexing worker."""
import hashlib
from pathlib import Path
from uuid import uuid4

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.config import MAX_UPLOAD_BYTES, UPLOAD_DIR
from app.core.locks import catalog_lock
from app.db.models import Chunk, CleanupJob, Course, Document, DocumentVersion, utc_now
from app.repositories.documents import documents_for_course, versions_for_document
from app.retrieval.loader import SUPPORTED_EXTENSIONS
from app.services.courses import CourseNotFound


class DocumentError(Exception):
    def __init__(self, detail: str, status: int = 400):
        self.detail, self.status = detail, status


def require_course(session: Session, course_id: str):
    if session.get(Course, course_id) is None:
        raise CourseNotFound(course_id)


def require_document(session: Session, course_id: str, document_id: str):
    require_course(session, course_id)
    document = session.get(Document, document_id)
    if document is None or document.course_id != course_id:
        raise DocumentError("资料不存在或不属于当前课程。", 404)
    return document


def raw_path(course_id: str, document_id: str, version_id: str, filename: str) -> Path:
    return UPLOAD_DIR / course_id / document_id / f"{version_id}{Path(filename).suffix.lower()}"


def version_info(version: DocumentVersion):
    return {key: getattr(version, key) for key in (
        "id", "number", "filename", "size", "status", "step", "error", "chunk_count", "embedding_profile", "created_at",
    )}


def document_info(session: Session, document: Document):
    versions = versions_for_document(session, document.id)
    active = next((version for version in versions if version.id == document.active_version_id), None)
    return {
        "id": document.id, "name": document.name, "course_id": document.course_id,
        "created_at": document.created_at, "updated_at": document.updated_at,
        "active_version_id": document.active_version_id,
        "active_version_number": active.number if active else None,
        "chunk_count": active.chunk_count if active else 0,
        "usable": active is not None and active.status == "ready",
        "latest": version_info(versions[0]),
    }


def list_documents(session: Session, course_id: str):
    require_course(session, course_id)
    return [document_info(session, document) for document in documents_for_course(session, course_id)]


def upload(session: Session, course_id: str, filename: str, data: bytes, document_id: str | None = None):
    name = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not name or len(name) > 255 or Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise DocumentError("请选择 TXT、Markdown、HTML、DOCX 或可提取文字的 PDF，文件名不超过 255 字。")
    if not data:
        raise DocumentError("文件为空，请选择含有正文的资料。")
    if len(data) > MAX_UPLOAD_BYTES:
        raise DocumentError("单个文件不能超过 20 MB。", 413)
    with catalog_lock:
        require_course(session, course_id)
        if document_id:
            document = require_document(session, course_id, document_id)
            versions = versions_for_document(session, document.id)
            if any(version.status in {"queued", "processing"} for version in versions):
                raise DocumentError("这份资料正在处理，请完成后再替换。", 409)
            number = versions[0].number + 1
        else:
            existing = session.scalar(select(Document).where(Document.course_id == course_id, Document.name == name))
            if existing:
                raise DocumentError("当前课程已有同名资料。请使用该资料的「替换」操作。", 409)
            document = Document(id=str(uuid4()), course_id=course_id, name=name, updated_at=utc_now())
            session.add(document)
            session.flush()
            number = 1
        version = DocumentVersion(
            id=str(uuid4()), document_id=document.id, number=number, filename=name,
            content_hash=hashlib.sha256(data).hexdigest(), size=len(data), status="queued", step="等待处理",
        )
        path = raw_path(course_id, document.id, version.id, name)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            document.updated_at = utc_now()
            session.add(version)
            session.commit()
        except Exception:
            session.rollback()
            path.unlink(missing_ok=True)
            raise
        return document_info(session, document)


def retry(session: Session, course_id: str, document_id: str):
    with catalog_lock:
        document = require_document(session, course_id, document_id)
        version = versions_for_document(session, document.id)[0]
        if version.status != "failed":
            raise DocumentError("只有失败的最新版本可以重试。", 409)
        if not raw_path(course_id, document.id, version.id, version.filename).is_file():
            raise DocumentError("原文件已缺失，请重新上传替换。", 409)
        session.execute(delete(Chunk).where(Chunk.version_id == version.id))
        version.status, version.step, version.error = "queued", "等待重试", ""
        session.commit()
        return document_info(session, document)


def remove(session: Session, course_id: str, document_id: str):
    with catalog_lock:
        document = require_document(session, course_id, document_id)
        session.add(CleanupJob(course_id=course_id, document_id=document_id))
        session.delete(document)
        session.commit()


def preview(session: Session, course_id: str, document_id: str, offset: int, limit: int):
    document = require_document(session, course_id, document_id)
    if not document.active_version_id:
        raise DocumentError("这份资料还没有可用版本，请等待处理完成或重试。", 409)
    version = session.get(DocumentVersion, document.active_version_id)
    chunks = list(session.scalars(select(Chunk).where(Chunk.version_id == version.id).order_by(Chunk.ordinal).offset(offset).limit(limit)))
    return {"source": version.filename, "version": version.number, "total": version.chunk_count, "offset": offset,
            "chunks": [{"id": chunk.id, "text": chunk.text, "page": chunk.page, "ordinal": chunk.ordinal + 1} for chunk in chunks]}
