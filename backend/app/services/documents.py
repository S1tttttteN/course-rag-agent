"""管理课程资料与版本。上传保存原件并登记待处理版本，索引线程负责发布。"""

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
    """携带客户端错误信息的资料业务异常。

    Attributes:
        detail: 客户端可见的错误说明。
        status: 对应的 HTTP 状态码。
    """

    def __init__(self, detail: str, status: int = 400):
        """保存资料业务错误说明和状态码。

        Args:
            detail: 面向客户端的错误说明。
            status: 对应的 HTTP 状态码。
        """
        self.detail, self.status = detail, status


def require_course(session: Session, course_id: str):
    """确认课程仍存在于目录中。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        course_id: 课程 ID。

    Raises:
        CourseNotFound: 指定课程不存在。
    """
    if session.get(Course, course_id) is None:
        raise CourseNotFound(course_id)


def require_document(session: Session, course_id: str, document_id: str):
    """确认资料属于指定课程。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        course_id: 课程 ID。
        document_id: 资料 ID。

    Returns:
        通过归属校验的资料记录。

    Raises:
        CourseNotFound: 指定课程不存在。
        DocumentError: 资料不存在或归属不符。
    """
    require_course(session, course_id)
    document = session.get(Document, document_id)
    if document is None or document.course_id != course_id:
        raise DocumentError("资料不存在或不属于当前课程。", 404)
    return document


def raw_path(course_id: str, document_id: str, version_id: str, filename: str) -> Path:
    """构造版本原件在上传目录中的路径。

    Args:
        course_id: 课程 ID。
        document_id: 资料 ID。
        version_id: 资料版本 ID。
        filename: 用于取得扩展名的原件名称。

    Returns:
        按课程和资料分目录、以版本 ID 命名的文件路径。
    """
    return (
        UPLOAD_DIR
        / course_id
        / document_id
        / f"{version_id}{Path(filename).suffix.lower()}"
    )


def version_info(version: DocumentVersion):
    """提取可公开的资料版本信息。

    Args:
        version: 要转换的版本 ORM 记录。

    Returns:
        包含文件信息、索引状态和片段数量的字典。
    """
    return {
        key: getattr(version, key)
        for key in (
            "id",
            "number",
            "filename",
            "size",
            "status",
            "step",
            "error",
            "chunk_count",
            "embedding_profile",
            "created_at",
        )
    }


def document_info(session: Session, document: Document):
    """汇总资料的最新版本与已发布版本。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        document: 要汇总的资料 ORM 记录。

    Returns:
        包含资料元数据、最新版本及当前版本可用性的字典。
    """
    versions = versions_for_document(session, document.id)
    active = next(
        (version for version in versions if version.id == document.active_version_id),
        None,
    )
    return {
        "id": document.id,
        "name": document.name,
        "course_id": document.course_id,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
        "active_version_id": document.active_version_id,
        "active_version_number": active.number if active else None,
        "chunk_count": active.chunk_count if active else 0,
        "usable": active is not None and active.status == "ready",
        "latest": version_info(versions[0]),
    }


def list_documents(session: Session, course_id: str):
    """读取课程资料及其版本状态。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        course_id: 课程 ID。

    Returns:
        按资料创建时间降序排列的摘要列表。

    Raises:
        CourseNotFound: 指定课程不存在。
    """
    require_course(session, course_id)
    return [
        document_info(session, document)
        for document in documents_for_course(session, course_id)
    ]


def upload(
    session: Session,
    course_id: str,
    filename: str,
    data: bytes,
    document_id: str | None = None,
):
    """保存原件并提交待索引版本。

    替换现有资料时，原可用版本继续提供检索。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        course_id: 课程 ID。
        filename: 上传文件的原始名称。
        data: 上传文件的原始字节。
        document_id: 要替换的资料 ID，省略时创建新资料。

    Returns:
        包含最新版本与可用版本状态的资料摘要。

    Raises:
        CourseNotFound: 指定课程不存在。
        DocumentError: 原件不符合上传要求、资料归属不符或版本操作冲突。
    """
    name = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if (
        not name
        or len(name) > 255
        or Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS
    ):
        raise DocumentError(
            "请选择 TXT、Markdown、HTML、DOCX 或可提取文字的 PDF，文件名不超过 255 字。"
        )
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
            existing = session.scalar(
                select(Document).where(
                    Document.course_id == course_id, Document.name == name
                )
            )
            if existing:
                raise DocumentError(
                    "当前课程已有同名资料。请使用该资料的「替换」操作。", 409
                )
            document = Document(
                id=str(uuid4()), course_id=course_id, name=name, updated_at=utc_now()
            )
            session.add(document)
            session.flush()
            number = 1
        version = DocumentVersion(
            id=str(uuid4()),
            document_id=document.id,
            number=number,
            filename=name,
            content_hash=hashlib.sha256(data).hexdigest(),
            size=len(data),
            status="queued",
            step="等待处理",
        )
        path = raw_path(course_id, document.id, version.id, name)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            document.updated_at = utc_now()
            session.add(version)
            session.commit()
        except Exception:
            # 文件写入无法随数据库回滚，显式移除未登记的原件。
            session.rollback()
            path.unlink(missing_ok=True)
            raise
        return document_info(session, document)


def retry(session: Session, course_id: str, document_id: str):
    """清除失败版本的片段并重新排队，要求该版本是最新版本且原件仍在。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        course_id: 课程 ID。
        document_id: 资料 ID。

    Returns:
        重新排队后的资料摘要。

    Raises:
        CourseNotFound: 指定课程不存在。
        DocumentError: 资料归属不符、最新版本未失败或原件缺失。
    """
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
    """在同一事务中删除资料记录并登记后台清理任务。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        course_id: 课程 ID。
        document_id: 资料 ID。

    Raises:
        CourseNotFound: 指定课程不存在。
        DocumentError: 资料不存在或归属不符。
    """
    with catalog_lock:
        document = require_document(session, course_id, document_id)
        session.add(CleanupJob(course_id=course_id, document_id=document_id))
        session.delete(document)
        session.commit()


def preview(
    session: Session, course_id: str, document_id: str, offset: int, limit: int
):
    """按片段顺序分页读取可用版本的原文。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        course_id: 课程 ID。
        document_id: 资料 ID。
        offset: 结果起始偏移量。
        limit: 最多返回的条数。

    Returns:
        包含来源文件名、版本号、总片段数、偏移量和当前页片段的字典。

    Raises:
        CourseNotFound: 指定课程不存在。
        DocumentError: 资料归属不符或没有可用版本。
    """
    document = require_document(session, course_id, document_id)
    if not document.active_version_id:
        raise DocumentError("这份资料还没有可用版本，请等待处理完成或重试。", 409)
    version = session.get(DocumentVersion, document.active_version_id)
    chunks = list(
        session.scalars(
            select(Chunk)
            .where(Chunk.version_id == version.id)
            .order_by(Chunk.ordinal)
            .offset(offset)
            .limit(limit)
        )
    )
    return {
        "source": version.filename,
        "version": version.number,
        "total": version.chunk_count,
        "offset": offset,
        "chunks": [
            {
                "id": chunk.id,
                "text": chunk.text,
                "page": chunk.page,
                "ordinal": chunk.ordinal + 1,
            }
            for chunk in chunks
        ],
    }
