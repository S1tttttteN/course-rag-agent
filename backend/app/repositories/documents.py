"""封装课程资料、版本和已发布片段的数据库查询。"""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document, DocumentVersion


def documents_for_course(session: Session, course_id: str):
    """查询指定课程的资料目录。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        course_id: 课程 ID。

    Returns:
        按创建时间降序排列的资料列表。
    """
    return list(
        session.scalars(
            select(Document)
            .where(Document.course_id == course_id)
            .order_by(Document.created_at.desc())
        )
    )


def _active_chunks_query(course_id: str):
    """构造限制在当前课程可用版本的片段查询。"""
    return (
        select(Chunk, Document, DocumentVersion)
        .join(DocumentVersion, Chunk.version_id == DocumentVersion.id)
        .join(Document, Document.active_version_id == DocumentVersion.id)
        .where(
            Document.course_id == course_id,
            DocumentVersion.document_id == Document.id,
            DocumentVersion.status == "ready",
        )
    )


def active_chunks(session: Session, course_id: str, *, chunk_ids=None):
    """读取当前课程的可用片段，支持按候选 ID 批量查询。

    Args:
        session: 当前数据库会话。
        course_id: 课程 ID。
        chunk_ids: 指定时只查询该批片段。

    Returns:
        按资料 ID 和片段顺序排列的片段、资料与版本记录。
    """
    query = _active_chunks_query(course_id)
    if chunk_ids is not None:
        query = query.where(Chunk.id.in_(chunk_ids))
    return session.execute(query.order_by(Document.id, Chunk.ordinal)).all()


def active_chunk_ids(session: Session, course_id: str, chunk_ids):
    """只取本批候选中仍属于当前课程可用版本的 ID。"""
    query = _active_chunks_query(course_id).with_only_columns(Chunk.id)
    return set(session.scalars(query.where(Chunk.id.in_(chunk_ids))))


def versions_for_document(session: Session, document_id: str):
    """查询资料的版本历史。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        document_id: 资料 ID。

    Returns:
        按版本号降序排列的资料版本列表。
    """
    return list(
        session.scalars(
            select(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .order_by(DocumentVersion.number.desc())
        )
    )
