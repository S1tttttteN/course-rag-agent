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


def active_chunks(session: Session, course_id: str):
    """查询当前课程已发布版本的片段，供关键词检索和向量候选复核使用。

    Args:
        session: 用于当前操作的 SQLAlchemy 会话。
        course_id: 课程 ID。

    Returns:
        按资料 ID 和片段顺序排列的结果，每行包含片段、资料及版本。
    """
    return list(
        session.execute(
            select(Chunk, Document, DocumentVersion)
            .join(DocumentVersion, Chunk.version_id == DocumentVersion.id)
            .join(Document, Document.active_version_id == DocumentVersion.id)
            .where(Document.course_id == course_id, DocumentVersion.status == "ready")
            .order_by(Document.id, Chunk.ordinal)
        ).all()
    )


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
