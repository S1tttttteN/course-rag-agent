from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document, DocumentVersion


def documents_for_course(session: Session, course_id: str):
    return list(session.scalars(select(Document).where(Document.course_id == course_id).order_by(Document.created_at.desc())))


def active_chunks(session: Session, course_id: str):
    return list(session.execute(
        select(Chunk, Document, DocumentVersion)
        .join(DocumentVersion, Chunk.version_id == DocumentVersion.id)
        .join(Document, Document.active_version_id == DocumentVersion.id)
        .where(Document.course_id == course_id, DocumentVersion.status == "ready")
        .order_by(Document.id, Chunk.ordinal)
    ).all())


def versions_for_document(session: Session, document_id: str):
    return list(session.scalars(select(DocumentVersion).where(DocumentVersion.document_id == document_id).order_by(DocumentVersion.number.desc())))
