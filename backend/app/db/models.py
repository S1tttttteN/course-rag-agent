from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import ForeignKey, Index, Integer, String, Text, UniqueConstraint, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Course(Base):
    __tablename__ = "courses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[str] = mapped_column(String(40), nullable=False, default=utc_now)
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False, default=utc_now, onupdate=utc_now)


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("course_id", "name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    active_version_id: Mapped[str | None] = mapped_column(String(36))
    created_at: Mapped[str] = mapped_column(String(40), default=utc_now)
    updated_at: Mapped[str] = mapped_column(String(40), default=utc_now)


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    __table_args__ = (UniqueConstraint("document_id", "number"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    number: Mapped[int] = mapped_column(Integer)
    filename: Mapped[str] = mapped_column(String(255))
    content_hash: Mapped[str] = mapped_column(String(64))
    size: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), index=True, default="queued")
    step: Mapped[str] = mapped_column(String(100), default="等待处理")
    error: Mapped[str] = mapped_column(Text, default="")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding_profile: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[str] = mapped_column(String(40), default=utc_now)


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("document_versions.id", ondelete="CASCADE"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    page: Mapped[int | None] = mapped_column(Integer)
    start_index: Mapped[int] = mapped_column(Integer, default=0)


class CleanupJob(Base):
    """Independent of foreign keys so cleanup survives deleting a course."""
    __tablename__ = "cleanup_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    course_id: Mapped[str] = mapped_column(String(36))
    document_id: Mapped[str | None] = mapped_column(String(36))
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[str] = mapped_column(String(40), default=utc_now)


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(80), default="新对话")
    created_at: Mapped[str] = mapped_column(String(40), default=utc_now)
    updated_at: Mapped[str] = mapped_column(String(40), default=utc_now)


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (Index("uq_runs_active_conversation", "conversation_id", unique=True,
                           sqlite_where=text("status IN ('queued', 'running', 'cancelling')")),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    question: Mapped[str] = mapped_column(Text)
    mode: Mapped[str] = mapped_column(String(10), default="rag")
    query: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="queued")
    phase: Mapped[str] = mapped_column(String(100), default="等待回答")
    error: Mapped[str] = mapped_column(Text, default="")
    trace: Mapped[str] = mapped_column(Text, default="{}")
    top_k: Mapped[int] = mapped_column(Integer, default=4)
    semantic: Mapped[int] = mapped_column(Integer, default=1)
    rerank: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[str] = mapped_column(String(40), default=utc_now)


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (UniqueConstraint("run_id", "role"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[str] = mapped_column(String(40), default=utc_now)


class Citation(Base):
    __tablename__ = "citations"
    __table_args__ = (UniqueConstraint("run_id", "number"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    number: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(20), default="course")
    url: Mapped[str] = mapped_column(Text, default="")
    retrieved_at: Mapped[str] = mapped_column(String(40), default="")
    # No FK to documents/chunks: snapshots survive removal of their original source.
    document_id: Mapped[str] = mapped_column(String(36))
    version_id: Mapped[str] = mapped_column(String(36))
    chunk_id: Mapped[str] = mapped_column(String(80))
    source: Mapped[str] = mapped_column(String(255))
    version: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    page: Mapped[int | None] = mapped_column(Integer)
    start_index: Mapped[int] = mapped_column(Integer, default=0)


class RunEvent(Base):
    __tablename__ = "run_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(30))
    data: Mapped[str] = mapped_column(Text)
