"""Versioned documents, authoritative chunks and durable cleanup."""
from alembic import op
import sqlalchemy as sa

revision = "0002_documents"
down_revision = "0001_courses"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table("documents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("course_id", sa.String(36), sa.ForeignKey("courses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("active_version_id", sa.String(36)),
        sa.Column("created_at", sa.String(40), nullable=False),
        sa.Column("updated_at", sa.String(40), nullable=False),
        sa.UniqueConstraint("course_id", "name"))
    op.create_index("ix_documents_course_id", "documents", ["course_id"])
    op.create_table("document_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("document_id", sa.String(36), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("size", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("step", sa.String(100), nullable=False),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("embedding_profile", sa.String(255), nullable=False),
        sa.Column("created_at", sa.String(40), nullable=False),
        sa.UniqueConstraint("document_id", "number"))
    op.create_index("ix_document_versions_document_id", "document_versions", ["document_id"])
    op.create_index("ix_document_versions_status", "document_versions", ["status"])
    op.create_table("chunks",
        sa.Column("id", sa.String(80), primary_key=True),
        sa.Column("version_id", sa.String(36), sa.ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("page", sa.Integer()),
        sa.Column("start_index", sa.Integer(), nullable=False))
    op.create_index("ix_chunks_version_id", "chunks", ["version_id"])
    op.create_table("cleanup_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("course_id", sa.String(36), nullable=False),
        sa.Column("document_id", sa.String(36)),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(40), nullable=False))


def downgrade():
    for table in ("cleanup_jobs", "chunks", "document_versions", "documents"):
        op.drop_table(table)
