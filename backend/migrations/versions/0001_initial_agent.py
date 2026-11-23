"""创建课程研习室单一 Agent 的数据库基线。"""

from alembic import op
import sqlalchemy as sa

revision = "0001_initial_agent"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """创建课程、资料、清理任务以及完整的 Agent 对话表。"""
    op.create_table(
        "courses",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(80), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(40), nullable=False),
        sa.Column("updated_at", sa.String(40), nullable=False),
    )

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
        sa.Column("conversation_id", sa.String(36)),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(40), nullable=False))

    op.create_table("conversations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("course_id", sa.String(36), sa.ForeignKey("courses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(80), nullable=False),
        sa.Column("created_at", sa.String(40), nullable=False),
        sa.Column("updated_at", sa.String(40), nullable=False))
    op.create_index("ix_conversations_course_id", "conversations", ["course_id"])
    op.create_table("runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("conversation_id", sa.String(36), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("phase", sa.String(100), nullable=False),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column("trace", sa.Text(), nullable=False),
        sa.Column("web_search", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("agent_checkpoint_id", sa.String(36)),
        sa.Column("created_at", sa.String(40), nullable=False))
    op.create_index("ix_runs_conversation_id", "runs", ["conversation_id"])
    op.create_index("uq_runs_active_conversation", "runs", ["conversation_id"], unique=True,
                    sqlite_where=sa.text("status IN ('queued', 'running', 'cancelling')"))
    op.create_table("messages",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("conversation_id", sa.String(36), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("created_at", sa.String(40), nullable=False),
        sa.UniqueConstraint("run_id", "role"))
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])
    op.create_index("ix_messages_run_id", "messages", ["run_id"])
    op.create_table("citations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.String(36), nullable=False),
        sa.Column("version_id", sa.String(36), nullable=False),
        sa.Column("chunk_id", sa.String(80), nullable=False),
        sa.Column("source", sa.String(255), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("page", sa.Integer()),
        sa.Column("start_index", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(20), nullable=False, server_default="course"),
        sa.Column("url", sa.Text(), nullable=False, server_default=""),
        sa.Column("retrieved_at", sa.String(40), nullable=False, server_default=""),
        sa.UniqueConstraint("run_id", "number"))
    op.create_index("ix_citations_run_id", "citations", ["run_id"])
    op.create_table("run_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(30), nullable=False),
        sa.Column("data", sa.Text(), nullable=False))
    op.create_index("ix_run_events_run_id", "run_events", ["run_id"])


def downgrade():
    """按外键依赖顺序移除基线表。"""
    for table in (
        "run_events", "citations", "messages", "runs", "conversations",
        "cleanup_jobs", "chunks", "document_versions", "documents", "courses",
    ):
        op.drop_table(table)
