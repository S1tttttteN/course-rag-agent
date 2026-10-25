"""Persistent conversations, streaming runs and independent evidence snapshots."""
from alembic import op
import sqlalchemy as sa

revision = "0003_conversations"
down_revision = "0002_documents"
branch_labels = None
depends_on = None


def upgrade():
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
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("phase", sa.String(100), nullable=False),
        sa.Column("error", sa.Text(), nullable=False),
        sa.Column("trace", sa.Text(), nullable=False),
        sa.Column("top_k", sa.Integer(), nullable=False),
        sa.Column("semantic", sa.Integer(), nullable=False),
        sa.Column("rerank", sa.Integer(), nullable=False),
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
        sa.UniqueConstraint("run_id", "number"))
    op.create_index("ix_citations_run_id", "citations", ["run_id"])
    op.create_table("run_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(30), nullable=False),
        sa.Column("data", sa.Text(), nullable=False))
    op.create_index("ix_run_events_run_id", "run_events", ["run_id"])


def downgrade():
    for table in ("run_events", "citations", "messages", "runs", "conversations"):
        op.drop_table(table)
