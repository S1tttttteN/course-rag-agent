"""Persist the selected execution mode; tool summaries live in runs.trace."""
from alembic import op
import sqlalchemy as sa

revision = "0004_agent_mode"
down_revision = "0003_conversations"
branch_labels = None
depends_on = None


def upgrade():
    """为运行记录添加问答模式列，历史记录默认为 rag。"""
    op.add_column("runs", sa.Column("mode", sa.String(10), nullable=False, server_default="rag"))


def downgrade():
    """移除运行记录的问答模式列。"""
    op.drop_column("runs", "mode")
