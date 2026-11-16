"""记录已完成的会话检查点和待清理的会话。"""

from alembic import op
import sqlalchemy as sa

revision = "0007_agent_checkpoints"
down_revision = "0006_run_answer_settings"
branch_labels = None
depends_on = None


def upgrade():
    """添加可空字段，使已有的问答和清理任务继续有效。"""
    op.add_column("runs", sa.Column("agent_checkpoint_id", sa.String(36)))
    op.add_column("cleanup_jobs", sa.Column("conversation_id", sa.String(36)))


def downgrade():
    """移除检查点相关的业务字段。"""
    op.drop_column("cleanup_jobs", "conversation_id")
    op.drop_column("runs", "agent_checkpoint_id")
