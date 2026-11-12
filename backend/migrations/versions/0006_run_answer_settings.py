"""Store the effective answer settings for each run."""
from alembic import op
import sqlalchemy as sa

revision = "0006_run_answer_settings"
down_revision = "0005_web_sources"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("runs", sa.Column("web_search", sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column("runs", sa.Column("answer_review", sa.Boolean(), nullable=False, server_default=sa.false()))


def downgrade():
    op.drop_column("runs", "answer_review")
    op.drop_column("runs", "web_search")
