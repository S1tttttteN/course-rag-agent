"""Create the persistent course catalog."""
from alembic import op
import sqlalchemy as sa

revision = "0001_courses"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """创建课程目录表。"""
    op.create_table(
        "courses",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(80), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(40), nullable=False),
        sa.Column("updated_at", sa.String(40), nullable=False),
    )


def downgrade():
    """删除课程目录表。"""
    op.drop_table("courses")
