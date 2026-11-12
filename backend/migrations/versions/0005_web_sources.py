"""Store external-source metadata alongside course citations."""
from alembic import op
import sqlalchemy as sa

revision = '0005_web_sources'
down_revision = '0004_agent_mode'
branch_labels = None
depends_on = None


def upgrade():
    """为引用添加来源类型、网址和获取时间字段。"""
    op.add_column('citations', sa.Column('kind', sa.String(20), nullable=False, server_default='course'))
    op.add_column('citations', sa.Column('url', sa.Text(), nullable=False, server_default=''))
    op.add_column('citations', sa.Column('retrieved_at', sa.String(40), nullable=False, server_default=''))


def downgrade():
    """移除引用中用于描述网络来源的三个字段。"""
    op.drop_column('citations', 'retrieved_at')
    op.drop_column('citations', 'url')
    op.drop_column('citations', 'kind')
