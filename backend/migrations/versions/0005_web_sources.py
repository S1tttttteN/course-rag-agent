"""Store external-source metadata alongside course citations."""
from alembic import op
import sqlalchemy as sa

revision = '0005_web_sources'
down_revision = '0004_agent_mode'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('citations', sa.Column('kind', sa.String(20), nullable=False, server_default='course'))
    op.add_column('citations', sa.Column('url', sa.Text(), nullable=False, server_default=''))
    op.add_column('citations', sa.Column('retrieved_at', sa.String(40), nullable=False, server_default=''))


def downgrade():
    op.drop_column('citations', 'retrieved_at')
    op.drop_column('citations', 'url')
    op.drop_column('citations', 'kind')
