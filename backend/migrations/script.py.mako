"""${message}"""
from alembic import op
import sqlalchemy as sa

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}

def upgrade():
    """将数据库架构迁移到当前版本。"""
    ${upgrades if upgrades else "pass"}

def downgrade():
    """将数据库架构回退到上一个版本。"""
    ${downgrades if downgrades else "pass"}
