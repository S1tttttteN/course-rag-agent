"""使用应用数据库引擎执行 Alembic 迁移。"""

from alembic import context

from app.db.models import Base
from app.db.session import engine

with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=Base.metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()
