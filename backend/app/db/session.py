"""配置 SQLite 连接与请求级数据库会话。"""

from sqlalchemy import create_engine, event
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker

from app.core.config import DATABASE_PATH

engine = create_engine(
    URL.create("sqlite", database=str(DATABASE_PATH)),
    connect_args={"check_same_thread": False, "timeout": 15},
)


@event.listens_for(engine, "connect")
def sqlite_settings(connection, _record):
    """为新连接启用外键约束和 WAL 日志模式。

    Args:
        connection: 新建的 SQLite DB-API 连接。
        _record: SQLAlchemy 传入的连接池记录。
    """
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")


SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def get_session():
    """为请求依赖提供 SQLAlchemy 会话，并在依赖退出时关闭。

    Yields:
        当前请求使用的 SQLAlchemy 会话。
    """
    with SessionLocal() as session:
        yield session
