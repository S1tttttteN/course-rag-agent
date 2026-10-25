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
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")


SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def get_session():
    with SessionLocal() as session:
        yield session
