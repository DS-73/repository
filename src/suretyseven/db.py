"""Database wiring.

A thin wrapper around SQLAlchemy so the rest of the codebase never imports
``create_engine`` directly and tests can swap in an in-memory database.

Production trade-off: schema is created with ``Base.metadata.create_all`` on
startup instead of Alembic migrations.  That is fine for a take-home (single
service, additive schema) and is called out in ``docs/architecture.md`` as the
first thing to replace before running real workloads.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from suretyseven.models import Base

__all__ = ["Database", "check_database", "create_db_engine", "init_db"]


def _is_memory_sqlite(database_url: str) -> bool:
    """True for in-memory SQLite URLs (``sqlite://``, ``sqlite:///:memory:``)."""
    if not database_url.startswith("sqlite"):
        return False
    target = database_url.split("///", 1)[-1] if "///" in database_url else database_url[9:]
    return target in {"", ":memory:"} or ":memory:" in target


def create_db_engine(database_url: str, *, echo: bool = False) -> Engine:
    """Create an engine with sensible pooling/connection settings."""
    kwargs: dict = {"echo": echo, "pool_pre_ping": True, "future": True}
    if database_url.startswith("sqlite"):
        # FastAPI runs sync endpoints in a threadpool -> connections cross threads.
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
        if _is_memory_sqlite(database_url):
            kwargs["poolclass"] = StaticPool
    engine = create_engine(database_url, **kwargs)

    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=5000")
            finally:
                cursor.close()

    return engine


def init_db(engine: Engine) -> None:
    """Create tables that do not exist yet."""
    Base.metadata.create_all(engine)


def check_database(engine: Engine) -> bool:
    """Readiness probe: can we execute a trivial query?"""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:  # pragma: no cover - depends on infra failures
        return False


class Database:
    """Owns the engine and hands out short lived sessions."""

    def __init__(self, database_url: str, *, echo: bool = False) -> None:
        self.engine = create_db_engine(database_url, echo=echo)
        self.session_factory = sessionmaker(
            bind=self.engine, expire_on_commit=False, autoflush=False, future=True
        )

    def create_schema(self) -> None:
        init_db(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Transactional scope: commit on success, roll back on failure."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def is_ready(self) -> bool:
        return check_database(self.engine)

    def dispose(self) -> None:
        self.engine.dispose()
