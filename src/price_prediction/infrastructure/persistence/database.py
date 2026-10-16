"""SQLAlchemy engine, session factory, and declarative base."""

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """Base class for persistence models."""


def get_database_url() -> str:
    """Read the database URL from the environment."""
    url = os.getenv("DATABASE_URL")

    if not url:
        raise RuntimeError(
            "DATABASE_URL must be configured when persistence is enabled."
        )

    return url


def create_session_factory(
    database_url: str | None = None,
) -> sessionmaker[Session]:
    """Create a session factory for a database URL."""
    url = database_url or get_database_url()

    engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
    )

    return sessionmaker(
        bind=engine,
        class_=Session,
        autoflush=False,
        expire_on_commit=False,
    )


def get_db_session() -> Generator[Session, None, None]:
    """Yield one session per request and always close it."""
    # The application initializes this factory once.
    factory = get_session_factory()

    with factory() as session:
        yield session


_session_factory: sessionmaker[Session] | None = None


def initialize_database(database_url: str | None = None) -> None:
    """Initialize the process-wide session factory."""
    global _session_factory

    if _session_factory is not None:
        return

    _session_factory = create_session_factory(database_url)


def get_session_factory() -> sessionmaker[Session]:
    """Return the initialized session factory."""
    if _session_factory is None:
        initialize_database()

    assert _session_factory is not None
    return _session_factory