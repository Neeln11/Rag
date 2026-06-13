"""
database.py
===========
SQLAlchemy ORM layer for the Adaptive RAG API.

Responsibilities
----------------
- Declare the SQLite engine and session factory
- Define all ORM models (currently: FeedbackRecord)
- Expose a `get_db()` dependency for FastAPI route injection
- Create all tables at import time (idempotent via `create_all`)

Design notes
------------
- SQLite is used because it requires zero infrastructure and is sufficient
  for the feedback store (write-once, low-concurrency).  For a high-traffic
  deployment, swap `sqlite:///...` for `postgresql+asyncpg://...` and change
  `_SessionLocal` to use `async_sessionmaker`.
- `check_same_thread=False` is required for SQLite when used behind FastAPI's
  async event loop, which may execute sync routes on different threads.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ---------------------------------------------------------------------------
# Engine & Session factory
# ---------------------------------------------------------------------------

_DB_PATH = Path(__file__).parent / "feedback.db"
_engine = create_engine(
    f"sqlite:///{_DB_PATH}",
    connect_args={"check_same_thread": False},
)
_SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)


# ---------------------------------------------------------------------------
# ORM base
# ---------------------------------------------------------------------------


class _Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class FeedbackRecord(_Base):
    """
    Stores user feedback (thumbs up / thumbs down) for generated answers.

    Fields
    ------
    id           : Auto-incrementing primary key.
    timestamp    : UTC time when feedback was submitted.
    session_id   : The LangGraph session that produced the answer.
    question     : The original user question.
    answer       : The generated answer that was rated.
    rating       : 'up' or 'down'.
    comment      : Optional free-text comment (≤1000 chars).
    """

    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    session_id = Column(String(128), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    rating = Column(String(4), nullable=False)   # "up" | "down"
    comment = Column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Table creation (idempotent)
# ---------------------------------------------------------------------------

_Base.metadata.create_all(_engine)


# ---------------------------------------------------------------------------
# Dependency
# ---------------------------------------------------------------------------


def get_db() -> Session:
    """
    Return a new SQLAlchemy session.

    The caller is responsible for closing the session (use try/finally or a
    context manager).  FastAPI dependency injection can use this directly:

        db: Session = Depends(get_db)

    For async routes, wrap the synchronous call in `run_in_executor` or switch
    to an async SQLAlchemy session.
    """
    return _SessionLocal()
