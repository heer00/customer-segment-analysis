"""
app/db/database.py
──────────────────
SQLAlchemy database setup and session management.

WHY:
  - Database operations require active connections. Managing connections
    manually leads to leaks, stale connections, or threading errors.
  - This file abstracts connection pools and provides a type-safe context-manager
    dependency ('get_db') that handles closing connections automatically after requests.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.core.config import settings

# For SQLite, we set check_same_thread=False to allow multiple threads to access it.
# (FastAPI is concurrent/async, while SQLite is synchronous by default).
connect_args = {}
if settings.database_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    echo=settings.debug,  # Prints generated SQL to stdout when debugging is True
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def get_db():
    """
    FastAPI dependency yielding a database session.
    Guarantees the connection is closed even if an exception occurs.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
