"""
Database engine and session management.

Supports PostgreSQL in production and SQLite for local development.
Configure via DATABASE_URL environment variable.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./f1_intelligence.db"
)

# Use check_same_thread=False only for SQLite
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


def get_db():
    """FastAPI dependency — yields a DB session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Called during startup or data ingestion."""
    from data.models import (  # noqa: F401 — ensure models are registered
        Circuit, Driver, Constructor, Season,
        Race, RaceResult, Qualifying, Weather, FeatureRow,
    )
    Base.metadata.create_all(bind=engine)
