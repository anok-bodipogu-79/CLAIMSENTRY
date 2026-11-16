"""
HealthGuard AI - Database Engine and Session Management
"""

import os
import logging
from sqlalchemy import create_engine, text as sa_text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import OperationalError

from backend.app.core.config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()


def create_db_engine():
    """
    Create SQLAlchemy engine.
    Production explicitly enforces PostgreSQL without silent fallback.
    SQLite is permitted only when explicitly requested in dev or test environments.
    Passwords and sensitive credentials are never emitted in logs or exceptions.
    """
    db_url = os.getenv("DATABASE_URL") or settings.DATABASE_URL
    masked_url = settings.get_masked_database_url()
    is_sqlite = db_url.startswith("sqlite") if db_url else False
    env = (os.getenv("ENVIRONMENT") or settings.ENVIRONMENT or "development").lower()
    use_sqlite = str(os.getenv("USE_SQLITE", settings.USE_SQLITE)).lower() in ("true", "1")

    # If SQLite is explicitly selected
    if is_sqlite:
        if env == "production":
            raise RuntimeError("Fatal: SQLite cannot be used as the database engine in production. PostgreSQL is required.")
        logger.info(f"Using explicitly configured SQLite database: {masked_url}")
        return create_engine(db_url, connect_args={"check_same_thread": False}, echo=False)

    # Connecting to PostgreSQL
    try:
        engine = create_engine(
            db_url,
            pool_pre_ping=True,
            echo=False,
        )
        # Test connection
        with engine.connect() as conn:
            conn.execute(sa_text("SELECT 1"))
        logger.info(f"Successfully connected to PostgreSQL engine at {masked_url}")
        return engine
    except Exception as e:
        if env in ["development", "test"] and use_sqlite:
            logger.warning(
                f"PostgreSQL connection to {masked_url} failed: {e}. USE_SQLITE is explicitly enabled for {env}. "
                f"Falling back to SQLite at {settings.SQLITE_FALLBACK_URL}."
            )
            return create_engine(
                settings.SQLITE_FALLBACK_URL,
                connect_args={"check_same_thread": False},
                echo=False,
            )
        
        # In production or when USE_SQLITE is False, silent fallback to SQLite is strictly forbidden!
        raise RuntimeError(
            f"Fatal: Could not connect to primary PostgreSQL database at {masked_url}: {e}. "
            "Silent fallback to SQLite is prohibited in production. "
            "Ensure the PostgreSQL database server is active and credentials are correct."
        )


engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """
    Initialize database schema safely.
    In production, schema management is strictly governed by Alembic migrations
    ('alembic -c backend/alembic.ini upgrade head') to prevent silent, unversioned DDL mutations.
    In development and testing, automatically ensures tables are created.
    """
    env = (os.getenv("ENVIRONMENT") or settings.ENVIRONMENT or "development").lower()
    if env == "production":
        logger.info(
            "Production environment: skipping implicit Base.metadata.create_all(). "
            "Database schema migrations must be applied explicitly using Alembic ('alembic upgrade head')."
        )
        with engine.connect() as conn:
            conn.execute(sa_text("SELECT 1"))
    else:
        Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency yielding a transactional SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
