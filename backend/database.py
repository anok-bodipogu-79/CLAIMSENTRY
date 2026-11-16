"""
HealthGuard AI - Unified Database Engine and Session Management
Re-exports the core database engine, sessionmaker, Base, and get_db from backend.app.db.database
to guarantee a single consistent database connection and metadata registry across the entire application.
"""

import backend.app.db.database as _app_db

Base = _app_db.Base
get_db = _app_db.get_db
init_db = _app_db.init_db
create_db_engine = _app_db.create_db_engine


def __getattr__(name):
    """Dynamically delegate attributes to backend.app.db.database to maintain single connection."""
    if hasattr(_app_db, name):
        return getattr(_app_db, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["engine", "SessionLocal", "Base", "get_db", "init_db", "create_db_engine"]
