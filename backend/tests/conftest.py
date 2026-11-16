"""
HealthGuard AI - Phase 4 Backend Test Configuration and Fixtures
"""

import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Set test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from backend.app.db.database import Base, get_db
from backend.app.main import app
import backend.app.db.models  # Ensure all 12 tables are registered in Base.metadata
from backend.app.db.models import (
    User, ModelVersion, Provider, Beneficiary, Claim,
    ProviderRiskScore, ClaimAnalysis, Explanation, CSVUpload, CSVUploadRow
)
from backend.app.core.security import hash_password, create_access_token

# In-memory SQLite engine for fast, isolated tests
TEST_ENGINE = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=TEST_ENGINE)

# Point main application database sessionmaker to TEST_ENGINE during tests
import backend.app.db.database as db_module
db_module.engine = TEST_ENGINE
db_module.SessionLocal.configure(bind=TEST_ENGINE)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db_schema():
    """Create all tables in the test engine once per session."""
    Base.metadata.create_all(bind=TEST_ENGINE)
    yield
    Base.metadata.drop_all(bind=TEST_ENGINE)


@pytest.fixture(autouse=True)
def clean_database():
    """Ensure clean database before and after each test."""
    with TEST_ENGINE.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())
    yield
    with TEST_ENGINE.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())


@pytest.fixture
def db_session():
    """Provides a transactional database session."""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session):
    """FastAPI TestClient with overridden get_db dependency."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def test_user(db_session):
    """Creates or retrieves a regular investigator user."""
    user = db_session.query(User).filter(User.email == "investigator@healthguard.ai").first()
    if not user:
        user = User(
            email="investigator@healthguard.ai",
            name="Test Investigator",
            password_hash=hash_password("SecurePass123!"),
            role="INVESTIGATOR",
            is_active=True,
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
    return user


@pytest.fixture
def test_admin(db_session):
    """Creates or retrieves an administrator user."""
    admin = db_session.query(User).filter(User.email == "admin@healthguard.ai").first()
    if not admin:
        admin = User(
            email="admin@healthguard.ai",
            name="Test Admin",
            password_hash=hash_password("AdminPass123!"),
            role="ADMIN",
            is_active=True,
        )
        db_session.add(admin)
        db_session.commit()
        db_session.refresh(admin)
    return admin


@pytest.fixture
def auth_headers(test_user):
    """Bearer auth headers for regular investigator user."""
    token = create_access_token(
        data={"sub": test_user.email, "role": test_user.role, "user_id": test_user.id}
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers(test_admin):
    """Bearer auth headers for admin user."""
    token = create_access_token(
        data={"sub": test_admin.email, "role": test_admin.role, "user_id": test_admin.id}
    )
    return {"Authorization": f"Bearer {token}"}
