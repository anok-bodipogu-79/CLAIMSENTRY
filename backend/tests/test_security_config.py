"""
HealthGuard AI - Security Configuration Unit Tests
Verifies:
  - DATABASE_URL requirement outside test mode.
  - JWT_SECRET_KEY requirement and length/strength enforcement.
  - Rejection of default/placeholder secrets.
  - Database URL credential masking (no passwords logged).
  - Production enforcement of PostgreSQL (rejection of SQLite).
"""

import pytest
from backend.app.core.config import Settings


def test_masked_database_url_hides_password():
    """Verify get_masked_database_url() masks database password and credentials."""
    s = Settings(
        ENVIRONMENT="test",
        DATABASE_URL="postgresql://myuser:SuperSecretPassword123@myhost:5432/healthguard_db",
        JWT_SECRET_KEY="a" * 32,
    )
    masked = s.get_masked_database_url()
    assert "SuperSecretPassword123" not in masked
    assert "myuser:****@myhost:5432" in masked


def test_production_rejects_sqlite():
    """Verify production environment strictly rejects SQLite."""
    with pytest.raises(ValueError, match="SQLite is strictly prohibited as a database engine in production"):
        Settings(
            ENVIRONMENT="production",
            DATABASE_URL="sqlite:///./backend/healthguard.db",
            JWT_SECRET_KEY="cryptographically-strong-secret-key-32-chars!",
        )


def test_production_rejects_missing_database_url():
    """Verify production environment strictly requires DATABASE_URL."""
    with pytest.raises(ValueError, match="DATABASE_URL must be provided"):
        Settings(
            ENVIRONMENT="production",
            DATABASE_URL="",
            JWT_SECRET_KEY="cryptographically-strong-secret-key-32-chars!",
        )


def test_production_rejects_placeholder_database_url():
    """Verify production environment rejects unpopulated template placeholders."""
    with pytest.raises(ValueError, match="DATABASE_URL contains unconfigured template placeholders"):
        Settings(
            ENVIRONMENT="production",
            DATABASE_URL="postgresql://<DB_USER>:<DB_PASSWORD>@<DB_HOST>:5432/<DB_NAME>",
            JWT_SECRET_KEY="cryptographically-strong-secret-key-32-chars!",
        )


def test_production_rejects_short_jwt_secret():
    """Verify secrets under 32 characters are rejected outside test mode."""
    with pytest.raises(ValueError, match="JWT_SECRET_KEY must be at least 32 characters in length"):
        Settings(
            ENVIRONMENT="production",
            DATABASE_URL="postgresql://user:pass@localhost:5432/healthguard",
            JWT_SECRET_KEY="short-secret-under-32-chars",
        )


def test_production_rejects_default_or_placeholder_jwt_secrets():
    """Verify known default and example secrets are rejected."""
    weak_secrets = [
        "healthguard-super-secret-jwt-key-2026-production-ready",
        "healthguard-super-secret-jwt-key-change-in-production",
        "<GENERATE_AND_INSERT_STRONG_RANDOM_SECRET_KEY_MIN_32_CHARS>",
        "change-me-change-me-change-me-32",
    ]
    for weak_sec in weak_secrets:
        with pytest.raises(ValueError, match="JWT_SECRET_KEY matches a known default or placeholder value"):
            Settings(
                ENVIRONMENT="production",
                DATABASE_URL="postgresql://user:pass@localhost:5432/healthguard",
                JWT_SECRET_KEY=weak_sec,
            )


def test_test_environment_allows_test_configuration():
    """Verify test environment safely allows test settings without requiring real DB credentials."""
    s = Settings(
        ENVIRONMENT="test",
        DATABASE_URL=None,
        JWT_SECRET_KEY=None,
    )
    assert s.DATABASE_URL == "sqlite:///:memory:"
    assert len(s.JWT_SECRET_KEY) >= 32
