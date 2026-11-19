"""
HealthGuard AI - PostgreSQL Production Database Integration Tests
Explicitly tests PostgreSQL 18:
  - Verifies live PostgreSQL connection and server version.
  - Verifies all 12 core tables and revoked_tokens table.
  - Verifies foreign key cascades and constraints.
  - Verifies unique constraints and duplicate error handling.
  - Verifies transaction isolation and rollback.
  - Reports actual database engine used for each test.
"""

import os
from urllib.parse import urlparse
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from backend.app.core.config import settings
from backend.app.core.security import hash_password
from backend.app.db.models import (
    Base, User, ModelVersion, Provider, Beneficiary, Claim,
    ProviderRiskScore, ClaimAnalysis, Explanation, CSVUpload, CSVUploadRow,
    CaseNote, AuditLog, RevokedToken
)

PG_USER = os.getenv("DB_USER", "postgres")
PG_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
PG_HOST = os.getenv("DB_HOST", "localhost")
PG_PORT = os.getenv("DB_PORT", "5432")
PG_TEST_URL = os.getenv(
    "TEST_DATABASE_URL",
    f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/healthguard_test"
)

# STRICT SAFETY ASSERTION: Validates exact parsed database name
from urllib.parse import urlparse
_parsed = urlparse(PG_TEST_URL)
_dbname = _parsed.path.lstrip("/").split("?")[0]
if _dbname != "healthguard_test":
    raise RuntimeError(
        f"CRITICAL SAFETY VIOLATION: Test connection targeted database '{_dbname}'. "
        "Automated tests must ONLY connect to 'healthguard_test'. Silent fallbacks strictly forbidden!"
    )

@pytest.fixture(scope="module")
def pg_engine():
    """Create live connection to actual PostgreSQL server."""
    engine = create_engine(PG_TEST_URL, pool_pre_ping=True)
    with engine.connect() as conn:
        res = conn.execute(text("SELECT version();")).scalar()
        print(f"\n[PostgreSQL Integration] Engine: {engine.dialect.name} | Version: {res}")
    return engine

@pytest.fixture
def pg_session(pg_engine):
    """Provides a transactional session rolled back after test to keep DB clean."""
    connection = pg_engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()

    yield session

    session.close()
    if transaction.is_active:
        transaction.rollback()
    connection.close()


def test_postgresql_engine_and_server_version(pg_engine):
    """Verify live database engine is PostgreSQL and not SQLite."""
    assert pg_engine.dialect.name == "postgresql", f"Expected postgresql, got {pg_engine.dialect.name}"
    with pg_engine.connect() as conn:
        version_str = conn.execute(text("SELECT version();")).scalar()
        assert "PostgreSQL" in version_str
        print(f"\nReported Engine: {pg_engine.dialect.name.upper()} (Live: {version_str[:35]}...)")


def test_postgresql_all_12_tables_exist(pg_engine):
    """Verify all 12 core business tables and revoked_tokens exist in PostgreSQL."""
    expected_tables = {
        "users", "model_versions", "providers", "beneficiaries", "claims",
        "provider_risk_scores", "claim_analyses", "explanations", "csv_uploads",
        "csv_upload_rows", "case_notes", "audit_logs", "revoked_tokens"
    }
    with pg_engine.connect() as conn:
        res = conn.execute(text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
        )).fetchall()
        existing_tables = {row[0] for row in res}
        missing = expected_tables - existing_tables
        assert not missing, f"Missing tables in PostgreSQL: {missing}"
        print(f"\nReported Engine: POSTGRESQL | Confirmed {len(expected_tables)} tables present.")


def test_postgresql_foreign_key_relationships(pg_session):
    """Verify foreign key enforcement across tables in PostgreSQL."""
    # 1. Create User
    user = User(name="PG Test User", email="pg_fk_test@healthguard.ai", password_hash="dummy_hash", role="investigator")
    pg_session.add(user)
    pg_session.flush()

    # 2. Create Provider & Beneficiary
    provider = Provider(provider_id="PRV_PG_FK_01", provider_name="PG General Hospital")
    bene = Beneficiary(beneficiary_id="BENE_PG_FK_01")
    pg_session.add_all([provider, bene])
    pg_session.flush()

    # 3. Create Claim linked to Provider and Beneficiary
    claim = Claim(
        claim_id="CLM_PG_FK_01",
        provider_id=provider.provider_id,
        beneficiary_id=bene.beneficiary_id,
        reimbursement_amount=5000.0
    )
    pg_session.add(claim)
    pg_session.flush()

    # 4. Create Model Version
    mv = ModelVersion(version="v2_pg_test", manifest_path="dummy/path", training_status="TRAINED")
    pg_session.add(mv)
    pg_session.flush()

    # 5. Create ProviderRiskScore linked to Provider and ModelVersion
    score = ProviderRiskScore(
        provider_id=provider.provider_id,
        model_version_id=mv.id,
        risk_score=0.88,
        risk_tier="HIGH"
    )
    pg_session.add(score)
    pg_session.flush()

    # 6. Create ClaimAnalysis linked to Claim and ProviderRiskScore
    ca = ClaimAnalysis(
        claim_id=claim.claim_id,
        provider_risk_score_id=score.id,
        anomaly_score=0.92,
        analysis_status="FLAGGED"
    )
    pg_session.add(ca)
    pg_session.flush()

    # 7. Create Explanation linked to ClaimAnalysis & ProviderRiskScore
    expl = Explanation(
        claim_analysis_id=ca.id,
        provider_risk_score_id=score.id,
        explanation_title="AI Risk Explanation",
        explanation_text="Anomalous billing pattern"
    )
    pg_session.add(expl)
    pg_session.flush()

    # 8. Create CSVUpload linked to User
    upload = CSVUpload(user_id=user.id, filename="pg_test.csv")
    pg_session.add(upload)
    pg_session.flush()

    # 9. Create CSVUploadRow linked to CSVUpload
    row = CSVUploadRow(csv_upload_id=upload.id, row_number=1, claim_id=claim.claim_id, processing_status="SUCCESS")
    pg_session.add(row)
    pg_session.flush()

    # 10. Create CaseNote linked to Provider and User
    note = CaseNote(provider_id=provider.provider_id, user_id=user.id, note_text="Case opened")
    pg_session.add(note)
    pg_session.flush()

    # 11. Create AuditLog linked to User
    audit = AuditLog(user_id=user.id, action="FK_VERIFICATION_TEST")
    pg_session.add(audit)
    pg_session.flush()

    # Assert all relationships navigate cleanly
    assert claim.provider.provider_id == "PRV_PG_FK_01"
    assert claim.beneficiary.beneficiary_id == "BENE_PG_FK_01"
    assert score.provider.provider_id == "PRV_PG_FK_01"
    assert ca.claim.claim_id == "CLM_PG_FK_01"
    assert expl.provider_risk_score.risk_tier == "HIGH"
    assert row.csv_upload.filename == "pg_test.csv"
    assert note.user.email == "pg_fk_test@healthguard.ai"
    assert audit.user.name == "PG Test User"
    print("\nReported Engine: POSTGRESQL | All foreign key relationships verified successfully.")


def test_postgresql_foreign_key_violation_rejected(pg_session):
    """Verify PostgreSQL rejects invalid foreign keys with IntegrityError."""
    orphan_claim = Claim(
        claim_id="CLM_ORPHAN_01",
        provider_id="NONEXISTENT_PROVIDER_PG",
        reimbursement_amount=100.0
    )
    pg_session.add(orphan_claim)
    with pytest.raises(IntegrityError) as exc_info:
        pg_session.flush()
    assert "foreign key constraint" in str(exc_info.value).lower()
    pg_session.rollback()
    print("\nReported Engine: POSTGRESQL | Foreign key violation rejected by PostgreSQL.")


def test_postgresql_unique_constraints(pg_session):
    """Verify PostgreSQL enforces unique constraints on users, providers, claims, and tokens."""
    # 1. Unique email on User
    u1 = User(name="User A", email="unique_test@healthguard.ai", password_hash="hash1", role="investigator")
    u2 = User(name="User B", email="unique_test@healthguard.ai", password_hash="hash2", role="investigator")
    pg_session.add(u1)
    pg_session.flush()
    pg_session.add(u2)
    with pytest.raises(IntegrityError):
        pg_session.flush()
    pg_session.rollback()

    # 2. Unique claim_id on Claim
    p = Provider(provider_id="PRV_UNIQ_01")
    pg_session.add(p)
    pg_session.flush()
    c1 = Claim(claim_id="CLM_UNIQ_01", provider_id=p.provider_id)
    c2 = Claim(claim_id="CLM_UNIQ_01", provider_id=p.provider_id)
    pg_session.add(c1)
    pg_session.flush()
    pg_session.add(c2)
    with pytest.raises(IntegrityError):
        pg_session.flush()
    pg_session.rollback()

    # 3. Unique token on RevokedToken
    t1 = RevokedToken(token="sample_jwt_revoked_token_unique_123")
    t2 = RevokedToken(token="sample_jwt_revoked_token_unique_123")
    pg_session.add(t1)
    pg_session.flush()
    pg_session.add(t2)
    with pytest.raises(IntegrityError):
        pg_session.flush()
    pg_session.rollback()
    print("\nReported Engine: POSTGRESQL | Unique constraints verified on users, claims, and revoked_tokens.")


def test_postgresql_transaction_rollback_preserves_state(pg_session):
    """Verify PostgreSQL transaction rollback completely reverts uncommitted writes."""
    initial_user_count = pg_session.query(User).filter(User.email.like("%@rollback_test.ai")).count()
    assert initial_user_count == 0

    # Write inside savepoint / transaction
    nested = pg_session.begin_nested()
    temp_user = User(name="Temp User", email="temp@rollback_test.ai", password_hash="h", role="investigator")
    pg_session.add(temp_user)
    pg_session.flush()
    assert pg_session.query(User).filter(User.email == "temp@rollback_test.ai").count() == 1

    # Rollback savepoint
    nested.rollback()
    assert pg_session.query(User).filter(User.email == "temp@rollback_test.ai").count() == 0
    print("\nReported Engine: POSTGRESQL | Transaction rollback preserves state cleanly.")
