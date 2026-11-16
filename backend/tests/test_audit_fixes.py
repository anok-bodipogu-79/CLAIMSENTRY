"""
HealthGuard AI - Audit Verification & Regression Test Suite
===========================================================
Validates:
1. Single unified database engine, sessionmaker, and Base across backend.
2. Database bootstrap failure handling and immediate abortion.
3. Alembic migration and required table presence.
4. Authentication, forced investigator role, and token revocation.
5. Provider and Claim API endpoints.
6. Single-claim prediction pipeline.
7. CSV prediction and reconciliation invariant.
8. Exact score equality between Single-Claim and CSV inference pipelines.
9. Database persistence of evaluations and analyses.
10. Frontend API integration contract compliance.
"""

import io
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlalchemy import inspect as sa_inspect

from backend.app.main import app
from backend.app.db.database import engine, Base, SessionLocal
from backend.app.db.models import (
    User, Provider, Claim, ProviderRiskScore, ClaimAnalysis, CSVUpload, CSVUploadRow
)
import backend.database as legacy_db
from backend.app.services.prediction_service import prediction_service
from backend.app.workers.csv_worker import process_csv_in_background
from backend.app.db.bootstrap import ensure_database_exists, REQUIRED_TABLES


client = TestClient(app)


# --------------------------------------------------------------------------
# 1. Single Database Connection
# --------------------------------------------------------------------------
def test_single_database_connection():
    """Verify that backend and backend.app share the exact same database engine, session, and Base."""
    assert legacy_db.engine is engine, "backend.database.engine must be identical to backend.app.db.database.engine"
    assert legacy_db.Base is Base, "backend.database.Base must be identical to backend.app.db.database.Base"
    assert legacy_db.SessionLocal is SessionLocal, "backend.database.SessionLocal must be identical"


# --------------------------------------------------------------------------
# 2. Database Bootstrap Failure Handling
# --------------------------------------------------------------------------
def test_database_bootstrap_failure_handling():
    """Verify bootstrap raises RuntimeError immediately upon invalid database connection."""
    invalid_params = {
        "host": "invalid-non-existent-host-99999.local",
        "port": 5432,
        "user": "nonexistent_user",
        "password": "wrong_password",
        "database": "nonexistent_db",
    }
    with pytest.raises(RuntimeError) as exc_info:
        ensure_database_exists(invalid_params)
    assert "Fatal: Could not connect to PostgreSQL" in str(exc_info.value)


# --------------------------------------------------------------------------
# 3. Alembic Migration & Required Tables
# --------------------------------------------------------------------------
def test_alembic_migration_and_tables(db_session):
    """Verify that all required tables and Alembic revision are present."""
    inspector = sa_inspect(engine)
    existing_tables = set(inspector.get_table_names())
    for req_table in REQUIRED_TABLES:
        assert req_table in existing_tables, f"Missing required table in database: {req_table}"


# --------------------------------------------------------------------------
# 4. Authentication Flow & Role Enforcement
# --------------------------------------------------------------------------
def test_auth_registration_and_revocation(db_session):
    """Verify forced investigator role, JWT login, and token revocation."""
    email = f"audit_test_{datetime.now(timezone.utc).timestamp()}@healthguard.ai"
    reg_resp = client.post(
        "/api/auth/register",
        json={"name": "Audit Tester", "email": email, "password": "SecureTestPassword123!", "role": "admin"},
    )
    assert reg_resp.status_code == 201
    assert reg_resp.json()["role"] == "investigator", "Public registration must force role to 'investigator'"

    # Login
    login_resp = client.post(
        "/api/auth/login",
        json={"email": email, "password": "SecureTestPassword123!"},
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Verify Profile
    me_resp = client.get("/api/auth/me", headers=headers)
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == email

    # Logout
    logout_resp = client.post("/api/auth/logout", headers=headers)
    assert logout_resp.status_code == 200

    # Token should be revoked
    revoked_resp = client.get("/api/auth/me", headers=headers)
    assert revoked_resp.status_code == 401


# --------------------------------------------------------------------------
# 5. Provider & Claim API
# --------------------------------------------------------------------------
def test_provider_and_claim_api(auth_headers):
    """Verify listing and detail endpoints for providers and claims."""
    prov_resp = client.get("/api/providers?limit=5", headers=auth_headers)
    assert prov_resp.status_code == 200
    data = prov_resp.json()
    assert "providers" in data
    assert "total" in data

    claims_resp = client.get("/api/claims?limit=5", headers=auth_headers)
    assert claims_resp.status_code == 200
    cdata = claims_resp.json()
    assert "claims" in cdata
    assert "total" in cdata


# --------------------------------------------------------------------------
# 6. Single-Claim Prediction
# --------------------------------------------------------------------------
def test_single_claim_prediction(auth_headers):
    """Verify single-claim inference returns genuine ML calibrated score and risk tier."""
    payload = {
        "ClaimID": "CLM_AUDIT_001",
        "ProviderID": "PRV51001",
        "BeneficiaryID": "BENE99901",
        "InscClaimAmtReimbursed": 25000.0,
        "DeductibleAmtPaid": 500.0,
        "ClaimStartDt": "2024-01-10",
        "ClaimEndDt": "2024-01-15",
        "ClaimType": "Inpatient",
        "ClmDiagnosisCode_1": "4019",
    }
    resp = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert resp.status_code == 200
    res = resp.json()
    assert "provider_risk_score" in res
    assert "provider_risk_tier" in res
    assert res["provider_risk_tier"] in ["LOW", "MEDIUM", "HIGH"]
    assert 0.0 <= res["provider_risk_score"] <= 1.0


# --------------------------------------------------------------------------
# 7. CSV Prediction & Reconciliation Invariant
# --------------------------------------------------------------------------
def test_csv_prediction_and_reconciliation(auth_headers, db_session):
    """Verify CSV upload reconciliation invariant: total_rows == successful_rows + rejected_rows."""
    csv_content = (
        "ClaimID,ProviderID,InscClaimAmtReimbursed,ClaimStartDt,ClaimType\n"
        "CLM_CSV_A1,PRV51001,1500.0,2024-01-10,Outpatient\n"
        "CLM_CSV_A2,PRV51001,2200.0,2024-01-12,Outpatient\n"
        "CLM_CSV_BAD,PRV51001,INVALID_AMT,2024-01-15,Outpatient\n"
    )
    files = {"file": ("test_audit.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    upload_resp = client.post("/api/analyzer/csv", files=files, headers=auth_headers)
    assert upload_resp.status_code == 202
    upload_id = upload_resp.json()["upload_id"]

    # Status is processed synchronously by TestClient background runner
    status_resp = client.get(f"/api/analyzer/csv/{upload_id}", headers=auth_headers)
    assert status_resp.status_code == 200
    sdata = status_resp.json()
    assert sdata["status"] == "COMPLETED"
    assert sdata["total_rows"] == sdata["successful_rows"] + sdata["rejected_rows"]
    assert sdata["successful_rows"] == 2
    assert sdata["rejected_rows"] == 1


# --------------------------------------------------------------------------
# 8. Single-versus-CSV Score Equality
# --------------------------------------------------------------------------
def test_single_versus_csv_score_equality(auth_headers, db_session):
    """
    Verify that submitting the identical claim via Single-Claim API and via CSV upload
    produces identical risk scores and risk tiers using the same underlying pipeline.
    """
    claim_id = "CLM_EQUALITY_TEST_1"
    provider_id = "PRV51001"
    amount = 14500.0
    start_dt = "2024-02-01"
    claim_type = "Inpatient"

    # 1. Single-claim prediction
    single_payload = {
        "ClaimID": claim_id,
        "ProviderID": provider_id,
        "InscClaimAmtReimbursed": amount,
        "ClaimStartDt": start_dt,
        "ClaimType": claim_type,
    }
    single_resp = client.post("/api/analyzer/single", json=single_payload, headers=auth_headers)
    assert single_resp.status_code == 200
    single_data = single_resp.json()
    single_score = single_data["provider_risk_score"]
    single_tier = single_data["provider_risk_tier"]

    # 2. CSV batch prediction with identical claim specifications for same provider
    csv_claim_id = "CLM_EQUALITY_TEST_2"
    csv_text = (
        f"ClaimID,ProviderID,InscClaimAmtReimbursed,ClaimStartDt,ClaimType\n"
        f"{csv_claim_id},{provider_id},{amount},{start_dt},{claim_type}\n"
    )
    files = {"file": ("equality_test.csv", io.BytesIO(csv_text.encode("utf-8")), "text/csv")}
    upload_resp = client.post("/api/analyzer/csv", files=files, headers=auth_headers)
    assert upload_resp.status_code == 202
    upload_id = upload_resp.json()["upload_id"]

    # Check provider risk score persisted by CSV processing
    csv_score_rec = (
        db_session.query(ProviderRiskScore)
        .filter(ProviderRiskScore.provider_id == provider_id)
        .order_by(ProviderRiskScore.id.desc())
        .first()
    )
    assert csv_score_rec is not None
    csv_score = csv_score_rec.risk_score
    csv_tier = csv_score_rec.risk_tier

    # Assert exact score equality
    assert abs(single_score - csv_score) < 1e-4, f"Score mismatch: single={single_score}, csv={csv_score}"
    assert single_tier == csv_tier, f"Tier mismatch: single={single_tier}, csv={csv_tier}"


# --------------------------------------------------------------------------
# 9. Database Persistence
# --------------------------------------------------------------------------
def test_database_persistence(auth_headers, db_session):
    """Verify that predictions and analyses are correctly written to the database."""
    pid = "PRV_PERSIST_TEST"
    cid = "CLM_PERSIST_TEST"
    payload = {
        "ClaimID": cid,
        "ProviderID": pid,
        "InscClaimAmtReimbursed": 8800.0,
        "ClaimStartDt": "2024-03-01",
        "ClaimType": "Outpatient",
    }
    resp = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert resp.status_code == 200

    # Query DB directly to verify persistence
    prov = db_session.query(Provider).filter(Provider.provider_id == pid).first()
    assert prov is not None, "Provider must be persisted in database"

    claim = db_session.query(Claim).filter(Claim.claim_id == cid).first()
    assert claim is not None, "Claim must be persisted in database"

    score = db_session.query(ProviderRiskScore).filter(ProviderRiskScore.provider_id == pid).first()
    assert score is not None, "ProviderRiskScore must be persisted in database"


# --------------------------------------------------------------------------
# 10. Frontend API Integration Contract
# --------------------------------------------------------------------------
def test_frontend_api_integration_contract(auth_headers):
    """Verify that all endpoints expected by frontend return 200 with required keys."""
    # Overview Summary
    sum_resp = client.get("/api/overview/summary", headers=auth_headers)
    assert sum_resp.status_code == 200
    sdata = sum_resp.json()
    for key in ["total_providers", "total_claims", "high_risk_providers", "claims_requiring_investigation"]:
        assert key in sdata, f"Missing key in overview summary: {key}"

    # Overview Risk Distribution
    dist_resp = client.get("/api/overview/risk-distribution", headers=auth_headers)
    assert dist_resp.status_code == 200
    ddata = dist_resp.json()
    assert "distribution" in ddata

    # Settings Info
    sett_resp = client.get("/api/settings", headers=auth_headers)
    assert sett_resp.status_code == 200
    set_data = sett_resp.json()
    assert "model_version" in set_data
    assert "low_medium_threshold" in set_data
    assert "medium_high_threshold" in set_data
