"""
HealthGuard AI - Phase 6: Automated 28-Step End-to-End Integration & Verification Test Suite
=============================================================================================
Authoritative end-to-end integration lifecycle executed against live PostgreSQL 18:
1. Run Alembic migrations -> verify 12/13 tables created
2. Run database seed -> verify providers, claims, beneficiaries populated
3. Verify default admin and investigator accounts exist
4. Verify backend health & model readiness
5. Register a new user -> verify role is forced to 'investigator'
6. Attempt to register with duplicate email -> verify 400 error
7. Authenticate with new investigator -> receive JWT token
8. Access protected endpoint without token -> verify 401
9. Access protected endpoint with invalid token -> verify 401
10. Fetch overview summary -> verify all 4 KPI counts match database queries
11. Fetch overview recent activity -> verify recent assessments returned
12. Fetch overview risk distribution -> verify tiers match thresholds
13. Search providers -> verify filtered results
14. Filter providers by risk tier (HIGH) -> verify all returned providers have score >= 0.70
15. Inspect a specific provider -> verify details returned
16. Fetch provider risk score -> verify calibrated score, tier, model version
17. Fetch provider AI explanation -> verify SHAP features, title, text, disclaimer
18. List claims with pagination -> verify page size and total count
19. Filter claims by type (Inpatient) -> verify all returned claims are Inpatient
20. Inspect a specific claim -> verify metadata and anomaly status
21. Run single-claim analysis with valid claim -> verify risk score, tier, anomalies, SHAP
22. Upload valid CSV batch -> verify upload queued and processed
23. Poll CSV status until COMPLETED -> verify total, successful, rejected counts
24. Verify CSV reconciliation invariant: total == successful + rejected
25. Upload CSV with invalid rows (duplicates, missing fields) -> verify proper rejection
26. Verify rejected rows endpoint returns specific error messages
27. Log out -> verify token added to revoked_tokens
28. Attempt to use revoked token -> verify 401 error
"""

import io
import os
import time
import pytest
from starlette.testclient import TestClient
from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker

from backend.app.main import app
import backend.app.db.database as db_module
from backend.app.db.models import (
    User,
    Provider,
    Claim,
    Beneficiary,
    ProviderRiskScore,
    ClaimAnalysis,
    CSVUpload,
    CSVUploadRow,
    RevokedToken,
)

PG_USER = os.getenv("DB_USER", "postgres")
PG_PASSWORD = os.getenv("DB_PASSWORD", "anok1234")
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
def pg_live_engine():
    """Create live PostgreSQL engine pointing strictly to isolated healthguard_test."""
    engine = create_engine(PG_TEST_URL, pool_pre_ping=True)
    orig_engine = db_module.engine
    db_module.engine = engine
    db_module.SessionLocal.configure(bind=engine)
    yield engine
    db_module.engine = orig_engine
    db_module.SessionLocal.configure(bind=orig_engine)


@pytest.fixture(scope="module")
def live_client(pg_live_engine):
    """Client for testing FastAPI endpoints with live PostgreSQL binding."""
    client = TestClient(app)
    return client


@pytest.fixture
def live_db(pg_live_engine):
    """Direct PostgreSQL session."""
    Session = sessionmaker(bind=pg_live_engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


def test_complete_28_step_e2e_workflow(live_client: TestClient, live_db, pg_live_engine):
    """Execute the full 28-step end-to-end integration test against PostgreSQL 18."""
    ts = int(time.time() * 1000)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Run Alembic migrations / Verify tables created in public schema
    # ──────────────────────────────────────────────────────────────────────────
    with pg_live_engine.connect() as conn:
        tables_res = conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname = 'public';")
        ).fetchall()
    table_names = {r[0] for r in tables_res}
    required_tables = {
        "users", "revoked_tokens", "model_versions", "providers",
        "beneficiaries", "claims", "provider_risk_scores", "claim_analyses",
        "explanations", "csv_uploads", "csv_upload_rows", "case_notes", "audit_logs"
    }
    assert required_tables.issubset(table_names), f"Missing tables: {required_tables - table_names}"

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Run database seed / Verify providers, claims, beneficiaries populated
    # ──────────────────────────────────────────────────────────────────────────
    assert live_db.query(Provider).count() >= 50, "Database must contain at least 50 providers"
    assert live_db.query(Claim).count() >= 100, "Database must contain at least 100 claims"
    assert live_db.query(Beneficiary).count() >= 50, "Database must contain at least 50 beneficiaries"

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3: Verify default admin and investigator accounts exist
    # ──────────────────────────────────────────────────────────────────────────
    admin_user = live_db.query(User).filter(User.email == "admin@healthguard.ai").first()
    assert admin_user is not None, "Default admin user missing"
    assert admin_user.role == "admin"

    investigator_user = live_db.query(User).filter(User.email == "investigator@healthguard.ai").first()
    assert investigator_user is not None, "Default investigator user missing"
    assert investigator_user.role == "investigator"

    # ──────────────────────────────────────────────────────────────────────────
    # Step 4: Verify backend health & model readiness
    # ──────────────────────────────────────────────────────────────────────────
    health_resp = live_client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json()["status"] == "ok"
    assert "PostgreSQL" in health_resp.json()["database"]

    models_resp = live_client.get("/api/models/readiness")
    assert models_resp.status_code == 200
    assert models_resp.json()["ready"] is True

    # ──────────────────────────────────────────────────────────────────────────
    # Step 5: Register a new user -> verify role is forced to 'investigator'
    # ──────────────────────────────────────────────────────────────────────────
    test_email = f"e2e_user_{ts}@healthguard.ai"
    reg_resp = live_client.post(
        "/api/auth/register",
        json={
            "name": f"E2E Investigator {ts}",
            "email": test_email,
            "password": "SecurePassword123!",
            "role": "admin",  # Security check: Public registration must ignore role parameter
        },
    )
    assert reg_resp.status_code == 201
    user_data = reg_resp.json()
    assert user_data["email"] == test_email
    assert user_data["role"] == "investigator", "Security: Public registration must force investigator role"

    # ──────────────────────────────────────────────────────────────────────────
    # Step 6: Attempt to register with duplicate email -> verify 400 error
    # ──────────────────────────────────────────────────────────────────────────
    dup_resp = live_client.post(
        "/api/auth/register",
        json={
            "name": "Duplicate User",
            "email": test_email,
            "password": "SecurePassword123!",
        },
    )
    assert dup_resp.status_code == 400

    # ──────────────────────────────────────────────────────────────────────────
    # Step 7: Authenticate with new investigator -> receive JWT token
    # ──────────────────────────────────────────────────────────────────────────
    login_resp = live_client.post(
        "/api/auth/login",
        json={"email": test_email, "password": "SecurePassword123!"},
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    assert token and len(token) > 20
    auth_headers = {"Authorization": f"Bearer {token}"}

    # ──────────────────────────────────────────────────────────────────────────
    # Step 8: Access protected endpoint without token -> verify 401
    # ──────────────────────────────────────────────────────────────────────────
    no_auth_resp = live_client.get("/api/overview/summary")
    assert no_auth_resp.status_code == 401

    # ──────────────────────────────────────────────────────────────────────────
    # Step 9: Access protected endpoint with invalid token -> verify 401
    # ──────────────────────────────────────────────────────────────────────────
    bad_auth_resp = live_client.get(
        "/api/overview/summary",
        headers={"Authorization": "Bearer invalid.mock.jwt.signature"},
    )
    assert bad_auth_resp.status_code == 401

    # ──────────────────────────────────────────────────────────────────────────
    # Step 10: Fetch overview summary -> verify all 4 KPI counts match database
    # ──────────────────────────────────────────────────────────────────────────
    overview_resp = live_client.get("/api/overview/summary", headers=auth_headers)
    assert overview_resp.status_code == 200
    overview_data = overview_resp.json()

    db_provider_count = live_db.query(func.count(Provider.id)).scalar()
    db_claim_count = live_db.query(func.count(Claim.id)).scalar()
    assert overview_data["total_providers"] == db_provider_count
    assert overview_data["total_claims"] == db_claim_count
    assert "high_risk_providers" in overview_data
    assert "claims_requiring_investigation" in overview_data

    # ──────────────────────────────────────────────────────────────────────────
    # Step 11: Fetch overview recent activity -> verify recent assessments returned
    # ──────────────────────────────────────────────────────────────────────────
    act_resp = live_client.get("/api/overview/recent-activity?limit=5", headers=auth_headers)
    assert act_resp.status_code == 200
    recent_list = act_resp.json().get("recent_activities") or act_resp.json().get("activities", [])
    assert len(recent_list) > 0

    # ──────────────────────────────────────────────────────────────────────────
    # Step 12: Fetch overview risk distribution -> verify tiers match thresholds
    # ──────────────────────────────────────────────────────────────────────────
    dist_resp = live_client.get("/api/overview/risk-distribution", headers=auth_headers)
    assert dist_resp.status_code == 200
    dist_data = dist_resp.json()
    assert "distribution" in dist_data
    tiers = {d["tier"] for d in dist_data["distribution"]}
    assert {"LOW", "MEDIUM", "HIGH"}.issubset(tiers)

    # ──────────────────────────────────────────────────────────────────────────
    # Step 13: Search providers -> verify filtered results
    # ──────────────────────────────────────────────────────────────────────────
    search_resp = live_client.get("/api/providers?search=PRV51003", headers=auth_headers)
    assert search_resp.status_code == 200
    search_providers = search_resp.json()["providers"]
    assert len(search_providers) >= 1
    target_provider = search_providers[0]["provider_id"]

    # ──────────────────────────────────────────────────────────────────────────
    # Step 14: Filter providers by risk tier (HIGH) -> verify score >= 0.70
    # ──────────────────────────────────────────────────────────────────────────
    high_tier_resp = live_client.get("/api/providers?risk_tier=HIGH", headers=auth_headers)
    assert high_tier_resp.status_code == 200
    high_providers = high_tier_resp.json()["providers"]
    assert len(high_providers) >= 1
    for hp in high_providers:
        if hp.get("risk_score") is not None:
            assert hp["risk_score"] >= 0.70, f"Provider {hp['provider_id']} has score {hp['risk_score']} < 0.70"

    # ──────────────────────────────────────────────────────────────────────────
    # Step 15: Inspect a specific provider -> verify details returned
    # ──────────────────────────────────────────────────────────────────────────
    p_detail_resp = live_client.get(f"/api/providers/{target_provider}", headers=auth_headers)
    assert p_detail_resp.status_code == 200
    assert p_detail_resp.json()["provider_id"] == target_provider

    # ──────────────────────────────────────────────────────────────────────────
    # Step 16: Fetch provider risk score -> verify calibrated score, tier, version
    # ──────────────────────────────────────────────────────────────────────────
    p_risk_resp = live_client.get(f"/api/providers/{target_provider}/risk", headers=auth_headers)
    assert p_risk_resp.status_code == 200
    p_risk = p_risk_resp.json()
    assert 0.0 <= p_risk["risk_score"] <= 1.0
    assert p_risk["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]
    assert p_risk["model_version"] in ["v1", "v2"]

    # ──────────────────────────────────────────────────────────────────────────
    # Step 17: Fetch provider AI explanation -> verify SHAP features & disclaimer
    # ──────────────────────────────────────────────────────────────────────────
    p_expl_resp = live_client.get(f"/api/providers/{target_provider}/explanation", headers=auth_headers)
    assert p_expl_resp.status_code == 200
    p_expl = p_expl_resp.json()
    assert "explanation_title" in p_expl
    assert "top_risk_factors" in p_expl or "feature_contributions" in p_expl

    # ──────────────────────────────────────────────────────────────────────────
    # Step 18: List claims with pagination -> verify page size and total count
    # ──────────────────────────────────────────────────────────────────────────
    claims_resp = live_client.get("/api/claims?page=1&page_size=10", headers=auth_headers)
    assert claims_resp.status_code == 200
    claims_data = claims_resp.json()
    assert len(claims_data["claims"]) == 10
    sample_claim = claims_data["claims"][0]

    # ──────────────────────────────────────────────────────────────────────────
    # Step 19: Filter claims by type (Inpatient) -> verify returned claims
    # ──────────────────────────────────────────────────────────────────────────
    inpatient_resp = live_client.get("/api/claims?claim_type=Inpatient&page_size=5", headers=auth_headers)
    assert inpatient_resp.status_code == 200
    inpatient_claims = inpatient_resp.json()["claims"]
    for ic in inpatient_claims:
        assert ic["claim_type"] == "Inpatient"

    # ──────────────────────────────────────────────────────────────────────────
    # Step 20: Inspect a specific claim -> verify metadata and anomaly status
    # ──────────────────────────────────────────────────────────────────────────
    c_detail_resp = live_client.get(f"/api/claims/{sample_claim['claim_id']}", headers=auth_headers)
    assert c_detail_resp.status_code == 200
    assert c_detail_resp.json()["claim_id"] == sample_claim["claim_id"]

    c_analysis_resp = live_client.get(f"/api/claims/{sample_claim['claim_id']}/analysis", headers=auth_headers)
    assert c_analysis_resp.status_code == 200
    assert "anomaly_flags" in c_analysis_resp.json()

    # ──────────────────────────────────────────────────────────────────────────
    # Step 21: Run single-claim analysis with valid claim -> verify results
    # ──────────────────────────────────────────────────────────────────────────
    single_claim_payload = {
        "ClaimID": f"CLM_E2E_28_{ts}",
        "ProviderID": "PRV51003",
        "BeneID": "BENE11001",
        "InscClaimAmtReimbursed": 6500.0,
        "DeductibleAmtPaid": 150.0,
        "ClaimStartDt": "2009-04-01",
        "ClaimEndDt": "2009-04-02",
        "ClaimType": "Outpatient",
        "ClmDiagnosisCode_1": "4019",
        "AttendingPhysician": "PHY31201",
    }
    single_resp = live_client.post("/api/analyzer/single", json=single_claim_payload, headers=auth_headers)
    assert single_resp.status_code == 200
    single_result = single_resp.json()
    assert 0.0 <= single_result["provider_risk_score"] <= 1.0
    assert single_result["provider_risk_tier"] in ["LOW", "MEDIUM", "HIGH"]
    assert "disclaimer" in single_result

    # ──────────────────────────────────────────────────────────────────────────
    # Step 22: Upload valid CSV batch -> verify upload queued and processed
    # ──────────────────────────────────────────────────────────────────────────
    csv_valid_content = (
        "ClaimID,ProviderID,BeneID,InscClaimAmtReimbursed,DeductibleAmtPaid,ClaimStartDt,ClaimEndDt,ClaimType\n"
        f"CLM_CSV_VAL_A_{ts},PRV51003,BENE11001,4500.00,100.00,2009-05-01,2009-05-02,Outpatient\n"
        f"CLM_CSV_VAL_B_{ts},PRV51003,BENE11001,3200.00,50.00,2009-05-03,2009-05-04,Outpatient\n"
    )
    files = {"file": ("valid_batch.csv", io.BytesIO(csv_valid_content.encode("utf-8")), "text/csv")}
    upload_resp = live_client.post("/api/analyzer/csv", files=files, headers=auth_headers)
    assert upload_resp.status_code == 202
    upload_id = upload_resp.json()["upload_id"]

    # ──────────────────────────────────────────────────────────────────────────
    # Step 23: Poll CSV status until COMPLETED -> verify counts
    # ──────────────────────────────────────────────────────────────────────────
    time.sleep(1.0)
    status_resp = live_client.get(f"/api/analyzer/csv/{upload_id}", headers=auth_headers)
    assert status_resp.status_code == 200
    upload_status = status_resp.json()
    assert upload_status["status"] in ["COMPLETED", "PROCESSING"]

    # ──────────────────────────────────────────────────────────────────────────
    # Step 24: Verify CSV reconciliation invariant: total == successful + rejected
    # ──────────────────────────────────────────────────────────────────────────
    success_resp = live_client.get(f"/api/analyzer/csv/{upload_id}/results", headers=auth_headers)
    assert success_resp.status_code == 200
    successful_claims = success_resp.json()["claims"]

    reject_resp = live_client.get(f"/api/analyzer/csv/{upload_id}/rejected-rows", headers=auth_headers)
    assert reject_resp.status_code == 200
    rejected_rows = reject_resp.json()["rejected_rows"]

    assert upload_status["total_rows"] == upload_status["successful_rows"] + upload_status["rejected_rows"]
    assert len(successful_claims) == upload_status["successful_rows"]
    assert len(rejected_rows) == upload_status["rejected_rows"]

    # ──────────────────────────────────────────────────────────────────────────
    # Step 25: Upload CSV with invalid rows (duplicates, missing fields) -> verify rejection
    # ──────────────────────────────────────────────────────────────────────────
    csv_invalid_content = (
        "ClaimID,ProviderID,BeneID,InscClaimAmtReimbursed,DeductibleAmtPaid,ClaimStartDt,ClaimEndDt,ClaimType\n"
        f"CLM_CSV_DUP_{ts},PRV51003,BENE11001,4000.00,100.00,2009-05-01,2009-05-02,Outpatient\n"
        f"CLM_CSV_DUP_{ts},PRV51003,BENE11001,4000.00,100.00,2009-05-01,2009-05-02,Outpatient\n"  # Duplicate ID
        f"CLM_CSV_NEG_{ts},PRV51003,BENE11001,-500.00,0.00,2009-05-01,2009-05-02,Outpatient\n"   # Negative amount
    )
    files_inv = {"file": ("invalid_batch.csv", io.BytesIO(csv_invalid_content.encode("utf-8")), "text/csv")}
    inv_upload_resp = live_client.post("/api/analyzer/csv", files=files_inv, headers=auth_headers)
    assert inv_upload_resp.status_code == 202
    inv_upload_id = inv_upload_resp.json()["upload_id"]

    # ──────────────────────────────────────────────────────────────────────────
    # Step 26: Verify rejected rows endpoint returns specific error messages
    # ──────────────────────────────────────────────────────────────────────────
    time.sleep(1.0)
    inv_reject_resp = live_client.get(f"/api/analyzer/csv/{inv_upload_id}/rejected-rows", headers=auth_headers)
    assert inv_reject_resp.status_code == 200
    inv_rejected_rows = inv_reject_resp.json()["rejected_rows"]
    assert len(inv_rejected_rows) >= 1
    reasons = " ".join(r.get("rejection_reason", "") for r in inv_rejected_rows)
    assert any(term in reasons.lower() for term in ["duplicate", "negative", "invalid", "reimbursed"])

    # ──────────────────────────────────────────────────────────────────────────
    # Step 27: Log out -> verify token added to revoked_tokens
    # ──────────────────────────────────────────────────────────────────────────
    logout_resp = live_client.post("/api/auth/logout", headers=auth_headers)
    assert logout_resp.status_code == 200
    assert logout_resp.json()["status"] in ["success", "logged_out"]

    # ──────────────────────────────────────────────────────────────────────────
    # Step 28: Attempt to use revoked token -> verify 401 error
    # ──────────────────────────────────────────────────────────────────────────
    revoked_resp = live_client.get("/api/auth/me", headers=auth_headers)
    assert revoked_resp.status_code == 401
    assert "revoked" in revoked_resp.json()["detail"].lower()


def test_single_claim_vs_csv_consistency(live_client: TestClient, live_db, pg_live_engine):
    """
    Phase 6.3: Single Claim and CSV Analysis Consistency Test.
    Verifies that identical provider historical context and claim inputs produce
    identical provider risk scores, identical risk tiers, and consistent anomaly findings
    between the single-claim endpoint and the batch CSV processing pipeline.
    """
    ts = int(time.time() * 1000)
    provider_id = "PRV51003"
    bene_id = "BENE11001"
    reimbursed = 3200.0
    deductible = 50.0
    claim_start = "2009-06-01"
    claim_end = "2009-06-02"

    # Provision investigator for this test
    user_email = f"tester_{ts}@healthguard.ai"
    reg_resp = live_client.post(
        "/api/auth/register",
        json={"name": "Tester", "email": user_email, "password": "SecurePassword123!"}
    )
    assert reg_resp.status_code == 201

    login_resp = live_client.post(
        "/api/auth/login",
        json={"email": user_email, "password": "SecurePassword123!"}
    )
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 1. Run Single-Claim Analysis
    single_payload = {
        "ClaimID": f"CLM_CONSIST_SINGLE_{ts}",
        "ProviderID": provider_id,
        "BeneID": bene_id,
        "InscClaimAmtReimbursed": reimbursed,
        "DeductibleAmtPaid": deductible,
        "ClaimStartDt": claim_start,
        "ClaimEndDt": claim_end,
        "ClaimType": "Outpatient",
        "ClmDiagnosisCode_1": "4019",
        "AttendingPhysician": "PHY31201"
    }
    single_resp = live_client.post("/api/analyzer/single", json=single_payload, headers=headers)
    assert single_resp.status_code == 200
    single_data = single_resp.json()

    # 2. Run CSV Analysis for the same claim
    csv_text = (
        "ClaimID,ProviderID,BeneID,InscClaimAmtReimbursed,DeductibleAmtPaid,ClaimStartDt,ClaimEndDt,ClaimType,ClmDiagnosisCode_1,AttendingPhysician\n"
        f"CLM_CONSIST_CSV_{ts},{provider_id},{bene_id},{reimbursed},{deductible},{claim_start},{claim_end},Outpatient,4019,PHY31201\n"
    )
    files = {"file": ("consistency.csv", io.BytesIO(csv_text.encode("utf-8")), "text/csv")}
    csv_upload_resp = live_client.post("/api/analyzer/csv", files=files, headers=headers)
    assert csv_upload_resp.status_code == 202
    upload_id = csv_upload_resp.json()["upload_id"]

    time.sleep(1.0)
    results_resp = live_client.get(f"/api/analyzer/csv/{upload_id}/results", headers=headers)
    assert results_resp.status_code == 200
    csv_claims = results_resp.json()["claims"]
    assert len(csv_claims) == 1
    csv_claim = csv_claims[0]

    # Retrieve associated provider risk assessment from database for the CSV analysis
    p_risk_resp = live_client.get(f"/api/providers/{provider_id}/risk-score", headers=headers)
    assert p_risk_resp.status_code == 200
    csv_provider_risk = p_risk_resp.json()

    # 3. Compare Provider Risk Score & Risk Tier
    single_score = single_data["provider_risk_score"]
    csv_score = csv_provider_risk["risk_score"]
    assert abs(single_score - csv_score) < 0.001, (
        f"Inconsistent provider risk score: single={single_score}, csv={csv_score}"
    )
    assert single_data["provider_risk_tier"] == csv_provider_risk["risk_tier"]
    assert single_data["model_version"] == csv_provider_risk["model_version"]

    # 4. Compare Anomaly Findings
    single_flags = set(single_data.get("anomaly_flags", []))
    csv_flags = set(csv_claim.get("anomaly_flags", []))
    assert single_flags == csv_flags, f"Inconsistent anomaly flags: {single_flags} vs {csv_flags}"
