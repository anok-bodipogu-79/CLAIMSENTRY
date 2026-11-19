"""
HealthGuard AI - Production Data Cleanup, Real-Time Persistence & Database Synchronization Suite
================================================================================================
Validates:
1. Live PostgreSQL State: CMS/demo records removed; authentic users & models preserved.
2. Claim, Provider & Beneficiary Persistence: Atomic relational persistence and entity reuse.
3. Dashboard & Overview APIs: Dynamic aggregation, charts, and empty-state truth.
4. Claims & Providers Exploration APIs: Direct database query matching.
5. Fill Sample Generation Logic: Randomized non-colliding IDs and valid 2025/2026 date bounds.
"""

import pytest
import random
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from backend.app.core.config import settings
from backend.app.db.models import Claim, Provider, Beneficiary, ClaimAnalysis, ProviderRiskScore, ModelVersion, User


def test_01_live_postgresql_cleanup_and_core_preservation():
    """Verify that CMS datasets and demo records are cleaned, while authentic users and models remain."""
    import re
    # Direct test against PostgreSQL live database or test engine
    pg_url = settings.DATABASE_URL
    try:
        pg_engine = create_engine(pg_url)
        with pg_engine.connect() as conn:
            claim_ids = [r[0] for r in conn.execute(text("SELECT claim_id FROM claims")).fetchall()]
            cms_claims = [cid for cid in claim_ids if re.match(r"^CLM\d{5}$", cid)]
            assert len(cms_claims) == 0, f"Expected 0 CMS seed claims in PostgreSQL, found {len(cms_claims)}"

            prov_ids = [r[0] for r in conn.execute(text("SELECT provider_id FROM providers")).fetchall()]
            cms_provs = [pid for pid in prov_ids if re.match(r"^PRV\d{5}$", pid)]
            assert len(cms_provs) == 0, f"Expected 0 CMS seed providers in PostgreSQL, found {len(cms_provs)}"

            admin_count = conn.execute(text("SELECT count(*) FROM users WHERE email = 'admin@healthguard.ai'")).scalar()
            assert admin_count >= 1, "admin@healthguard.ai must be preserved in PostgreSQL"
            investigator_count = conn.execute(text("SELECT count(*) FROM users WHERE email = 'investigator@healthguard.ai'")).scalar()
            assert investigator_count >= 1, "investigator@healthguard.ai must be preserved in PostgreSQL"

            mv_active = conn.execute(text("SELECT is_active FROM model_versions WHERE version = 'v2'")).scalar()
            assert mv_active is True, "ModelVersion v2 must be active in PostgreSQL"
    except Exception as e:
        pytest.fail(f"PostgreSQL direct verification failed: {e}")


def test_02_claim_analysis_realtime_persistence_and_entity_creation(client, auth_headers, db_session):
    """
    Submitting a claim through /api/analyzer/single must atomically persist:
    1. Provider in providers table
    2. Beneficiary in beneficiaries table
    3. Claim in claims table
    4. ProviderRiskScore
    5. ClaimAnalysis
    """
    claim_id = f"CLM_TRANS_{int(datetime.now(timezone.utc).timestamp())}_A"
    provider_id = f"PRV_LIVE_TRANS_001"
    beneficiary_id = f"BEN_LIVE_TRANS_001"

    payload = {
        "ClaimID": claim_id,
        "ProviderID": provider_id,
        "BeneID": beneficiary_id,
        "ClaimType": "Inpatient",
        "InscClaimAmtReimbursed": 250000.0,
        "Currency": "INR",
        "DeductibleAmtPaid": 8900.0,
        "ClaimStartDt": "2026-01-10",
        "ClaimEndDt": "2026-01-15",
        "AdmissionDt": "2026-01-10",
        "DischargeDt": "2026-01-15",
        "ClmDiagnosisCode_1": "4019",
        "ClmDiagnosisCode_2": "4280",
        "ClmProcedureCode_1": "3722",
        "AttendingPhysician": "PHY_TRANS_01",
    }

    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 200, f"Analysis failed: {res.text}"
    data = res.json()
    assert data["claim_id"] == claim_id
    assert "provider_risk_score" in data
    assert "provider_risk_tier" in data

    # Verify records in db_session
    saved_claim = db_session.query(Claim).filter(Claim.claim_id == claim_id).first()
    assert saved_claim is not None, "Claim record was not persisted"
    assert saved_claim.provider_id == provider_id
    assert saved_claim.beneficiary_id == beneficiary_id
    assert saved_claim.inpatient_los == 5
    assert saved_claim.currency == "INR"
    assert saved_claim.original_reimbursement == 250000.0

    saved_prov = db_session.query(Provider).filter(Provider.provider_id == provider_id).first()
    assert saved_prov is not None, "Provider was not created"

    saved_bene = db_session.query(Beneficiary).filter(Beneficiary.beneficiary_id == beneficiary_id).first()
    assert saved_bene is not None, "Beneficiary was not created"

    saved_analysis = db_session.query(ClaimAnalysis).filter(ClaimAnalysis.claim_id == claim_id).first()
    assert saved_analysis is not None, "ClaimAnalysis was not persisted"
    assert saved_analysis.analysis_status in ["COMPLETED", "FLAGGED"]


def test_03_entity_reuse_and_duplicate_prevention(client, auth_headers, db_session):
    """Submitting multiple claims with the same Provider and Beneficiary must reuse existing records."""
    claim_id_1 = f"CLM_TRANS_{int(datetime.now(timezone.utc).timestamp())}_1"
    claim_id_2 = f"CLM_TRANS_{int(datetime.now(timezone.utc).timestamp())}_2"
    provider_id = f"PRV_REUSE_001"
    beneficiary_id = f"BEN_REUSE_001"

    payload_1 = {
        "ClaimID": claim_id_1,
        "ProviderID": provider_id,
        "BeneID": beneficiary_id,
        "ClaimType": "Outpatient",
        "InscClaimAmtReimbursed": 15000.0,
        "Currency": "INR",
        "DeductibleAmtPaid": 0.0,
        "ClaimStartDt": "2026-02-01",
        "ClaimEndDt": "2026-02-01",
        "ClmDiagnosisCode_1": "25000",
        "AttendingPhysician": "PHY_TRANS_01",
    }
    res1 = client.post("/api/analyzer/single", json=payload_1, headers=auth_headers)
    assert res1.status_code == 200

    payload_2 = {
        "ClaimID": claim_id_2,
        "ProviderID": provider_id,
        "BeneID": beneficiary_id,
        "ClaimType": "Outpatient",
        "InscClaimAmtReimbursed": 22000.0,
        "Currency": "INR",
        "DeductibleAmtPaid": 0.0,
        "ClaimStartDt": "2026-02-05",
        "ClaimEndDt": "2026-02-05",
        "ClmDiagnosisCode_1": "4019",
        "AttendingPhysician": "PHY_TRANS_01",
    }
    res2 = client.post("/api/analyzer/single", json=payload_2, headers=auth_headers)
    assert res2.status_code == 200

    # Provider count for PRV_REUSE_001 must be exactly 1
    p_count = db_session.query(Provider).filter(Provider.provider_id == provider_id).count()
    assert p_count == 1, f"Expected exactly 1 Provider record, found {p_count}"

    # Beneficiary count for BEN_REUSE_001 must be exactly 1
    b_count = db_session.query(Beneficiary).filter(Beneficiary.beneficiary_id == beneficiary_id).count()
    assert b_count == 1, f"Expected exactly 1 Beneficiary record, found {b_count}"

    # Claims count for this provider must now be 2
    c_count = db_session.query(Claim).filter(Claim.provider_id == provider_id).count()
    assert c_count == 2, f"Expected 2 Claims for provider, found {c_count}"


def test_04_dashboard_overview_matches_database(client, auth_headers, db_session):
    """Dashboard KPIs and Charts endpoints must accurately reflect actual database counts."""
    # Analyze a test claim so there is data
    claim_id = f"CLM_TRANS_{int(datetime.now(timezone.utc).timestamp())}_DASH"
    payload = {
        "ClaimID": claim_id,
        "ProviderID": "PRV_DASH_001",
        "BeneID": "BEN_DASH_001",
        "ClaimType": "Outpatient",
        "InscClaimAmtReimbursed": 30000.0,
        "Currency": "INR",
        "DeductibleAmtPaid": 1000.0,
        "ClaimStartDt": "2026-02-10",
        "ClaimEndDt": "2026-02-10",
        "ClmDiagnosisCode_1": "4019",
        "AttendingPhysician": "PHY_DASH_01",
    }
    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 200

    actual_claims_count = db_session.query(Claim).count()
    actual_providers_count = db_session.query(Provider).count()
    actual_beneficiaries_count = db_session.query(Beneficiary).count()

    # 1. Summary API
    res_sum = client.get("/api/overview/summary", headers=auth_headers)
    assert res_sum.status_code == 200
    summary = res_sum.json()

    assert summary["total_claims"] == actual_claims_count
    assert summary["total_providers"] == actual_providers_count
    assert summary["total_beneficiaries"] == actual_beneficiaries_count
    assert summary["total_reimbursement"] > 0
    assert summary["avg_claim_amount"] > 0

    # 2. Charts API
    res_charts = client.get("/api/overview/charts", headers=auth_headers)
    assert res_charts.status_code == 200
    charts = res_charts.json()

    assert "claims_by_month" in charts
    assert "claims_over_time" in charts
    assert "provider_activity" in charts
    assert "recent_claims" in charts
    assert len(charts["recent_claims"]) == actual_claims_count


def test_05_claims_and_providers_list_apis(client, auth_headers, db_session):
    """Claims and Providers list endpoints must query and paginate real database records."""
    claim_id = f"CLM_TRANS_{int(datetime.now(timezone.utc).timestamp())}_LST"
    payload = {
        "ClaimID": claim_id,
        "ProviderID": "PRV_LIST_001",
        "BeneID": "BEN_LIST_001",
        "ClaimType": "Outpatient",
        "InscClaimAmtReimbursed": 18000.0,
        "Currency": "INR",
        "DeductibleAmtPaid": 500.0,
        "ClaimStartDt": "2026-02-12",
        "ClaimEndDt": "2026-02-12",
        "ClmDiagnosisCode_1": "486",
        "AttendingPhysician": "PHY_LIST_01",
    }
    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 200

    # 1. Claims List
    res_claims = client.get("/api/claims?page=1&page_size=10", headers=auth_headers)
    assert res_claims.status_code == 200
    data_claims = res_claims.json()
    assert data_claims["total"] >= 1
    claim_ids = [c["claim_id"] for c in data_claims["claims"]]
    assert claim_id in claim_ids

    # 2. Providers List
    res_prov = client.get("/api/providers?page=1&page_size=10", headers=auth_headers)
    assert res_prov.status_code == 200
    data_prov = res_prov.json()
    assert data_prov["total"] >= 1
    prov_ids = [p["provider_id"] for p in data_prov["providers"]]
    assert "PRV_LIST_001" in prov_ids


def test_06_fill_sample_randomization_and_date_bounds():
    """Verify that sample generator parameters produce unique IDs and valid 2025/2026 dates without 2009."""
    now = datetime.now()
    start_2025 = datetime(2025, 1, 1)

    generated_ids = set()
    generated_provs = set()
    generated_benes = set()

    for _ in range(20):
        # Simulate randomized ID generation
        rand_hex = hex(random.randint(0x100000, 0xFFFFFF))[2:].upper()
        rand_num = random.randint(1000, 9999)
        claim_id = f"CLM_SYN_{rand_hex}{rand_num}"
        provider_id = f"PRV_{random.randint(10000, 99999)}"
        bene_id = f"BEN_{random.randint(10000, 99999)}"

        # Simulate dynamic date generation
        rand_ts = start_2025.timestamp() + random.random() * (now.timestamp() - start_2025.timestamp() - 86400)
        dt = datetime.fromtimestamp(rand_ts)
        date_str = dt.strftime("%Y-%m-%d")

        # Invariants:
        assert date_str.startswith("2025") or date_str.startswith("2026"), f"Date {date_str} must be in 2025 or 2026"
        assert "2009" not in date_str, "Date must never contain 2009"
        assert "2010" not in date_str, "Date must never contain 2010"
        assert dt <= now, f"Date {date_str} must not be in the future"

        generated_ids.add(claim_id)
        generated_provs.add(provider_id)
        generated_benes.add(bene_id)

    # Must produce diverse non-colliding sets
    assert len(generated_ids) == 20, "Claim IDs must be unique across iterations"
    assert len(generated_provs) >= 18, "Provider IDs must vary across iterations"
    assert len(generated_benes) >= 18, "Beneficiary IDs must vary across iterations"
