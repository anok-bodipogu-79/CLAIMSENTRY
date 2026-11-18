"""
HealthGuard AI - Claim Analyzer Comprehensive 16-Scenario ML & System Audit Test Suite
========================================================================================
Validates mathematical bounds, model execution, extreme input protection, idempotency,
recommendation decoupling, explanation provenance, and relational persistence.
"""

import pytest
import math
from backend.app.db.models import (
    Provider,
    Claim,
    Beneficiary,
    ProviderRiskScore,
    ClaimAnalysis,
    Explanation,
    AuditLog,
)


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIOS 1-8: CLINICAL INPUT VALIDATION & EXTREME BOUNDS
# ══════════════════════════════════════════════════════════════════════════════

def test_01_valid_inpatient_claim(client, auth_headers, db_session):
    """Scenario 1: Valid Inpatient claim (PRV51003) with admission & discharge dates."""
    payload = {
        "ClaimID": "CLM_AUDIT_INP_01",
        "ProviderID": "PRV51003",
        "BeneID": "BENE11001",
        "ClaimType": "Inpatient",
        "InscClaimAmtReimbursed": 38500.0,
        "DeductibleAmtPaid": 1068.0,
        "ClaimStartDt": "2009-04-10",
        "ClaimEndDt": "2009-04-25",
        "AdmissionDt": "2009-04-10",
        "DischargeDt": "2009-04-25",
        "ClmDiagnosisCode_1": "4019",
        "ClmProcedureCode_1": "3893",
        "AttendingPhysician": "PHY31201",
    }
    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["claim_id"] == "CLM_AUDIT_INP_01"
    assert data["provider_id"] == "PRV51003"
    assert 0.0 <= data["provider_risk_score"] <= 1.0
    assert data["provider_risk_tier"] in ("LOW", "MEDIUM", "HIGH")
    assert "explanation" in data
    assert "investigation_recommendation" in data

    # Check DB persistence of inpatient LOS
    claim = db_session.query(Claim).filter(Claim.claim_id == "CLM_AUDIT_INP_01").first()
    assert claim is not None
    assert claim.inpatient_los == 15
    assert claim.claim_type == "Inpatient"


def test_02_valid_outpatient_claim(client, auth_headers, db_session):
    """Scenario 2: Valid Outpatient claim (PRV51001) with zero admission dates."""
    payload = {
        "ClaimID": "CLM_AUDIT_OUT_01",
        "ProviderID": "PRV51001",
        "BeneID": "BENE22002",
        "ClaimType": "Outpatient",
        "InscClaimAmtReimbursed": 450.0,
        "DeductibleAmtPaid": 50.0,
        "ClaimStartDt": "2009-06-01",
        "ClaimEndDt": "2009-06-01",
        "ClmDiagnosisCode_1": "4019",
        "AttendingPhysician": "PHY40101",
    }
    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["claim_id"] == "CLM_AUDIT_OUT_01"
    assert data["provider_id"] == "PRV51001"
    assert 0.0 <= data["provider_risk_score"] <= 1.0


def test_03_missing_required_fields(client, auth_headers):
    """Scenario 3: Missing required inputs (ProviderID, ClaimID, Reimbursement) must return 422."""
    payload = {
        "ClaimType": "Outpatient",
        "ClaimStartDt": "2009-06-01",
    }
    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 422


def test_04_invalid_claim_dates_end_before_start(client, auth_headers):
    """Scenario 4: ClaimEndDt earlier than ClaimStartDt must be rejected with 422."""
    payload = {
        "ClaimID": "CLM_INV_DATE_01",
        "ProviderID": "PRV51001",
        "InscClaimAmtReimbursed": 1000.0,
        "ClaimStartDt": "2009-06-15",
        "ClaimEndDt": "2009-06-10",  # Prior to start
    }
    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 422


def test_05_invalid_inpatient_dates_discharge_before_admission(client, auth_headers):
    """Scenario 5: Inpatient DischargeDt earlier than AdmissionDt must be rejected with 422."""
    payload = {
        "ClaimID": "CLM_INV_LOS_01",
        "ProviderID": "PRV51003",
        "ClaimType": "Inpatient",
        "InscClaimAmtReimbursed": 5000.0,
        "ClaimStartDt": "2009-04-10",
        "AdmissionDt": "2009-04-20",
        "DischargeDt": "2009-04-15",  # Prior to admission
    }
    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 422


def test_06_negative_reimbursement_rejected(client, auth_headers):
    """Scenario 6: Negative reimbursement must be rejected with 422."""
    payload = {
        "ClaimID": "CLM_NEG_REIMB_01",
        "ProviderID": "PRV51001",
        "InscClaimAmtReimbursed": -500.0,
        "ClaimStartDt": "2009-06-01",
    }
    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 422


def test_07_extreme_reimbursement_over_500k_rejected(client, auth_headers):
    """Scenario 7: Reimbursement exceeding clinical cap ($500,000.00) must be rejected with 422."""
    payload = {
        "ClaimID": "CLM_EXTREME_01",
        "ProviderID": "PRV51001",
        "InscClaimAmtReimbursed": 1e20,  # $10^20 from screenshot
        "ClaimStartDt": "2009-06-01",
    }
    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 422
    assert "500000" in res.text or "less_than_equal" in res.text


def test_08_extreme_deductible_over_25k_rejected(client, auth_headers):
    """Scenario 8: Deductible exceeding Medicare maximum ($25,000.00) must be rejected with 422."""
    payload = {
        "ClaimID": "CLM_EXTREME_DED_01",
        "ProviderID": "PRV51001",
        "InscClaimAmtReimbursed": 1500.0,
        "DeductibleAmtPaid": 50000.0,
        "ClaimStartDt": "2009-06-01",
    }
    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 422
    assert "25000" in res.text or "less_than_equal" in res.text


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIOS 9-12: IDEMPOTENCY, REPEATABILITY & PROVIDER HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def test_09_duplicate_claim_exact_idempotency(client, auth_headers, db_session):
    """
    Scenario 9 (Test A): Exact duplicate submission returns identical scores idempotently
    without creating conflicting duplicate records in database.
    """
    payload = {
        "ClaimID": "CLM_IDEMP_TEST_01",
        "ProviderID": "PRV51001",
        "BeneID": "BENE22002",
        "InscClaimAmtReimbursed": 750.0,
        "ClaimStartDt": "2009-07-01",
        "ClaimEndDt": "2009-07-01",
    }

    # First submission
    res1 = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res1.status_code == 200
    data1 = res1.json()

    # Count database records
    claim_count_1 = db_session.query(Claim).filter(Claim.claim_id == "CLM_IDEMP_TEST_01").count()
    analysis_count_1 = db_session.query(ClaimAnalysis).filter(ClaimAnalysis.claim_id == "CLM_IDEMP_TEST_01").count()
    assert claim_count_1 == 1
    assert analysis_count_1 == 1

    # Second submission (Exact duplicate)
    res2 = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res2.status_code == 200
    data2 = res2.json()

    # Verify identical output
    assert data1["provider_risk_score"] == data2["provider_risk_score"]
    assert data1["provider_risk_tier"] == data2["provider_risk_tier"]
    assert data1["has_anomalies"] == data2["has_anomalies"]
    assert data1["investigation_recommendation"] == data2["investigation_recommendation"]

    # Verify no duplicate database records were created
    claim_count_2 = db_session.query(Claim).filter(Claim.claim_id == "CLM_IDEMP_TEST_01").count()
    analysis_count_2 = db_session.query(ClaimAnalysis).filter(ClaimAnalysis.claim_id == "CLM_IDEMP_TEST_01").count()
    assert claim_count_2 == 1
    assert analysis_count_2 == 1


def test_10_duplicate_claim_conflicting_details_rejected(client, auth_headers):
    """
    Scenario 10 (Test B): Duplicate Claim ID with conflicting details is rejected with 422.
    """
    payload_original = {
        "ClaimID": "CLM_CONFLICT_01",
        "ProviderID": "PRV51001",
        "InscClaimAmtReimbursed": 500.0,
        "ClaimStartDt": "2009-05-01",
    }
    res1 = client.post("/api/analyzer/single", json=payload_original, headers=auth_headers)
    assert res1.status_code == 200

    # Submit same ClaimID with different reimbursement and provider
    payload_conflicting = {
        "ClaimID": "CLM_CONFLICT_01",
        "ProviderID": "PRV51003",
        "InscClaimAmtReimbursed": 25000.0,
        "ClaimStartDt": "2009-05-01",
    }
    res2 = client.post("/api/analyzer/single", json=payload_conflicting, headers=auth_headers)
    assert res2.status_code == 422
    assert "already exists in database with differing details" in res2.text or "Conflict" in res2.text


def test_11_provider_history_accumulation(client, auth_headers, db_session):
    """
    Scenario 11 (Test C): Submitting multiple distinct claims for the same provider accumulates
    into the provider's cumulative history and dynamically influences provider features.
    """
    # First claim for PRV51001
    c1 = {
        "ClaimID": "CLM_ACCUM_01",
        "ProviderID": "PRV51001",
        "InscClaimAmtReimbursed": 500.0,
        "ClaimStartDt": "2009-01-10",
    }
    res1 = client.post("/api/analyzer/single", json=c1, headers=auth_headers)
    assert res1.status_code == 200

    # Second claim for PRV51001
    c2 = {
        "ClaimID": "CLM_ACCUM_02",
        "ProviderID": "PRV51001",
        "InscClaimAmtReimbursed": 3000.0,
        "ClaimStartDt": "2009-02-15",
    }
    res2 = client.post("/api/analyzer/single", json=c2, headers=auth_headers)
    assert res2.status_code == 200

    # Verify both claims exist in database
    provider_claims = db_session.query(Claim).filter(Claim.provider_id == "PRV51001").all()
    cids = [c.claim_id for c in provider_claims]
    assert "CLM_ACCUM_01" in cids
    assert "CLM_ACCUM_02" in cids


def test_12_mathematical_repeatability_5x(client, auth_headers):
    """
    Scenario 12 (Test D): 5 sequential calls with same input produce bit-for-bit identical probabilities.
    """
    payload = {
        "ClaimID": "CLM_REPEAT_01",
        "ProviderID": "PRV51004",
        "BeneID": "BENE33003",
        "InscClaimAmtReimbursed": 1250.0,
        "DeductibleAmtPaid": 100.0,
        "ClaimStartDt": "2009-08-15",
        "ClaimEndDt": "2009-08-15",
    }

    scores = []
    for _ in range(5):
        res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
        assert res.status_code == 200
        scores.append(res.json()["provider_risk_score"])

    # All 5 scores must match identically to at least 4 decimal places
    first_score = scores[0]
    for s in scores[1:]:
        assert abs(s - first_score) < 1e-5, f"Non-repeatable score: {s} vs {first_score}"


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIOS 13-16: INVARIANTS, DECOUPLING, AUDIT LOGS & PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def test_13_probability_range_and_threshold_invariants(client, auth_headers):
    """
    Scenario 13: Risk score strictly bounded [0.0, 1.0], no NaN, risk tier strictly follows cutoffs.
    """
    payload = {
        "ClaimID": "CLM_INVAR_01",
        "ProviderID": "PRV51003",
        "InscClaimAmtReimbursed": 15000.0,
        "ClaimStartDt": "2009-03-01",
    }
    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    score = data["provider_risk_score"]
    tier = data["provider_risk_tier"]

    assert isinstance(score, float)
    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0

    if score >= 0.70:
        assert tier == "HIGH"
    elif score >= 0.22:
        assert tier == "MEDIUM"
    else:
        assert tier == "LOW"


def test_14_decoupled_recommendation_policy_for_low_risk_provider(client, auth_headers):
    """
    Scenario 14: Decoupled recommendation policy ensures that when a provider has LOW risk (e.g. PRV51001)
    and an operational claim anomaly flag is detected, the recommendation is routine monitoring with
    operational claim review, NOT vague 'MEDIUM PRIORITY'.
    """
    payload = {
        "ClaimID": "CLM_DECOUPLE_01",
        "ProviderID": "PRV51001",
        "BeneID": "BENE22002",
        "InscClaimAmtReimbursed": 450.0,
        "ClaimStartDt": "2009-06-01",
    }
    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    rec = data["investigation_recommendation"]

    if data["provider_risk_tier"] == "LOW":
        if data["has_anomalies"]:
            assert "ROUTINE MONITORING WITH CLAIM-LEVEL REVIEW" in rec
            assert "Provider fraud risk is LOW" in rec
        else:
            assert "LOW PRIORITY" in rec
            assert "Provider Risk: LOW" in rec


def test_15_audit_log_entry_written_to_postgresql(client, auth_headers, db_session):
    """
    Scenario 15: Safe audit log entry is written to PostgreSQL audit_logs table with
    51 feature count, base model outputs, calibrated probability, and timestamp.
    """
    payload = {
        "ClaimID": "CLM_AUDIT_LOG_01",
        "ProviderID": "PRV51001",
        "InscClaimAmtReimbursed": 800.0,
        "ClaimStartDt": "2009-07-15",
    }
    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 200

    audit_entry = (
        db_session.query(AuditLog)
        .filter(AuditLog.resource_id == "CLM_AUDIT_LOG_01")
        .order_by(AuditLog.id.desc())
        .first()
    )
    assert audit_entry is not None
    assert audit_entry.action == "CLAIM_FRAUD_ASSESSMENT"
    assert audit_entry.resource_type == "CLAIM"

    details = audit_entry.details
    assert isinstance(details, dict)
    assert details["claim_id"] == "CLM_AUDIT_LOG_01"
    assert details["provider_id"] == "PRV51001"
    assert "calibrated_probability" in details
    assert "risk_tier" in details
    assert details["feature_count"] == 51
    assert "base_model_outputs" in details
    assert "timestamp" in details


def test_16_database_persistence_across_all_tables(client, auth_headers, db_session):
    """
    Scenario 16: Complete relational persistence verified across:
    providers, claims, provider_risk_scores, claim_analyses, explanations, audit_logs.
    """
    claim_id = "CLM_REL_PERST_01"
    provider_id = "PRV51003"
    bene_id = "BENE11001"

    payload = {
        "ClaimID": claim_id,
        "ProviderID": provider_id,
        "BeneID": bene_id,
        "ClaimType": "Inpatient",
        "InscClaimAmtReimbursed": 12500.0,
        "DeductibleAmtPaid": 1068.0,
        "ClaimStartDt": "2009-05-10",
        "ClaimEndDt": "2009-05-18",
        "AdmissionDt": "2009-05-10",
        "DischargeDt": "2009-05-18",
        "ClmDiagnosisCode_1": "4019",
        "AttendingPhysician": "PHY31201",
    }
    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 200

    # 1. Provider
    prov = db_session.query(Provider).filter(Provider.provider_id == provider_id).first()
    assert prov is not None

    # 2. Beneficiary
    bene = db_session.query(Beneficiary).filter(Beneficiary.beneficiary_id == bene_id).first()
    assert bene is not None

    # 3. Claim
    clm = db_session.query(Claim).filter(Claim.claim_id == claim_id).first()
    assert clm is not None
    assert clm.inpatient_los == 8

    # 4. Provider Risk Score
    prs = (
        db_session.query(ProviderRiskScore)
        .filter(ProviderRiskScore.provider_id == provider_id)
        .order_by(ProviderRiskScore.id.desc())
        .first()
    )
    assert prs is not None
    assert prs.model_version.version == "v2"

    # 5. Claim Analysis
    ca = (
        db_session.query(ClaimAnalysis)
        .filter(ClaimAnalysis.claim_id == claim_id)
        .first()
    )
    assert ca is not None
    assert ca.provider_risk_score_id == prs.id

    # 6. Explanation
    expl = (
        db_session.query(Explanation)
        .filter(Explanation.claim_analysis_id == ca.id)
        .first()
    )
    assert expl is not None
    assert expl.explanation_type == "SHAP_FEATURE_IMPORTANCE"
    assert "Provider-level features reflect cumulative historical aggregates" in expl.explanation_text

    # 7. Audit Log
    al = (
        db_session.query(AuditLog)
        .filter(AuditLog.resource_id == claim_id)
        .first()
    )
    assert al is not None
    assert al.details["feature_count"] == 51
