"""
HealthGuard AI - Universal Claim Analysis & Pipeline Comprehensive Tests
========================================================================
Parameterized regression and validation test suite covering:
- Multiple dynamic providers and claims (not just demo fixtures).
- Risk tiers (LOW, MEDIUM, HIGH) and probability boundary validations.
- Missing provider, missing beneficiary, missing features.
- Zero, positive, negative, and invalid reimbursement amounts.
- Temporal sequence dynamics: padding months vs genuine zero-claim months in active window.
- SHAP positive and negative contributor attributions and actual feature values.
- Claim anomaly rules: clean claims (no anomalies) vs single vs multiple anomalies.
- Central response validation enforcement.
- Mixed batch CSV processing and duplicate claim reconciliation.
- Single-claim versus CSV prediction parity.
- Future/dynamically inserted PostgreSQL records.
"""

import math
import uuid
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.main import app
from backend.app.db.database import get_db, SessionLocal
from backend.app.db.models import Provider, Claim, Beneficiary, ProviderRiskScore, ClaimAnalysis, Explanation
from backend.app.core.config import settings
from backend.app.core.security import create_access_token
from backend.app.services.claim_analysis_pipeline import (
    claim_analysis_pipeline,
    UniversalResponseValidator,
    AnalysisValidationError,
)
from backend.app.workers.csv_worker import process_csv_in_background
from ML.predict import get_predictor
from ML.claim_anomaly import detect_claim_anomalies



# ══════════════════════════════════════════════════════════════════════════════
# 1. PARAMETERIZED PROBABILITY & RISK TIER VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("prob,expected_tier", [
    (0.00, "LOW"),
    (0.15, "LOW"),
    (0.219, "LOW"),
    (0.22, "MEDIUM"),
    (0.45, "MEDIUM"),
    (0.699, "MEDIUM"),
    (0.70, "HIGH"),
    (0.95, "HIGH"),
    (1.00, "HIGH"),
])
def test_universal_probability_and_risk_validation_valid(prob, expected_tier):
    score_f, tier = UniversalResponseValidator.validate_probability_and_tier(
        score=prob,
        tier=expected_tier,
        thresholds={"low_medium_cutoff": 0.22, "medium_high_cutoff": 0.70},
    )
    assert score_f == prob
    assert tier == expected_tier


@pytest.mark.parametrize("invalid_prob,tier", [
    (-0.01, "LOW"),
    (1.01, "HIGH"),
    (float("nan"), "LOW"),
    (float("inf"), "HIGH"),
    (float("-inf"), "LOW"),
    (None, "LOW"),
    ("not_a_number", "LOW"),
])
def test_universal_probability_validation_rejects_invalid(invalid_prob, tier):
    with pytest.raises(AnalysisValidationError):
        UniversalResponseValidator.validate_probability_and_tier(
            score=invalid_prob,
            tier=tier,
            thresholds={"low_medium_cutoff": 0.22, "medium_high_cutoff": 0.70},
        )


def test_universal_risk_tier_mismatch_rejection():
    # 0.85 should be HIGH; passed as LOW should raise error
    with pytest.raises(AnalysisValidationError) as exc_info:
        UniversalResponseValidator.validate_probability_and_tier(
            score=0.85,
            tier="LOW",
            thresholds={"low_medium_cutoff": 0.22, "medium_high_cutoff": 0.70},
        )
    assert "Risk tier mismatch" in str(exc_info.value)


# ══════════════════════════════════════════════════════════════════════════════
# 2. PARAMETERIZED SINGLE-CLAIM INFERENCE WITH VARIED INPUT CONDITIONS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("provider_id,claim_type,reimb,los,expected_status", [
    ("PRV51001", "Outpatient", 250.0, 0, 200),          # Normal low-reimb outpatient
    ("PRV51003", "Inpatient", 42000.0, 12, 200),         # High-reimb inpatient
    (f"PRV_NEW_{uuid.uuid4().hex[:6]}", "Outpatient", 500.0, 0, 200),  # Future unknown provider
    ("PRV51001", "Outpatient", 0.0, 0, 200),            # Zero reimbursement
])
def test_single_claim_analysis_varied_inputs(client, auth_headers, provider_id, claim_type, reimb, los, expected_status):
    unique_cid = f"CLM_PAR_{uuid.uuid4().hex[:8]}"
    start_dt = "2009-03-01"
    end_dt = "2009-03-05" if los > 0 else "2009-03-01"
    adm_dt = start_dt if claim_type == "Inpatient" else None
    dis_dt = end_dt if claim_type == "Inpatient" else None

    payload = {
        "ClaimID": unique_cid,
        "ProviderID": provider_id,
        "BeneID": f"BENE_{uuid.uuid4().hex[:6]}",
        "ClaimType": claim_type,
        "InscClaimAmtReimbursed": reimb,
        "DeductibleAmtPaid": 50.0 if reimb > 0 else 0.0,
        "ClaimStartDt": start_dt,
        "ClaimEndDt": end_dt,
        "AdmissionDt": adm_dt,
        "DischargeDt": dis_dt,
        "ClmDiagnosisCode_1": "4019",
        "AttendingPhysician": "PHY31201",
    }

    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == expected_status
    data = res.json()

    assert data["claim_id"] == unique_cid
    assert data["provider_id"] == provider_id
    assert 0.0 <= data["provider_risk_score"] <= 1.0
    assert data["provider_risk_tier"] in ("LOW", "MEDIUM", "HIGH")
    assert not math.isnan(data["provider_risk_score"])
    assert data["model_version"] in ("v1", "v2")
    assert "disclaimer" in data and len(data["disclaimer"]) > 10
    assert "investigation_recommendation" in data and len(data["investigation_recommendation"]) > 10


def test_single_claim_analysis_negative_reimbursement_rejected(client, auth_headers):
    payload = {
        "ClaimID": f"CLM_NEG_{uuid.uuid4().hex[:8]}",
        "ProviderID": "PRV51001",
        "InscClaimAmtReimbursed": -500.0,  # Negative amount rejected by schema
        "ClaimStartDt": "2009-03-01",
    }
    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 422


def test_single_claim_analysis_missing_optional_fields(client, auth_headers):
    """Analysis must succeed even if BeneID, physician, and diagnosis are omitted."""
    unique_cid = f"CLM_MIN_{uuid.uuid4().hex[:8]}"
    payload = {
        "ClaimID": unique_cid,
        "ProviderID": "PRV51001",
        "InscClaimAmtReimbursed": 300.0,
        "ClaimStartDt": "2009-04-01",
        # BeneID omitted, physicians omitted, diagnosis omitted
    }
    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["claim_id"] == unique_cid
    assert data["provider_risk_score"] >= 0.0
    # Anomaly detector flags missing physician info
    assert "MISSING_PHYSICIAN_INFO" in data["anomaly_flags"]


# ══════════════════════════════════════════════════════════════════════════════
# 3. UNIVERSAL SHAP EXPLANATION & FEATURE INTEGRITY
# ══════════════════════════════════════════════════════════════════════════════

def test_universal_shap_explanation_consistency(client, auth_headers):
    unique_cid = f"CLM_SHAP_{uuid.uuid4().hex[:8]}"
    payload = {
        "ClaimID": unique_cid,
        "ProviderID": "PRV51003",
        "BeneID": "BENE11001",
        "ClaimType": "Inpatient",
        "InscClaimAmtReimbursed": 38000.0,
        "ClaimStartDt": "2009-05-01",
        "ClaimEndDt": "2009-05-15",
        "AdmissionDt": "2009-05-01",
        "DischargeDt": "2009-05-15",
        "ClmDiagnosisCode_1": "4019",
        "ClmProcedureCode_1": "3893",
        "AttendingPhysician": "PHY31201",
    }
    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()

    expl = data["explanation"]
    assert "text" in expl and len(expl["text"]) > 20
    assert "top_risk_factors" in expl and len(expl["top_risk_factors"]) > 0

    # Verify structured factors
    for f in expl["top_risk_factors"]:
        assert "feature" in f and len(f["feature"]) > 0
        act_val = f.get("actual_value", f.get("feature_value"))
        assert act_val is not None and math.isfinite(float(act_val))
        shap_val = f.get("shap_value")
        assert shap_val is not None and math.isfinite(float(shap_val))
        assert f["direction"] in ("INCREASES_FRAUD_RISK", "DECREASES_FRAUD_RISK")

    # Positive contributors strictly have shap_value > 0
    pos_factors = expl.get("positive_contributors", [])
    for f in pos_factors:
        assert float(f["shap_value"]) > 0.0
        assert f["direction"] == "INCREASES_FRAUD_RISK"

    # Negative contributors strictly have shap_value < 0
    neg_factors = expl.get("negative_contributors", [])
    for f in neg_factors:
        assert float(f["shap_value"]) < 0.0
        assert f["direction"] == "DECREASES_FRAUD_RISK"


# ══════════════════════════════════════════════════════════════════════════════
# 4. UNIVERSAL ANOMALY ANALYSIS & DISCLAIMER TRUTHFULNESS
# ══════════════════════════════════════════════════════════════════════════════

def test_clean_claim_no_false_anomaly_claims(client, auth_headers):
    """
    When a clean standard outpatient claim with valid physician and diagnosis is analyzed,
    no anomaly is detected, and disclaimer does NOT claim unusual indicators were found.
    """
    unique_cid = f"CLM_CLN_{uuid.uuid4().hex[:8]}"
    payload = {
        "ClaimID": unique_cid,
        "ProviderID": "PRV51001",
        "BeneID": f"BENE_CLN_{uuid.uuid4().hex[:4]}",
        "ClaimType": "Outpatient",
        "InscClaimAmtReimbursed": 350.0,  # Below P95 benchmark
        "ClaimStartDt": "2009-07-01",
        "ClaimEndDt": "2009-07-01",
        "ClmDiagnosisCode_1": "4019",
        "AttendingPhysician": "PHY31201",
    }
    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()

    assert data["has_anomalies"] is False
    assert len(data["anomaly_flags"]) == 0

    findings = data["claim_anomaly_findings"]
    assert len(findings) == 1
    f = findings[0]
    assert f["is_anomalous"] is False
    assert f["anomaly_count"] == 0
    # Must NOT claim unusual indicators were found
    assert "unusual indicators that may require investigation" not in f["disclaimer"].lower()
    assert "no anomalous indicators were detected" in f["disclaimer"].lower()


def test_multiple_anomalies_detected_honestly(client, auth_headers):
    """Trigger multiple distinct anomalies: high reimbursement (>P95) and post-death claim."""
    unique_cid = f"CLM_ANOM_{uuid.uuid4().hex[:8]}"
    payload = {
        "ClaimID": unique_cid,
        "ProviderID": "PRV51003",
        "BeneID": "BENE11001",
        "ClaimType": "Inpatient",
        "InscClaimAmtReimbursed": 75000.0,  # Strictly exceeds $57,000 P95
        "ClaimStartDt": "2009-08-10",
        "ClaimEndDt": "2009-08-15",
        "AdmissionDt": "2009-08-10",
        "DischargeDt": "2009-08-15",
        "DOD": "2009-01-01",  # Beneficiary date of death prior to claim
        "ClmDiagnosisCode_1": "4019",
        "AttendingPhysician": "PHY31201",
    }
    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()

    assert data["has_anomalies"] is True
    assert "HIGH_REIMBURSEMENT" in data["anomaly_flags"]
    assert "POST_DEATH_CLAIM" in data["anomaly_flags"]

    findings = data["claim_anomaly_findings"]
    f = findings[0]
    assert f["is_anomalous"] is True
    assert f["anomaly_count"] >= 2
    assert "unusual indicators that may require investigation" in f["disclaimer"].lower()


# ══════════════════════════════════════════════════════════════════════════════
# 5. UNIVERSAL TEMPORAL SEQUENCE DYNAMICS (Bi-LSTM)
# ══════════════════════════════════════════════════════════════════════════════

def test_temporal_sequence_explanation_validation():
    predictor = get_predictor()
    explainer = predictor.explainer

    # 1. Sequence with 6 padded months and 6 valid observed months (including a zero-claim month)
    seq = np.zeros((1, 12, 6), dtype=np.float32)
    mask = np.zeros((1, 12), dtype=np.float32)

    # Months 6..11 are within active window
    mask[0, 6:] = 1.0
    for m in range(7, 12):
        seq[0, m, 0] = 5.0      # monthly_claim_count
        seq[0, m, 1] = 8.5      # log_total_reimbursement
        seq[0, m, 2] = 4.0      # unique_bene
        seq[0, m, 4] = 1200.0   # avg_reimbursement
    # Month 6 has mask=1.0 but 0 claims -> ZERO_CLAIMS_IN_WINDOW

    expl = explainer.explain_temporal_sequence(predictor.lstm_detector, seq, mask)
    impacts = expl["monthly_temporal_impacts"]
    assert len(impacts) == 12

    # Verify padded months (0..5)
    for idx in range(6):
        m = impacts[idx]
        assert m["is_valid_observation"] is False
        assert m["observation_type"] == "PADDING"
        assert m["impact"] == 0.0

    # Verify zero-claim month in active window (month 7, 0-indexed 6)
    m6 = impacts[6]
    assert m6["is_valid_observation"] is True
    assert m6["observation_type"] == "ZERO_CLAIMS_IN_WINDOW"

    # Verify active observation months (months 7..11)
    for idx in range(7, 12):
        m = impacts[idx]
        assert m["is_valid_observation"] is True
        assert m["observation_type"] == "ACTIVE_CLAIMS"

    assert "trajectory_summary" in expl and len(expl["trajectory_summary"]) > 0


def test_bilstm_explicit_mask_vs_heuristic_zero_masking():
    """
    Verify that an explicit validity mask correctly preserves genuine zero-claim
    months in the active observation window, whereas a naive heuristic
    `mask = (seq.abs().sum(dim=-1) > 0)` would incorrectly classify them as padding.
    """
    predictor = get_predictor()
    detector = predictor.lstm_detector

    seq = np.zeros((1, 12, 6), dtype=np.float32)
    mask_explicit = np.zeros((1, 12), dtype=np.float32)

    # Active observation window: months 6..11 (6 months)
    mask_explicit[0, 6:] = 1.0

    # Month 6 is an active window month with ZERO claims (all 6 features remain 0.0)
    # Months 7..11 have active billing patterns
    for m in range(7, 12):
        seq[0, m, 0] = 8.0     # monthly_claim_count
        seq[0, m, 1] = 9.2     # log_total_reimbursement
        seq[0, m, 2] = 6.0     # unique_beneficiaries
        seq[0, m, 4] = 1500.0  # avg_reimbursement

    # 1. Demonstrate flaw in naive heuristic:
    heuristic_mask = (np.abs(seq).sum(axis=-1) > 0).astype(np.float32)
    # Month 6 has sum == 0, so heuristic falsely flags it as padding (0.0)
    assert heuristic_mask[0, 6] == 0.0, "Heuristic should have falsely marked zero-claim month as 0.0"
    assert mask_explicit[0, 6] == 1.0, "Explicit mask must preserve zero-claim month as 1.0 (valid observation)"

    # 2. Verify RobustScaler normalization preserves leading padding zeros (0..5)
    normalized = detector._normalize_sequences(seq, mask_explicit, fit_scaler=False)
    for p in range(6):
        assert np.all(normalized[0, p, :] == 0.0), f"Padded month {p} must remain strictly zero after normalization"
    
    # Valid month 6 was in active window, so after RobustScaler centering it is non-zero
    assert not np.all(normalized[0, 6, :] == 0.0), "Valid month 6 should be transformed by RobustScaler"

    # 3. Verify Bi-LSTM forward inference succeeds cleanly with explicit mask
    prob_explicit = detector.predict_proba(seq, mask_explicit)
    prob_heuristic = detector.predict_proba(seq, heuristic_mask)

    assert 0.0 <= prob_explicit[0] <= 1.0
    assert 0.0 <= prob_heuristic[0] <= 1.0
    # Different masks result in distinct pooled temporal representations and probabilities
    assert prob_explicit[0] != prob_heuristic[0]


# ══════════════════════════════════════════════════════════════════════════════
# 6. BATCH CSV PROCESSING, DUPLICATE RECONCILIATION & PARITY
# ══════════════════════════════════════════════════════════════════════════════

def test_csv_batch_mixed_and_reconciliation(db_session):
    cid_valid_1 = f"CLM_CSV_VAL1_{uuid.uuid4().hex[:6]}"
    cid_valid_2 = f"CLM_CSV_VAL2_{uuid.uuid4().hex[:6]}"
    cid_dup = cid_valid_1  # In-batch duplicate of valid_1

    csv_content = (
        "ClaimID,ProviderID,BeneID,InscClaimAmtReimbursed,ClaimStartDt,ClaimType\n"
        f"{cid_valid_1},PRV51001,BENE11001,450.00,2009-03-01,Outpatient\n"
        f",PRV51001,BENE11002,300.00,2009-03-02,Outpatient\n"                  # Missing ClaimID -> REJECT
        f"{cid_valid_2},PRV51003,BENE11003,35000.00,2009-03-05,Inpatient\n"
        f"CLM_BAD_REIMB,PRV51001,BENE11004,INVALID_AMT,2009-03-06,Outpatient\n" # Bad Reimb -> REJECT
        f"{cid_dup},PRV51001,BENE11001,450.00,2009-03-01,Outpatient\n"          # Duplicate -> REJECT
    ).encode("utf-8")

    from backend.app.db.models import CSVUpload
    upload = CSVUpload(
        filename="test_mixed.csv",
        total_rows=5,
        successful_rows=0,
        rejected_rows=0,
        status="PENDING",
    )
    db_session.add(upload)
    db_session.commit()
    db_session.refresh(upload)

    process_csv_in_background(upload.id, csv_content, db=db_session)
    db_session.refresh(upload)

    assert upload.status == "COMPLETED"
    assert upload.total_rows == 5
    assert upload.successful_rows == 2
    assert upload.rejected_rows == 3
    # Strict reconciliation invariant check
    assert upload.successful_rows + upload.rejected_rows == upload.total_rows


def test_single_versus_csv_prediction_parity(client, auth_headers, db_session):
    """
    Assert that single-claim and CSV analysis of the same input claim
    produce identical model risk scores and tier assignments.
    """
    cid_parity = f"CLM_PARITY_{uuid.uuid4().hex[:6]}"
    pid_parity = "PRV51001"
    reimb = 1250.0
    start_dt = "2009-04-10"

    # 1. Run via Single-Claim API
    payload = {
        "ClaimID": cid_parity,
        "ProviderID": pid_parity,
        "BeneID": "BENE11001",
        "ClaimType": "Outpatient",
        "InscClaimAmtReimbursed": reimb,
        "ClaimStartDt": start_dt,
        "ClmDiagnosisCode_1": "4019",
        "AttendingPhysician": "PHY31201",
    }
    res_single = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res_single.status_code == 200
    single_data = res_single.json()

    # 2. Run identical claim through prediction pipeline
    df_claim = pd.DataFrame([{
        "ClaimID": cid_parity,
        "ProviderID": pid_parity,
        "BeneID": "BENE11001",
        "ClaimType": "Outpatient",
        "InscClaimAmtReimbursed": reimb,
        "ClaimStartDt": start_dt,
        "ClmDiagnosisCode_1": "4019",
        "AttendingPhysician": "PHY31201",
    }])
    predictor = get_predictor()
    pipeline_res = predictor.predict_claims(df_claim)

    csv_p_res = pipeline_res["provider_results"][pid_parity]
    csv_score = round(float(csv_p_res["calibrated_fraud_probability"]), 4)
    csv_tier = csv_p_res["risk_tier"]

    # 3. Assert exact equality
    assert single_data["provider_risk_score"] == csv_score
    assert single_data["provider_risk_tier"] == csv_tier


# ══════════════════════════════════════════════════════════════════════════════
# 7. FUTURE & DYNAMIC POSTGRESQL DATA COMPATIBILITY
# ══════════════════════════════════════════════════════════════════════════════

def test_future_provider_not_in_demo_dataset(client, auth_headers, db_session):
    """
    Ensure the system functions flawlessly with newly generated future providers
    and beneficiaries never observed in the demo dataset.
    """
    future_pid = f"PRV_FUTURE_{uuid.uuid4().hex[:6]}"
    future_bid = f"BENE_FUTURE_{uuid.uuid4().hex[:6]}"
    future_cid = f"CLM_FUTURE_{uuid.uuid4().hex[:8]}"

    payload = {
        "ClaimID": future_cid,
        "ProviderID": future_pid,
        "BeneID": future_bid,
        "ClaimType": "Outpatient",
        "InscClaimAmtReimbursed": 780.0,
        "ClaimStartDt": "2009-09-01",
        "ClmDiagnosisCode_1": "25000",
        "AttendingPhysician": "PHY_FUTURE_01",
    }

    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()

    assert data["claim_id"] == future_cid
    assert data["provider_id"] == future_pid
    assert 0.0 <= data["provider_risk_score"] <= 1.0

    # Verify records persisted in PostgreSQL
    p_rec = db_session.query(Provider).filter(Provider.provider_id == future_pid).first()
    assert p_rec is not None

    b_rec = db_session.query(Beneficiary).filter(Beneficiary.beneficiary_id == future_bid).first()
    assert b_rec is not None

    c_rec = db_session.query(Claim).filter(Claim.claim_id == future_cid).first()
    assert c_rec is not None
    assert c_rec.provider_id == future_pid
    assert c_rec.beneficiary_id == future_bid

    score_rec = db_session.query(ProviderRiskScore).filter(ProviderRiskScore.provider_id == future_pid).first()
    assert score_rec is not None
    assert score_rec.risk_score == data["provider_risk_score"]
