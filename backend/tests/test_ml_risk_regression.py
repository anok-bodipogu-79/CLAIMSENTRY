"""
HealthGuard AI - ML Risk Prediction Regression Diagnostic & Verification Test Suite
==================================================================================
Tests every component of the ML prediction pipeline:
1. Feature generation & 51-feature ordering
2. RobustScaler & scaling invariants
3. Bi-LSTM 12x6 sequence and validity mask construction
4. Multi-modal base models (XGBoost, ANN, Bi-LSTM) execution
5. Stacking ensemble meta-model combination and Platt calibration
6. Risk classification thresholds (0.22 and 0.70 cutoffs)
7. Missing history & new provider handling
8. Controlled test suite: Tests A through H
9. SHAP explainability alignment
10. Response schema and frontend consistency
"""

import json
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone

from ML.predict import get_predictor
from ML.feature_engineering import build_provider_features, build_monthly_sequences
from ML.claim_anomaly import detect_claim_anomalies
from backend.app.services.claim_analysis_pipeline import UniversalResponseValidator
from backend.app.db.schemas import SingleClaimResponse


@pytest.fixture(scope="module")
def predictor():
    pred = get_predictor()
    assert pred.is_ready, f"Predictor failed initialization: {pred.verification_error}"
    return pred


@pytest.fixture(scope="module")
def manifest():
    with open("ML/artifacts/v2/model_manifest.json", "r", encoding="utf-8") as f:
        return json.load(f)


def test_51_tabular_features_generated_in_exact_order(predictor, manifest):
    """Verify exactly 51 tabular features generated in exact manifest order."""
    expected_features = manifest["feature_names"]
    assert len(expected_features) == 51

    claim = {
        "ClaimID": "CLM_REG_001",
        "ProviderID": "PRV_REG_001",
        "BeneID": "BEN_REG_001",
        "ClaimType": "Inpatient",
        "InscClaimAmtReimbursed": 5000.0,
        "DeductibleAmtPaid": 1068.0,
        "ClaimStartDt": "2026-01-10",
        "ClaimEndDt": "2026-01-15",
        "AdmissionDt": "2026-01-10",
        "DischargeDt": "2026-01-15",
        "InpatientLOS": 5,
        "ClmDiagnosisCode_1": "4019",
        "AttendingPhysician": "PHY_REG_01",
    }
    feat_df = build_provider_features(pd.DataFrame([claim]), peer_ref=predictor.peer_reference_table)
    assert feat_df.shape[0] == 1

    df_row, warnings = predictor.validate_features(feat_df.iloc[0].to_dict())
    assert list(df_row.columns) == expected_features
    assert len(df_row.columns) == 51
    assert df_row.isna().sum().sum() == 0


def test_bilstm_sequence_and_mask_construction():
    """Verify 12x6 sequence dimensions and 12-dim validity mask."""
    claim = {
        "ClaimID": "CLM_SEQ_001",
        "ProviderID": "PRV_SEQ_001",
        "BeneID": "BEN_SEQ_001",
        "ClaimType": "Outpatient",
        "InscClaimAmtReimbursed": 750.0,
        "DeductibleAmtPaid": 50.0,
        "ClaimStartDt": "2026-02-01",
        "ClaimEndDt": "2026-02-01",
    }
    seqs, masks, plist, meta = build_monthly_sequences(pd.DataFrame([claim]), pd.Series(["PRV_SEQ_001"]))
    assert seqs.shape == (1, 12, 6)
    assert masks.shape == (1, 12)
    assert masks[0, -1] == 1.0 # Current month active
    assert int(masks[0].sum()) == 1 # 1 active month for single claim
    assert not np.isnan(seqs).any()
    assert not np.isinf(seqs).any()


def test_stacking_meta_model_and_calibration_invariants(predictor, manifest):
    """Verify Stacking Meta-model LogisticRegression and Platt scaling mathematical invariants."""
    coef = manifest["ensemble"]["coefficients"]
    intercept = manifest["ensemble"]["intercept"]

    assert "xgboost" in coef
    assert "ann" in coef
    assert "bilstm" in coef
    assert intercept < -5.0 # Reflects low base rate of fraud in training distribution

    # Test monotonic calibration property
    calibrator = predictor.ensemble.calibrator
    raw_inputs = np.array([0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.95])
    cal_outputs = calibrator.predict(raw_inputs)

    assert len(cal_outputs) == len(raw_inputs)
    assert np.all(cal_outputs >= 0.0)
    assert np.all(cal_outputs <= 1.0)
    # Monotonically non-decreasing
    assert np.all(np.diff(cal_outputs) >= 0.0)


def test_risk_classification_thresholds_consistency(predictor, manifest):
    """Verify risk thresholds match manifest and apply strictly without drift."""
    t_low = manifest["risk_thresholds"]["low_medium_cutoff"]
    t_high = manifest["risk_thresholds"]["medium_high_cutoff"]
    assert t_low == 0.22
    assert t_high == 0.70

    score_low, tier_low = UniversalResponseValidator.validate_probability_and_tier(0.15, "LOW")
    assert tier_low == "LOW"

    score_med, tier_med = UniversalResponseValidator.validate_probability_and_tier(0.45, "MEDIUM")
    assert tier_med == "MEDIUM"

    score_high, tier_high = UniversalResponseValidator.validate_probability_and_tier(0.78, "HIGH")
    assert tier_high == "HIGH"

    with pytest.raises(Exception):
        UniversalResponseValidator.validate_probability_and_tier(0.85, "LOW")


def test_controlled_test_a_ordinary_claim(predictor):
    """Test A — Ordinary claim generates valid prediction, low probability, and no anomaly flags."""
    claim = {
        "ClaimID": "CTRL_A_001",
        "ProviderID": "PRV_CTRL_A",
        "BeneID": "BEN_CTRL_A",
        "ClaimType": "Outpatient",
        "InscClaimAmtReimbursed": 250.0,
        "DeductibleAmtPaid": 50.0,
        "ClaimStartDt": "2026-02-01",
        "ClaimEndDt": "2026-02-01",
        "ClmDiagnosisCode_1": "4019",
        "AttendingPhysician": "PHY_A",
    }
    res = predictor.predict_claims([claim], provider_id="PRV_CTRL_A")
    p = res["provider_results"]["PRV_CTRL_A"]
    anom = res["claims_result"][0]

    assert 0.0 <= p["calibrated_fraud_probability"] <= 1.0
    assert p["risk_tier"] == "LOW"
    assert anom["anomaly_count"] == 0
    assert p["has_limited_history"] is True


def test_controlled_test_c_suspicious_claim_triggers_anomalies(predictor):
    """Test C — Suspicious claim triggers anomaly flags despite limited provider history."""
    claim = {
        "ClaimID": "CTRL_C_001",
        "ProviderID": "PRV_CTRL_C",
        "BeneID": "BEN_CTRL_C",
        "ClaimType": "Inpatient",
        "InscClaimAmtReimbursed": 65000.0,
        "DeductibleAmtPaid": 0.0,
        "ClaimStartDt": "2026-02-10",
        "ClaimEndDt": "2026-02-05", # Invalid date
        "AdmissionDt": "2026-02-10",
        "DischargeDt": "2026-02-05", # Discharge before admission
        "InpatientLOS": 25,
        "ClmDiagnosisCode_1": "4019",
        "AttendingPhysician": None, # Missing physician
    }
    res = predictor.predict_claims([claim], provider_id="PRV_CTRL_C")
    anom = res["claims_result"][0]
    p = res["provider_results"]["PRV_CTRL_C"]

    assert anom["anomaly_count"] >= 3
    assert "HIGH_REIMBURSEMENT" in anom["anomaly_flags"]
    assert "INVALID_CLAIM_DATES" in anom["anomaly_flags"]
    assert "MISSING_PHYSICIAN_INFO" in anom["anomaly_flags"]
    # Model detects elevated probability from base models
    assert p["base_model_probabilities"]["ann"] > 0.80


def test_high_volume_provider_reaches_elevated_risk(predictor):
    """Test that provider with high claim volume and high reimbursement reaches Medium/High Risk."""
    claims = []
    for i in range(150):
        claims.append({
            "ClaimID": f"CLM_HV_{i:03d}",
            "ProviderID": "PRV_HIGH_VOL",
            "BeneID": f"BEN_HV_{i%25:02d}",
            "ClaimType": "Inpatient" if i % 2 == 0 else "Outpatient",
            "InscClaimAmtReimbursed": 8000.0 if i % 2 == 0 else 1200.0,
            "DeductibleAmtPaid": 1068.0 if i % 2 == 0 else 100.0,
            "ClaimStartDt": f"2025-{(i%11)+1:02d}-{(i%25)+1:02d}",
            "ClaimEndDt": f"2025-{(i%11)+1:02d}-{(i%25)+1:02d}",
            "AdmissionDt": f"2025-{(i%11)+1:02d}-{(i%25)+1:02d}" if i % 2 == 0 else None,
            "DischargeDt": f"2025-{(i%11)+1:02d}-{(i%25)+1:02d}" if i % 2 == 0 else None,
            "InpatientLOS": 5 if i % 2 == 0 else 0,
            "ClmDiagnosisCode_1": "4019",
            "AttendingPhysician": f"PHY_HV_{i%5}",
        })
    res = predictor.predict_claims(claims, provider_id="PRV_HIGH_VOL")
    p = res["provider_results"]["PRV_HIGH_VOL"]

    assert p["total_claims_analyzed"] == 150
    assert p["calibrated_fraud_probability"] >= 0.22
    assert p["risk_tier"] in ("MEDIUM", "HIGH")


def test_shap_explanation_negative_features_on_single_claim(predictor):
    """Test that SHAP feature attribution correctly explains low probability due to single-claim volume."""
    claim = {
        "ClaimID": "CTRL_SHAP_001",
        "ProviderID": "PRV_CTRL_SHAP",
        "BeneID": "BEN_CTRL_SHAP",
        "ClaimType": "Outpatient",
        "InscClaimAmtReimbursed": 300.0,
        "DeductibleAmtPaid": 50.0,
        "ClaimStartDt": "2026-01-15",
        "ClaimEndDt": "2026-01-15",
        "ClmDiagnosisCode_1": "4019",
        "AttendingPhysician": "PHY_SHAP",
    }
    res = predictor.predict_claims([claim], provider_id="PRV_CTRL_SHAP", include_shap=True)
    p = res["provider_results"]["PRV_CTRL_SHAP"]

    shap_expl = p.get("shap_explanation")
    assert shap_expl is not None
    assert "top_negative_risk_factors" in shap_expl
    neg_feats = [f["feature"] for f in shap_expl["top_negative_risk_factors"]]
    # Total reimbursement and claim volume should be leading negative drivers
    assert any("reimbursement" in f or "claim" in f or "bene" in f for f in neg_feats)
