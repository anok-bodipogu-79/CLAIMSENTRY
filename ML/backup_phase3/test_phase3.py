"""
HealthGuard AI - Phase 3 Automated Test Suite
=============================================
Verification tests for Phase 3 objectives:
  1. Evaluation, ablation comparisons, and robustness checks
  2. XGBoost SHAP feature attribution (finite values, actual model outputs, top 5 pos/neg)
  3. Bi-LSTM temporal sequence explainability (actual sequences, occlusion differences, valid months)
  4. Claim-level rule-based anomaly detection (unit tests for all 8 rules on synthetic fixtures)
  5. Shared predict_claims() pipeline (single claim, multiple claims, multiple providers, duplicate ClaimIDs)
  6. Single-claim versus CSV prediction consistency
  7. Row rejection reasons and count reconciliation (analysed + rejected == total_uploaded)
  8. Model preservation (v1 and v2 artifacts unchanged, thresholds remain 0.22/0.70, 5 main folders present)
"""

import json
import hashlib
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROC = ROOT / "data" / "processed"
ART_V1 = ROOT / "ML" / "artifacts" / "v1"
ART_V2 = ROOT / "ML" / "artifacts" / "v2"
REP = ROOT / "reports" / "evaluation"

from ML.predict import HealthGuardPredictor, get_predictor, predict_claims
from ML.claim_anomaly import detect_claim_anomalies, CLAIM_DISCLAIMER
from ML.explainability import FraudExplainer


@pytest.fixture(scope="module")
def predictor():
    return get_predictor()


@pytest.fixture(scope="module")
def sample_test_data():
    df_test = pd.read_csv(PROC / "test_provider_features.csv")
    seqs = np.load(PROC / "test_provider_sequences.npy")
    masks = np.load(PROC / "test_provider_masks.npy")
    return df_test, seqs, masks


# ══════════════════════════════════════════════════════════════════════════════
# 1. EVALUATION, ABLATION & ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════════════

def test_ablation_study_file_exists_and_valid():
    """Verify ablation study results exist and contain all 5 required configurations."""
    ablation_csv = REP / "ablation_study.csv"
    assert ablation_csv.exists(), "ablation_study.csv must exist in reports/evaluation/"

    df = pd.read_csv(ablation_csv)
    assert len(df) == 5, f"Expected 5 ablation configurations, got {len(df)}"

    expected_configs = [
        "XGBoost Only",
        "ANN Only",
        "Bi-LSTM Only",
        "Ensemble Without Calibration (Raw Stacking)",
        "Calibrated Stacking Ensemble (Full Framework)",
    ]
    for cfg in expected_configs:
        assert cfg in df["Configuration"].values, f"Missing configuration: {cfg}"

    # Verify metrics are within valid probability and ranking bounds
    for col in ["PR_AUC", "ROC_AUC", "F1_Score", "Precision", "Recall", "Brier_Score"]:
        assert (df[col] >= 0.0).all() and (df[col] <= 1.0).all()


def test_robustness_results_file_exists_and_all_pass():
    """Verify all 10 robustness scenarios executed and passed."""
    rob_json = REP / "robustness_results.json"
    assert rob_json.exists(), "robustness_results.json must exist"

    with open(rob_json, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    assert len(scenarios) == 10, f"Expected 10 robustness scenarios, got {len(scenarios)}"
    for s in scenarios:
        assert s["status"] == "PASS", f"Scenario {s['scenario_id']} failed"


def test_risk_tier_summary_monotonic():
    """Verify risk tier summary exists and monotonic fraud progression holds."""
    tier_json = REP / "risk_tier_summary.json"
    assert tier_json.exists(), "risk_tier_summary.json must exist"

    with open(tier_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["monotonic_risk_progression"] is True
    assert data["LOW"]["fraud_rate"] <= data["MEDIUM"]["fraud_rate"] <= data["HIGH"]["fraud_rate"]


# ══════════════════════════════════════════════════════════════════════════════
# 2. XGBOOST SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

def test_shap_explanations_are_finite_and_separated(predictor, sample_test_data):
    """Verify SHAP produces finite values with top 5 positive and negative risk factors."""
    df_test, _, _ = sample_test_data
    feat_dict = df_test.iloc[0].to_dict()

    expl = predictor.explainer.explain_provider(feat_dict, top_n=5)
    assert "top_risk_factors" in expl
    assert "top_positive_risk_factors" in expl
    assert "top_negative_risk_factors" in expl
    assert "scope_disclosure" in expl

    assert len(expl["top_risk_factors"]) <= 5
    assert len(expl["top_positive_risk_factors"]) <= 5
    assert len(expl["top_negative_risk_factors"]) <= 5

    # Check finite values and directions
    for factor in expl["top_positive_risk_factors"]:
        assert np.isfinite(factor["shap_value"])
        assert factor["shap_value"] > 0
        assert factor["direction"] == "INCREASES_FRAUD_RISK"
        assert "feature_value" in factor
        assert "peer_percentile" in factor

    for factor in expl["top_negative_risk_factors"]:
        assert np.isfinite(factor["shap_value"])
        assert factor["shap_value"] < 0
        assert factor["direction"] == "DECREASES_FRAUD_RISK"


def test_shap_rejects_nan_features(predictor, sample_test_data):
    """Verify NaN in provider features is strictly rejected by explainer."""
    df_test, _, _ = sample_test_data
    feat_dict = df_test.iloc[0].to_dict()
    feat_dict["total_reimbursement"] = np.nan

    with pytest.raises(ValueError) as exc:
        predictor.explainer.explain_provider(feat_dict)
    assert "NaN or Infinity" in str(exc.value)


# ══════════════════════════════════════════════════════════════════════════════
# 3. BI-LSTM TEMPORAL EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

def test_temporal_explanation_occlusion(predictor, sample_test_data):
    """Verify temporal occlusion analysis on genuine monthly sequence."""
    _, seqs, masks = sample_test_data
    seq = seqs[0:1]
    mask = masks[0:1]

    temp_expl = predictor.explainer.explain_temporal_sequence(
        predictor.lstm_detector, seq, mask
    )

    assert "baseline_lstm_probability" in temp_expl
    assert "monthly_temporal_impacts" in temp_expl
    assert "most_influential_months" in temp_expl
    assert "trajectory_summary" in temp_expl

    assert len(temp_expl["monthly_temporal_impacts"]) == 12
    for m in temp_expl["monthly_temporal_impacts"]:
        assert np.isfinite(m["impact"])
        if m["is_valid_observation"]:
            assert "observed_features" in m
            assert len(m["observed_features"]) == 6


# ══════════════════════════════════════════════════════════════════════════════
# 4. CLAIM-LEVEL ANOMALY DETECTION (UNIT TESTS FOR ALL 8 RULES)
# ══════════════════════════════════════════════════════════════════════════════

def test_anomaly_rule_high_reimbursement():
    claim = {
        "ClaimID": "CLM_ANOM_REIMB",
        "ProviderID": "PRV_ANOM",
        "InscClaimAmtReimbursed": 35000.0,
        "ClaimStartDt": "2009-01-01",
    }
    res = detect_claim_anomalies([claim], peer_reimbursement_p95=10000.0, peer_reimbursement_p99=30000.0)
    assert len(res) == 1
    assert "HIGH_REIMBURSEMENT" in res[0]["anomaly_flags"]


def test_anomaly_rule_repeat_beneficiary():
    claims = [
        {"ClaimID": f"CLM_REP_{i}", "ProviderID": "PRV_ANOM", "BeneID": "BENE_SAME", "InscClaimAmtReimbursed": 100, "ClaimStartDt": "2009-01-01"}
        for i in range(5)
    ]
    res = detect_claim_anomalies(claims)
    assert any("REPEAT_BENEFICIARY" in r["anomaly_flags"] for r in res)


def test_anomaly_rule_post_death_claim():
    claim = {
        "ClaimID": "CLM_DEATH",
        "ProviderID": "PRV_ANOM",
        "BeneID": "BENE_DEAD",
        "InscClaimAmtReimbursed": 100,
        "ClaimStartDt": "2009-06-01",
        "DOD": "2009-05-01",
    }
    res = detect_claim_anomalies([claim])
    assert "POST_DEATH_CLAIM" in res[0]["anomaly_flags"]


def test_anomaly_rule_same_day_activity():
    claims = [
        {"ClaimID": f"CLM_SD_{i}", "ProviderID": "PRV_SPIKE", "InscClaimAmtReimbursed": 100, "ClaimStartDt": "2009-04-10"}
        for i in range(12)
    ]
    res = detect_claim_anomalies(claims)
    assert any("HIGH_SAME_DAY_ACTIVITY" in r["anomaly_flags"] for r in res)


def test_anomaly_rule_invalid_claim_dates():
    claim = {
        "ClaimID": "CLM_DATE_INV",
        "ProviderID": "PRV_ANOM",
        "InscClaimAmtReimbursed": 100,
        "ClaimStartDt": "2009-07-10",
        "ClaimEndDt": "2009-07-01",  # End before Start
    }
    res = detect_claim_anomalies([claim])
    assert "INVALID_CLAIM_DATES" in res[0]["anomaly_flags"]


def test_anomaly_rule_missing_physician_info():
    claim = {
        "ClaimID": "CLM_NO_PHYS",
        "ProviderID": "PRV_ANOM",
        "InscClaimAmtReimbursed": 100,
        "ClaimStartDt": "2009-01-01",
        "AttendingPhysician": None,
        "OperatingPhysician": None,
        "OtherPhysician": None,
    }
    res = detect_claim_anomalies([claim])
    assert "MISSING_PHYSICIAN_INFO" in res[0]["anomaly_flags"]


def test_anomaly_rule_missing_diagnosis_procedure():
    claim = {
        "ClaimID": "CLM_NO_DIAG",
        "ProviderID": "PRV_ANOM",
        "InscClaimAmtReimbursed": 100,
        "ClaimStartDt": "2009-01-01",
        "ClmDiagnosisCode_1": None,
        "ClmDiagnosisCode_2": None,
        "ClmAdmitDiagnosisCode": None,
    }
    res = detect_claim_anomalies([claim])
    assert "MISSING_DIAGNOSIS_PROCEDURE" in res[0]["anomaly_flags"]


def test_anomaly_rule_unusual_claim_duration():
    claim = {
        "ClaimID": "CLM_LONG_LOS",
        "ProviderID": "PRV_ANOM",
        "InscClaimAmtReimbursed": 1000,
        "ClaimStartDt": "2009-01-01",
        "AdmissionDt": "2009-01-01",
        "DischargeDt": "2009-03-01",  # LOS ~60 days
        "InpatientLOS": 59,
        "ClaimType": "Inpatient",
    }
    res = detect_claim_anomalies([claim], max_inpatient_los_days=30)
    assert "UNUSUAL_CLAIM_DURATION" in res[0]["anomaly_flags"]


# ══════════════════════════════════════════════════════════════════════════════
# 5. SHARED PREDICT_CLAIMS() PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def test_predict_claims_single_claim():
    claim = {
        "ClaimID": "CLM_SHARED_001",
        "ProviderID": "PRV51003",
        "BeneID": "BENE11001",
        "InscClaimAmtReimbursed": 1200.0,
        "ClaimStartDt": "2009-05-01",
        "ClaimEndDt": "2009-05-03",
        "ClaimType": "Outpatient",
        "AttendingPhysician": "PHY12345",
    }
    res = predict_claims(claim)
    assert "provider_result" in res
    assert "claims_result" in res
    p_res = res["provider_result"]
    assert p_res["provider_id"] == "PRV51003"
    assert 0.0 <= p_res["calibrated_fraud_probability"] <= 1.0
    assert p_res["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]
    assert res["disclaimer"] == "This is a risk assessment to guide investigation. It is not proof of fraud."


def test_predict_claims_multiple_providers():
    claims = [
        {"ClaimID": "CLM_P1", "ProviderID": "PRV51003", "BeneID": "B1", "InscClaimAmtReimbursed": 500, "ClaimStartDt": "2009-01-01"},
        {"ClaimID": "CLM_P2", "ProviderID": "PRV55462", "BeneID": "B2", "InscClaimAmtReimbursed": 800, "ClaimStartDt": "2009-01-02"},
    ]
    res = predict_claims(claims)
    assert res["total_uploaded_rows"] == 2
    assert res["analysed_claims"] == 2
    assert res["rejected_rows"] == 0
    assert res["analysed_providers"] == 2
    assert "PRV51003" in res["provider_results"]
    assert "PRV55462" in res["provider_results"]


def test_predict_claims_duplicate_claim_id_handling():
    claims = [
        {"ClaimID": "CLM_DUP", "ProviderID": "PRV51003", "BeneID": "B1", "InscClaimAmtReimbursed": 500, "ClaimStartDt": "2009-01-01"},
        {"ClaimID": "CLM_DUP", "ProviderID": "PRV51003", "BeneID": "B1", "InscClaimAmtReimbursed": 500, "ClaimStartDt": "2009-01-01"},
    ]
    res = predict_claims(claims)
    assert res["total_uploaded_rows"] == 2
    assert res["analysed_claims"] == 2
    assert res["analysed_providers"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# 6. CSV WORKFLOW, RECONCILIATION & CONSISTENCY
# ══════════════════════════════════════════════════════════════════════════════

def test_csv_reconciliation_with_rejected_rows():
    """Verify analysed_claims + rejected_rows == total_uploaded_rows."""
    input_batch = [
        {"ClaimID": "CLM_VAL_1", "ProviderID": "PRV51003", "InscClaimAmtReimbursed": 500, "ClaimStartDt": "2009-01-01"},
        {"ClaimID": "", "ProviderID": "PRV51003", "InscClaimAmtReimbursed": 500, "ClaimStartDt": "2009-01-01"},  # Invalid: blank ClaimID
        {"ClaimID": "CLM_VAL_2", "ProviderID": None, "InscClaimAmtReimbursed": 500, "ClaimStartDt": "2009-01-01"},  # Invalid: None ProviderID
        {"ClaimID": "CLM_VAL_3", "ProviderID": "PRV51003", "InscClaimAmtReimbursed": "corrupted", "ClaimStartDt": "2009-01-01"},  # Invalid: non-numeric
    ]
    res = predict_claims(input_batch)
    assert res["total_uploaded_rows"] == 4
    assert res["analysed_claims"] == 1
    assert res["rejected_rows"] == 3
    assert res["analysed_claims"] + res["rejected_rows"] == res["total_uploaded_rows"]
    assert len(res["rejection_reasons"]) == 3


def test_single_claim_vs_csv_consistency():
    """Verify identical claim evaluated as single vs CSV yields identical provider probability."""
    claim = {
        "ClaimID": "CLM_CONSISTENCY_CHECK",
        "ProviderID": "PRV51003",
        "BeneID": "BENE11001",
        "InscClaimAmtReimbursed": 2500.0,
        "ClaimStartDt": "2009-07-01",
        "ClaimEndDt": "2009-07-03",
        "ClaimType": "Outpatient",
        "AttendingPhysician": "PHY12345",
    }
    # 1. Single claim prediction
    res_single = predict_claims(claim)
    prob_single = res_single["provider_result"]["calibrated_fraud_probability"]

    # 2. Batch DataFrame prediction
    res_batch = predict_claims(pd.DataFrame([claim]))
    prob_batch = res_batch["provider_results"]["PRV51003"]["calibrated_fraud_probability"]

    assert prob_single == prob_batch, f"Mismatch: single={prob_single} vs batch={prob_batch}"


# ══════════════════════════════════════════════════════════════════════════════
# 7. MODEL ARTIFACT AND METRIC PRESERVATION
# ══════════════════════════════════════════════════════════════════════════════

def test_five_main_folders_preserved():
    for f in ["frontend", "backend", "ML", "data", "reports"]:
        p = ROOT / f
        assert p.exists() and p.is_dir(), f"Main folder {f} missing or deleted!"


def test_v1_and_v2_artifacts_preserved():
    """Verify both v1 and v2 directories contain all 13 required files with pristine hashes."""
    for ver_name, art_path, exp_ver in [("v1", ART_V1, "1.0.0"), ("v2", ART_V2, "2.0.0")]:
        assert art_path.exists(), f"Artifact directory {ver_name} missing"
        manifest_path = art_path / "model_manifest.json"
        assert manifest_path.exists(), f"Manifest in {ver_name} missing"

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["model_version"] == exp_ver
        assert manifest["training_status"] == "TRAINED"
        assert len(manifest["artifact_filenames"]) == 13
        assert len(manifest["sha256_checksums"]) == 12

        # Check hashes
        for fname, exp_hash in manifest["sha256_checksums"].items():
            f_bytes = (art_path / fname).read_bytes()
            act_hash = hashlib.sha256(f_bytes).hexdigest()
            assert act_hash == exp_hash, f"Hash mismatch for {fname} in {ver_name}"


def test_risk_thresholds_strictly_preserved():
    """Verify thresholds remain 0.22 and 0.70."""
    with open(ART_V2 / "model_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    thresh = manifest["risk_thresholds"]
    assert thresh["low_medium_cutoff"] == 0.22
    assert thresh["medium_high_cutoff"] == 0.70
