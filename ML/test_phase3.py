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
        "InscClaimAmtReimbursed": 60000.0,
        "ClaimStartDt": "2009-01-01",
    }
    res = detect_claim_anomalies([claim], peer_reimbursement_p95=57000.0)
    assert len(res) == 1
    assert "HIGH_REIMBURSEMENT_P95" in res[0]["anomaly_flags"]


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
        "ClmProcedureCode_1": None,
    }
    res = detect_claim_anomalies([claim])
    assert "MISSING_DIAGNOSIS_INFO" in res[0]["anomaly_flags"]
    assert "MISSING_PROCEDURE_INFO" in res[0]["anomaly_flags"]


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
    assert res["analysed_claims"] == 1
    assert res["rejected_rows"] == 1
    assert res["analysed_claims"] + res["rejected_rows"] == res["total_uploaded_rows"]
    assert res["rejected_records"][0]["ClaimID"] == "CLM_DUP"
    assert "Duplicate ClaimID" in res["rejected_records"][0]["rejection_reason"]
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


# ══════════════════════════════════════════════════════════════════════════════
# 8. REGRESSION TESTS FOR AUDIT FIXES (FIXES 1 THROUGH 6)
# ══════════════════════════════════════════════════════════════════════════════

def test_fix1_lstm_sequence_metadata_matches_feature_engineering_spec():
    """Fix 1: Verify sequence feature order matches feature engineering across all metadata."""
    from ML.feature_engineering import SEQUENCE_FEATURES

    expected_features = [
        "monthly_claim_count",
        "log_total_reimbursement",
        "unique_beneficiaries",
        "ip_claim_pct",
        "avg_reimbursement",
        "repeat_bene_count",
    ]
    assert SEQUENCE_FEATURES == expected_features, "SEQUENCE_FEATURES mismatch in feature_engineering.py"

    # Verify sequence_metadata.json
    seq_meta_file = PROC / "sequence_metadata.json"
    assert seq_meta_file.exists(), "sequence_metadata.json missing"
    with open(seq_meta_file, "r", encoding="utf-8") as f:
        seq_meta = json.load(f)
    assert seq_meta["feature_names"] == expected_features

    # Verify both v1 and v2 manifests
    for art_path in [ART_V1, ART_V2]:
        with open(art_path / "model_manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["sequence_features"] == expected_features, f"Manifest in {art_path} has incorrect sequence_features"


def test_fix2_anomaly_high_reimbursement_peer_p95_threshold():
    """Fix 2: Verify high reimbursement anomaly uses actual peer reference table P95 ($57,000.00)."""
    from ML.claim_anomaly import DEFAULT_PEER_REIMBURSEMENT_P95

    # Check peer reference table
    peer_ref_file = REP / "peer_reference_table.json"
    assert peer_ref_file.exists(), "peer_reference_table.json missing"
    with open(peer_ref_file, "r", encoding="utf-8") as f:
        peer_ref = json.load(f)
    p95_from_table = float(peer_ref["feature_distributions"]["max_reimbursement"]["p95"])
    assert p95_from_table == 57000.0, f"Expected P95=57000.0 in peer table, got {p95_from_table}"
    assert DEFAULT_PEER_REIMBURSEMENT_P95 == 57000.0, f"Expected default P95=57000.0, got {DEFAULT_PEER_REIMBURSEMENT_P95}"

    # Claim just below or equal to P95 -> No anomaly
    claim_norm = {
        "ClaimID": "CLM_NORM",
        "ProviderID": "PRV_P95",
        "InscClaimAmtReimbursed": 57000.0,
        "ClaimStartDt": "2009-01-01",
    }
    res_norm = detect_claim_anomalies([claim_norm])
    assert "HIGH_REIMBURSEMENT_P95" not in res_norm[0]["anomaly_flags"]

    # Claim strictly above P95 -> Triggers HIGH_REIMBURSEMENT_P95
    claim_high = {
        "ClaimID": "CLM_HIGH",
        "ProviderID": "PRV_P95",
        "InscClaimAmtReimbursed": 57001.0,
        "ClaimStartDt": "2009-01-01",
    }
    res_high = detect_claim_anomalies([claim_high])
    assert "HIGH_REIMBURSEMENT_P95" in res_high[0]["anomaly_flags"]


def test_fix3_anomaly_missing_diagnosis_only():
    """Fix 3: Missing diagnosis info flagged separately when procedure code is present."""
    claim = {
        "ClaimID": "CLM_NO_DIAG_ONLY",
        "ProviderID": "PRV_ANOM",
        "InscClaimAmtReimbursed": 100,
        "ClaimStartDt": "2009-01-01",
        "ClmDiagnosisCode_1": None,
        "ClmAdmitDiagnosisCode": None,
        "ClmProcedureCode_1": "9904",
        "AttendingPhysician": "PHY1",
    }
    res = detect_claim_anomalies([claim])
    assert "MISSING_DIAGNOSIS_INFO" in res[0]["anomaly_flags"]
    assert "MISSING_PROCEDURE_INFO" not in res[0]["anomaly_flags"]


def test_fix3_anomaly_missing_procedure_only():
    """Fix 3: Missing procedure info flagged separately when diagnosis code is present."""
    claim = {
        "ClaimID": "CLM_NO_PROC_ONLY",
        "ProviderID": "PRV_ANOM",
        "InscClaimAmtReimbursed": 100,
        "ClaimStartDt": "2009-01-01",
        "ClmDiagnosisCode_1": "4019",
        "ClmProcedureCode_1": None,
        "ClmProcedureCode_2": None,
        "AttendingPhysician": "PHY1",
    }
    res = detect_claim_anomalies([claim])
    assert "MISSING_PROCEDURE_INFO" in res[0]["anomaly_flags"]
    assert "MISSING_DIAGNOSIS_INFO" not in res[0]["anomaly_flags"]


def test_fix3_anomaly_both_diagnosis_and_procedure_missing():
    """Fix 3: Both missing flags raised when both sets of codes are absent."""
    claim = {
        "ClaimID": "CLM_BOTH_MISSING",
        "ProviderID": "PRV_ANOM",
        "InscClaimAmtReimbursed": 100,
        "ClaimStartDt": "2009-01-01",
        "ClmDiagnosisCode_1": None,
        "ClmAdmitDiagnosisCode": None,
        "ClmProcedureCode_1": None,
        "AttendingPhysician": "PHY1",
    }
    res = detect_claim_anomalies([claim])
    assert "MISSING_DIAGNOSIS_INFO" in res[0]["anomaly_flags"]
    assert "MISSING_PROCEDURE_INFO" in res[0]["anomaly_flags"]


def test_fix3_anomaly_both_diagnosis_and_procedure_available():
    """Fix 3: Neither flag raised when both diagnosis and procedure codes exist."""
    claim = {
        "ClaimID": "CLM_BOTH_PRESENT",
        "ProviderID": "PRV_ANOM",
        "InscClaimAmtReimbursed": 100,
        "ClaimStartDt": "2009-01-01",
        "ClmDiagnosisCode_1": "4019",
        "ClmProcedureCode_1": "9904",
        "AttendingPhysician": "PHY1",
    }
    res = detect_claim_anomalies([claim])
    assert "MISSING_DIAGNOSIS_INFO" not in res[0]["anomaly_flags"]
    assert "MISSING_PROCEDURE_INFO" not in res[0]["anomaly_flags"]


def test_fix3_anomaly_diagnosis_procedure_columns_not_present_in_input():
    """Fix 3: When diagnosis or procedure columns are omitted from input, record warning without false anomalies."""
    claim = {
        "ClaimID": "CLM_COL_ABSENT",
        "ProviderID": "PRV_ANOM",
        "InscClaimAmtReimbursed": 100,
        "ClaimStartDt": "2009-01-01",
        "AttendingPhysician": "PHY1",
    }
    res = detect_claim_anomalies([claim])
    assert "MISSING_DIAGNOSIS_INFO" not in res[0]["anomaly_flags"]
    assert "MISSING_PROCEDURE_INFO" not in res[0]["anomaly_flags"]
    warnings = res[0].get("data_availability_warnings", [])
    assert any("Diagnosis code columns" in w for w in warnings)
    assert any("Procedure code columns" in w for w in warnings)


def test_fix4_predict_claims_duplicate_claim_id_within_csv_rejected():
    """Fix 4: Duplicate ClaimIDs within uploaded batch are rejected with row index and reason."""
    claims = [
        {"ClaimID": "CLM_001", "ProviderID": "PRV51003", "BeneID": "B01", "InscClaimAmtReimbursed": 100, "ClaimStartDt": "2009-01-01"},
        {"ClaimID": "CLM_002", "ProviderID": "PRV51003", "BeneID": "B02", "InscClaimAmtReimbursed": 200, "ClaimStartDt": "2009-02-01"},
        {"ClaimID": "CLM_001", "ProviderID": "PRV51003", "BeneID": "B01", "InscClaimAmtReimbursed": 100, "ClaimStartDt": "2009-01-01"},  # duplicate
    ]
    res = predict_claims(claims)
    assert res["total_uploaded_rows"] == 3
    assert res["analysed_claims"] == 2
    assert res["rejected_rows"] == 1
    assert res["analysed_claims"] + res["rejected_rows"] == res["total_uploaded_rows"]

    rej = res["rejected_records"][0]
    assert rej["ClaimID"] == "CLM_001"
    assert rej["row_index"] == 2
    assert "Duplicate ClaimID" in rej["rejection_reason"]


def test_fix4_predict_claims_historical_claim_id_replacement():
    """Fix 4 & 5: Uploaded claim with same ClaimID as historical record replaces it without duplicate billing."""
    # Build historical claims
    hist_df = pd.DataFrame([
        {"ClaimID": "CLM_HIST_1", "ProviderID": "PRV_HIST", "BeneID": "B_HIST", "InscClaimAmtReimbursed": 500, "ClaimStartDt": "2009-01-01", "ClaimType": "Outpatient"},
        {"ClaimID": "CLM_HIST_2", "ProviderID": "PRV_HIST", "BeneID": "B_HIST", "InscClaimAmtReimbursed": 300, "ClaimStartDt": "2009-01-15", "ClaimType": "Outpatient"},
    ])
    # Uploaded update for CLM_HIST_2
    new_claim = {
        "ClaimID": "CLM_HIST_2",
        "ProviderID": "PRV_HIST",
        "BeneID": "B_HIST",
        "InscClaimAmtReimbursed": 450,
        "ClaimStartDt": "2009-01-15",
        "ClaimType": "Outpatient",
    }
    res = predict_claims([new_claim], historical_claims_df=hist_df)
    assert res["total_uploaded_rows"] == 1
    assert res["analysed_claims"] == 1
    assert res["rejected_rows"] == 0


def test_fix5_repeat_beneficiary_uses_provider_history():
    """Fix 5: Repeat beneficiary counts incorporate historical claims for that provider."""
    # Historical claim has 3 claims for BENE_REP
    hist_df = pd.DataFrame([
        {"ClaimID": f"HIST_CLM_{i}", "ProviderID": "PRV_REP", "BeneID": "BENE_REP", "InscClaimAmtReimbursed": 100, "ClaimStartDt": "2009-01-01"}
        for i in range(3)
    ])
    # New claim for same beneficiary
    new_claim = {
        "ClaimID": "NEW_CLM_01",
        "ProviderID": "PRV_REP",
        "BeneID": "BENE_REP",
        "InscClaimAmtReimbursed": 100,
        "ClaimStartDt": "2009-02-01",
    }
    # Anomaly detection with historical context
    res = detect_claim_anomalies([new_claim], historical_claims_df=hist_df, repeat_bene_threshold=3)
    assert len(res) == 1
    assert "REPEAT_BENEFICIARY" in res[0]["anomaly_flags"]


def test_fix5_repeat_beneficiary_cross_provider_isolation():
    """Fix 5: Same BeneID under different providers does not trigger repeat beneficiary across providers."""
    hist_df = pd.DataFrame([
        {"ClaimID": f"PRV1_CLM_{i}", "ProviderID": "PRV_ONE", "BeneID": "BENE_SHARED", "InscClaimAmtReimbursed": 100, "ClaimStartDt": "2009-01-01"}
        for i in range(5)
    ])
    # New claim for same beneficiary under PRV_TWO (which has 0 history)
    new_claim = {
        "ClaimID": "PRV2_CLM_01",
        "ProviderID": "PRV_TWO",
        "BeneID": "BENE_SHARED",
        "InscClaimAmtReimbursed": 100,
        "ClaimStartDt": "2009-02-01",
    }
    res = detect_claim_anomalies([new_claim], historical_claims_df=hist_df, repeat_bene_threshold=3)
    assert "REPEAT_BENEFICIARY" not in res[0]["anomaly_flags"]


def test_fix6_reconciliation_when_provider_prediction_fails(monkeypatch):
    """Fix 6: If provider prediction fails, claims are moved to rejected_records and total reconciliation holds."""
    # Monkeypatch predict_provider to simulate a provider evaluation failure
    def mock_predict_failure(*args, **kwargs):
        raise ValueError("Simulated provider feature computation error")

    pred = get_predictor()
    monkeypatch.setattr(pred, "predict_provider", mock_predict_failure)

    claims = [
        {"ClaimID": "CLM_FAIL_1", "ProviderID": "PRV51003", "BeneID": "B1", "InscClaimAmtReimbursed": 100, "ClaimStartDt": "2009-01-01"},
        {"ClaimID": "CLM_FAIL_2", "ProviderID": "PRV51003", "BeneID": "B2", "InscClaimAmtReimbursed": 200, "ClaimStartDt": "2009-01-02"},
    ]
    res = predict_claims(claims)

    assert res["total_uploaded_rows"] == 2
    assert res["analysed_claims"] == 0
    assert res["rejected_rows"] == 2
    assert res["analysed_claims"] + res["rejected_rows"] == res["total_uploaded_rows"]
    assert len(res["rejected_records"]) == 2
    for r in res["rejected_records"]:
        assert "Provider analysis failed" in r["rejection_reason"]


def test_fix6_reconciliation_mixed_success_and_failure_providers(monkeypatch):
    """Fix 6: Mixed batch where one provider succeeds and one provider fails maintains exact reconciliation."""
    pred = get_predictor()
    orig_method = pred.predict_provider

    def mock_conditional_predict(provider_id, features, sequence=None, mask=None, **kwargs):
        # Fail only for PRV_FAIL
        if str(provider_id) == "PRV_FAIL":
            raise RuntimeError("Provider PRV_FAIL sequence validation error")
        return orig_method(provider_id, features, sequence=sequence, mask=mask, **kwargs)

    monkeypatch.setattr(pred, "predict_provider", mock_conditional_predict)

    claims = [
        # Provider 1: PRV51003 (valid)
        {"ClaimID": "CLM_SUCC_1", "ProviderID": "PRV51003", "BeneID": "B1", "InscClaimAmtReimbursed": 100, "ClaimStartDt": "2009-01-01"},
        {"ClaimID": "CLM_SUCC_2", "ProviderID": "PRV51003", "BeneID": "B2", "InscClaimAmtReimbursed": 200, "ClaimStartDt": "2009-01-02"},
        # Provider 2: PRV_FAIL (will fail during prediction)
        {"ClaimID": "CLM_ERR_1", "ProviderID": "PRV_FAIL", "BeneID": "B3", "InscClaimAmtReimbursed": 300, "ClaimStartDt": "2009-01-03"},
    ]
    res = predict_claims(claims)

    assert res["total_uploaded_rows"] == 3
    assert res["analysed_claims"] == 2
    assert res["rejected_rows"] == 1
    assert res["analysed_claims"] + res["rejected_rows"] == res["total_uploaded_rows"]
    assert "PRV51003" in res["provider_results"]
    assert "PRV_FAIL" not in res["provider_results"]
    assert res["rejected_records"][0]["ClaimID"] == "CLM_ERR_1"

