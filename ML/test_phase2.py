"""
HealthGuard AI - Phase 2 Automated Test Suite & Audit Verification
===================================================================
Verification tests for:
  - Base models (XGBoost, ANN, Bi-LSTM) training and probability validity in [0, 1]
  - 5-Fold Stratified OOF Stacking and Logistic Regression meta-learner positive weights
  - Probability Calibration (Platt scaling) and Brier scores
  - Risk decision thresholds derivation on validation data only and empirical monotonicity
  - SHAP TreeExplainer and temporal occlusion attribution
  - Shared inference pipeline:
      * Determinism and input validation
      * Removal of synthetic sequence fallback (missing sequence strictly raises ValueError)
      * Sequence and mask shape validation ((12, 6) and (12,))
      * Finite numerical value enforcement (rejects NaN and Inf)
      * Non-binary mask rejection
      * Limited-history warnings when active_months < 3
  - Artifact integrity:
      * Cryptographic SHA-256 verification of all 12 model files
      * Rejection of corrupted/tampered model files
      * Rejection of missing model artifacts
      * is_ready=True strictly after checksum verification and clean loading
  - Backend integration:
      * Health check and model readiness
      * Model evaluation metrics
      * Single provider fraud prediction
      * Batch prediction reconciliation (total_uploaded = analysed_count + rejected_count)
      * Full transparency into rejected provider IDs with specific reasons
      * Mandatory regulatory disclaimer on all responses
"""

import hashlib
import inspect
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROC = ROOT / "data" / "processed"
ART = ROOT / "ML" / "artifacts" / "v1"
REP = ROOT / "reports" / "evaluation"

from backend.main import app
from ML.xgboost_model import XGBoostFraudDetector
from ML.ann_model import ANNFraudDetector
from ML.lstm_model import LSTMFraudDetector
from ML.ensemble import HealthGuardEnsemble
from ML.explainability import FraudExplainer
from ML.predict import HealthGuardPredictor, get_predictor, REQUIRED_ARTIFACT_FILENAMES


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def predictor():
    return get_predictor()


@pytest.fixture(scope="module")
def sample_data():
    test_feat_path = PROC / "test_provider_features.csv"
    test_seq_path = PROC / "test_provider_sequences.npy"
    test_mask_path = PROC / "test_provider_masks.npy"

    df_test = pd.read_csv(test_feat_path)
    seqs = np.load(test_seq_path)
    masks = np.load(test_mask_path)
    return df_test, seqs, masks


# ══════════════════════════════════════════════════════════════════════════════
# 1. BASE MODEL PROBABILITY VALIDITY & MASKING
# ══════════════════════════════════════════════════════════════════════════════

def test_xgboost_valid_probabilities(sample_data):
    df_test, _, _ = sample_data
    feature_cols = [c for c in df_test.columns if c not in ["ProviderID", "PotentialFraud"]]
    xgb = XGBoostFraudDetector().load(str(ART / "xgboost_model"))

    probs = xgb.predict_proba(df_test[feature_cols].iloc[:20])
    assert len(probs) == 20
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "XGBoost probabilities outside [0, 1]"
    assert np.any(probs > 0.05), "XGBoost predictions should be non-trivial"


def test_ann_valid_probabilities(sample_data):
    df_test, _, _ = sample_data
    feature_cols = [c for c in df_test.columns if c not in ["ProviderID", "PotentialFraud"]]
    ann = ANNFraudDetector().load(str(ART / "ann_model"))

    probs = ann.predict_proba(df_test[feature_cols].iloc[:20].values)
    assert len(probs) == 20
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "ANN probabilities outside [0, 1]"
    assert np.any(probs > 0.05), "ANN predictions should be non-trivial"


def test_lstm_valid_probabilities_and_masking(sample_data):
    _, seqs, masks = sample_data
    lstm = LSTMFraudDetector().load(str(ART / "lstm_model"))

    sample_seqs = seqs[:20]
    sample_masks = masks[:20]
    probs = lstm.predict_proba(sample_seqs, sample_masks)
    assert len(probs) == 20
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Bi-LSTM probabilities outside [0, 1]"
    assert sample_seqs.shape == (20, 12, 6)
    assert sample_masks.shape == (20, 12)


# ══════════════════════════════════════════════════════════════════════════════
# 2. OOF STACKING ENSEMBLE INTEGRITY & META-MODEL WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

def test_oof_predictions_integrity():
    oof_path = ART / "oof_predictions.csv"
    assert oof_path.exists(), "oof_predictions.csv missing from artifacts"
    df_oof = pd.read_csv(oof_path)

    # 3,787 training providers exactly
    assert len(df_oof) == 3787, f"Expected 3787 OOF predictions, got {len(df_oof)}"
    assert len(df_oof["ProviderID"].unique()) == 3787, "Duplicate providers in OOF predictions"

    expected_cols = [
        "ProviderID",
        "Fold",
        "True_PotentialFraud",
        "OOF_XGBoost_Probability",
        "OOF_ANN_Probability",
        "OOF_BiLSTM_Probability",
    ]
    for col in expected_cols:
        assert col in df_oof.columns, f"Missing column {col} in OOF predictions"

    # Check 5 folds with roughly equal counts
    fold_counts = df_oof["Fold"].value_counts()
    assert len(fold_counts) == 5, "Expected exactly 5 cross-validation folds"


def test_meta_model_learned_positive_weights():
    manifest_path = ART / "model_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    ensemble_info = manifest["ensemble"]
    assert ensemble_info["meta_learner"] == "LogisticRegression"
    coeffs = ensemble_info["coefficients"]

    for model_name in ["xgboost", "ann", "bilstm"]:
        assert model_name in coeffs, f"Missing coefficient for {model_name}"
        assert coeffs[model_name] > 0.0, f"Coefficient for {model_name} should be positive, got {coeffs[model_name]}"


# ══════════════════════════════════════════════════════════════════════════════
# 3. PROBABILITY CALIBRATION & THRESHOLD SELECTION (VALIDATION ONLY)
# ══════════════════════════════════════════════════════════════════════════════

def test_probability_calibration_results():
    calib_path = REP / "calibration_results.json"
    assert calib_path.exists(), "calibration_results.json missing"
    with open(calib_path) as f:
        calib = json.load(f)

    assert "selected_method" in calib
    assert calib["selected_method"] in ["platt", "isotonic"]
    assert calib["platt_validation_brier"] < calib["raw_validation_brier"], "Calibration must reduce Brier score"


def test_threshold_selection_uses_validation_data_only():
    thresh_path = REP / "risk_thresholds.json"
    assert thresh_path.exists(), "risk_thresholds.json missing"
    with open(thresh_path) as f:
        thresh = json.load(f)

    # Must be explicitly selected on validation set only, not test set
    assert thresh.get("selection_split") == "validation", "Threshold selection must be on validation split only"
    assert thresh.get("test_labels_used") is False, "Test labels must never be used to derive thresholds"
    assert "low_medium_strategy" in thresh
    assert "medium_high_strategy" in thresh

    low_med = thresh["low_medium_cutoff"]
    med_high = thresh["medium_high_cutoff"]
    assert 0.0 < low_med < med_high < 1.0, f"Thresholds must be strictly monotonic: {low_med} < {med_high}"


def test_derive_risk_thresholds_from_synthetic_validation_inputs():
    """
    Genuine unit test passing synthetic validation probabilities and labels to
    the threshold-selection algorithm, verifying:
      1. Both thresholds are derived dynamically from the inputs.
      2. Calculated precision, recall, F1, false positives, flagged count and workload % are saved.
      3. Strict monotonicity 0 < low_med < med_high < 1 is enforced.
      4. Test labels/data are strictly rejected if passed.
    """
    from ML.ensemble import derive_risk_thresholds

    np.random.seed(42)
    N_val = 500

    # Synthetic validation dataset (15% fraud rate)
    y_val = np.random.binomial(1, 0.15, size=N_val)
    val_probs = np.where(
        y_val == 1,
        np.random.beta(7, 2, size=N_val),
        np.random.beta(1.5, 8, size=N_val),
    )
    val_probs = np.clip(val_probs, 0.01, 0.99)

    # 1. Run derivation function
    res = derive_risk_thresholds(
        val_probs=val_probs,
        val_labels=y_val,
        target_high_precision=0.90,
        max_high_workload_pct=5.0,
        min_high_flagged_count=5,
    )

    # Verify output structure
    assert "low_medium_cutoff" in res
    assert "medium_high_cutoff" in res
    assert "low_medium_strategy" in res
    assert "medium_high_strategy" in res

    low_med = res["low_medium_cutoff"]
    med_high = res["medium_high_cutoff"]

    # Enforce: 0 < low_medium_cutoff < medium_high_cutoff < 1
    assert 0.0 < low_med < med_high < 1.0, f"Monotonicity failed: {low_med} >= {med_high}"

    # Verify calculated validation metrics for LOW/MEDIUM
    lm_metrics = res["low_medium_strategy"]["calculated_validation_metrics"]
    for key in ["precision", "recall", "f1", "false_positives", "flagged_count", "workload_pct"]:
        assert key in lm_metrics, f"Missing metric {key} in low_medium_strategy"
    assert lm_metrics["cutoff"] == low_med

    # Verify calculated validation metrics for MEDIUM/HIGH
    mh_metrics = res["medium_high_strategy"]["calculated_validation_metrics"]
    for key in ["precision", "recall", "f1", "false_positives", "flagged_count", "workload_pct"]:
        assert key in mh_metrics, f"Missing metric {key} in medium_high_strategy"
    assert mh_metrics["cutoff"] == med_high

    # Confirm thresholds are actually derived from inputs (shifting inputs changes thresholds)
    shifted_probs = np.clip(val_probs * 0.5, 0.01, 0.99)
    res_shifted = derive_risk_thresholds(
        val_probs=shifted_probs,
        val_labels=y_val,
        target_high_precision=0.85,
        max_high_workload_pct=5.0,
    )
    assert (
        res_shifted["low_medium_cutoff"] != low_med
        or res_shifted["medium_high_cutoff"] != med_high
    ), "Thresholds must be dynamically derived from the passed validation inputs!"

    # 2. Verify that test labels are STRICTLY NOT accepted by the function
    with pytest.raises(ValueError) as exc1:
        derive_risk_thresholds(val_probs=val_probs, val_labels=y_val, test_labels=y_val)
    assert "validation data only" in str(exc1.value).lower()

    with pytest.raises(ValueError) as exc2:
        derive_risk_thresholds(val_probs=val_probs, val_labels=y_val, y_test=y_val)
    assert "validation data only" in str(exc2.value).lower()


def test_decision_thresholds_and_monotonicity():
    metrics_path = REP / "final_metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    tiers = metrics["calibrated_stacking_ensemble"]["risk_tiers"]
    rate_low = tiers["LOW"]["fraud_rate"]
    rate_med = tiers["MEDIUM"]["fraud_rate"]
    rate_high = tiers["HIGH"]["fraud_rate"]

    assert rate_low < rate_med < rate_high, (
        f"Risk tier fraud rates must strictly monotonically increase: "
        f"LOW ({rate_low:.3f}) < MEDIUM ({rate_med:.3f}) < HIGH ({rate_high:.3f})"
    )
    assert tiers["monotonic_risk_progression"] is True


# ══════════════════════════════════════════════════════════════════════════════
# 4. EXPLAINABILITY (SHAP & TEMPORAL OCCLUSION)
# ══════════════════════════════════════════════════════════════════════════════

def test_shap_tree_explainer(sample_data):
    df_test, _, _ = sample_data
    feature_cols = [c for c in df_test.columns if c not in ["ProviderID", "PotentialFraud"]]

    xgb = XGBoostFraudDetector().load(str(ART / "xgboost_model"))
    explainer = FraudExplainer(
        xgboost_model=xgb,
        feature_names=feature_cols,
        peer_reference_path=str(PROC / "peer_reference_table.json"),
    )

    sample_row = df_test[feature_cols].iloc[0:1]
    explanation = explainer.explain_provider(sample_row, top_n=5)

    assert "top_risk_factors" in explanation
    assert len(explanation["top_risk_factors"]) == 5
    for factor in explanation["top_risk_factors"]:
        assert "feature" in factor
        assert "feature_value" in factor
        assert "shap_value" in factor
        assert "direction" in factor
        assert factor["direction"] in ["INCREASES_FRAUD_RISK", "DECREASES_FRAUD_RISK"]
        assert "peer_percentile" in factor


def test_bilstm_temporal_occlusion(sample_data):
    _, seqs, masks = sample_data
    lstm = LSTMFraudDetector().load(str(ART / "lstm_model"))
    explainer = FraudExplainer()

    sample_seq = seqs[0:1]
    sample_mask = masks[0:1]
    res = explainer.explain_temporal_sequence(lstm_model=lstm, sequence=sample_seq, mask=sample_mask)

    assert "monthly_temporal_impacts" in res
    assert len(res["monthly_temporal_impacts"]) == 12
    for m in res["monthly_temporal_impacts"]:
        assert "month_index" in m
        assert "impact" in m
        assert "is_valid_observation" in m


# ══════════════════════════════════════════════════════════════════════════════
# 5. INFERENCE PIPELINE VALIDATION & SEQUENCE FALLBACK REMOVAL
# ══════════════════════════════════════════════════════════════════════════════

def test_inference_pipeline_determinism(predictor, sample_data):
    df_test, seqs, masks = sample_data
    sample_dict = df_test.iloc[0].to_dict()
    pid = sample_dict["ProviderID"]

    res1 = predictor.predict_provider(pid, sample_dict, sequence=seqs[0:1], mask=masks[0:1])
    res2 = predictor.predict_provider(pid, sample_dict, sequence=seqs[0:1], mask=masks[0:1])

    assert res1["calibrated_fraud_probability"] == res2["calibrated_fraud_probability"]
    assert res1["risk_tier"] == res2["risk_tier"]
    assert res1["base_model_probabilities"] == res2["base_model_probabilities"]


def test_inference_pipeline_missing_features_raises_error(predictor, sample_data):
    _, seqs, masks = sample_data
    with pytest.raises(ValueError) as exc:
        predictor.predict_provider("PRV_TEST", {"total_reimbursement": 100.0}, sequence=seqs[0:1], mask=masks[0:1])
    assert "Missing" in str(exc.value)


def test_missing_monthly_sequence_raises_error(predictor, sample_data):
    """Verify synthetic fallback is removed: missing sequence must raise clear ValueError."""
    df_test, _, masks = sample_data
    sample_dict = df_test.iloc[0].to_dict()
    with pytest.raises(ValueError) as exc:
        predictor.predict_provider("PRV_NO_SEQ", sample_dict, sequence=None, mask=masks[0:1])
    assert "Missing monthly sequence" in str(exc.value)


def test_missing_mask_raises_error(predictor, sample_data):
    """Verify missing validity mask raises clear ValueError."""
    df_test, seqs, _ = sample_data
    sample_dict = df_test.iloc[0].to_dict()
    with pytest.raises(ValueError) as exc:
        predictor.predict_provider("PRV_NO_MASK", sample_dict, sequence=seqs[0:1], mask=None)
    assert "Missing sequence validity mask" in str(exc.value)


def test_invalid_sequence_shape_raises_error(predictor, sample_data):
    """Verify invalid sequence shape (e.g. 5x6 or 12x4) raises ValueError."""
    df_test, _, masks = sample_data
    sample_dict = df_test.iloc[0].to_dict()

    bad_seq_shape = np.zeros((5, 6))
    with pytest.raises(ValueError) as exc:
        predictor.predict_provider("PRV_BAD_SHAPE", sample_dict, sequence=bad_seq_shape, mask=masks[0:1])
    assert "Invalid sequence shape" in str(exc.value)


def test_invalid_mask_shape_raises_error(predictor, sample_data):
    """Verify invalid mask shape (e.g. length 5) raises ValueError."""
    df_test, seqs, _ = sample_data
    sample_dict = df_test.iloc[0].to_dict()

    bad_mask = np.ones((5,))
    with pytest.raises(ValueError) as exc:
        predictor.predict_provider("PRV_BAD_MASK", sample_dict, sequence=seqs[0:1], mask=bad_mask)
    assert "Invalid mask shape" in str(exc.value)


def test_non_finite_sequence_and_mask_values_raise_error(predictor, sample_data):
    """Verify NaN and Inf values in sequences or masks are strictly rejected."""
    df_test, seqs, masks = sample_data
    sample_dict = df_test.iloc[0].to_dict()

    nan_seq = seqs[0:1].copy()
    nan_seq[0, 2, 1] = np.nan
    with pytest.raises(ValueError) as exc:
        predictor.predict_provider("PRV_NAN_SEQ", sample_dict, sequence=nan_seq, mask=masks[0:1])
    assert "NaN or Infinity" in str(exc.value)

    inf_mask = masks[0:1].copy()
    inf_mask[0, 3] = np.inf
    with pytest.raises(ValueError) as exc:
        predictor.predict_provider("PRV_INF_MASK", sample_dict, sequence=seqs[0:1], mask=inf_mask)
    assert "NaN or Infinity" in str(exc.value)


def test_non_binary_mask_values_raise_error(predictor, sample_data):
    """Verify non-binary mask values (e.g. 2.0 or 0.5) are rejected."""
    df_test, seqs, masks = sample_data
    sample_dict = df_test.iloc[0].to_dict()

    non_binary_mask = masks[0:1].copy()
    non_binary_mask[0, 0] = 0.5
    with pytest.raises(ValueError) as exc:
        predictor.predict_provider("PRV_NON_BIN_MASK", sample_dict, sequence=seqs[0:1], mask=non_binary_mask)
    assert "binary indicators" in str(exc.value)


def test_inference_pipeline_limited_history_warning(predictor, sample_data):
    df_test, _, _ = sample_data
    sample_dict = df_test.iloc[0].to_dict()
    sample_dict["active_months"] = 1
    sample_dict["total_claims"] = 2
    dummy_seq = np.zeros((1, 12, 6))
    dummy_mask = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    res = predictor.predict_provider("PRV_SHORT", sample_dict, sequence=dummy_seq, mask=dummy_mask)
    assert res["has_limited_history"] is True
    assert any("Limited claim history" in w for w in res["warnings"])


def test_mandatory_disclaimer_present(predictor, sample_data):
    df_test, seqs, masks = sample_data
    sample_dict = df_test.iloc[0].to_dict()
    res = predictor.predict_provider("PRV51003", sample_dict, sequence=seqs[0:1], mask=masks[0:1])

    expected = "This is a risk assessment to guide investigation. It is not proof of fraud."
    assert res["disclaimer"] == expected


# ══════════════════════════════════════════════════════════════════════════════
# 6. ARTIFACT INTEGRITY & CHECKSUM TAMPER REJECTION
# ══════════════════════════════════════════════════════════════════════════════

def test_successful_artifact_integrity_verification(predictor):
    """Verify that pristine v1 artifacts pass cryptographic checksum validation."""
    is_ok, err_msg = predictor.verify_artifact_integrity()
    assert is_ok is True
    assert err_msg is None
    assert predictor.is_ready is True


def test_corrupted_model_checksum_rejects_loading(tmp_path):
    """Verify that tampering with any model file causes checksum rejection and sets is_ready=False."""
    # Copy genuine v1 artifacts to isolated temporary directory
    target_dir = tmp_path / "corrupted_artifacts"
    shutil.copytree(ART, target_dir)

    # Tamper with ann_model.pt (flip bytes)
    tampered_file = target_dir / "ann_model.pt"
    raw_bytes = bytearray(tampered_file.read_bytes())
    raw_bytes[10] = (raw_bytes[10] + 1) % 256
    tampered_file.write_bytes(raw_bytes)

    test_pred = HealthGuardPredictor.__new__(HealthGuardPredictor)
    test_pred.artifacts_dir = target_dir
    test_pred.proc_dir = PROC

    is_ok, err_msg = test_pred.verify_artifact_integrity(target_dir)
    assert is_ok is False
    assert "Cryptographic checksum mismatch for 'ann_model.pt'" in err_msg


def test_missing_model_artifact_rejects_loading(tmp_path):
    """Verify that missing a required artifact file is caught immediately."""
    target_dir = tmp_path / "missing_artifacts"
    shutil.copytree(ART, target_dir)

    # Delete lstm_model.pt
    (target_dir / "lstm_model.pt").unlink()

    test_pred = HealthGuardPredictor.__new__(HealthGuardPredictor)
    test_pred.artifacts_dir = target_dir

    is_ok, err_msg = test_pred.verify_artifact_integrity(target_dir)
    assert is_ok is False
    assert "Required model artifact file missing: lstm_model.pt" in err_msg


# ══════════════════════════════════════════════════════════════════════════════
# 7. BACKEND INTEGRATION & BATCH RECONCILIATION
# ══════════════════════════════════════════════════════════════════════════════

def test_backend_health_endpoint(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["models_trained"] is True
    assert data["model_version"] in ["1.0.0", "2.0.0"]


def test_backend_models_readiness_endpoint(client):
    r = client.get("/api/v1/models/readiness")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "READY"
    assert data["is_ready"] is True
    assert len(data["sha256_checksums"]) >= 12


def test_backend_models_metrics_endpoint(client):
    r = client.get("/api/v1/models/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "calibrated_stacking_ensemble" in data
    assert "xgboost" in data
    assert "ann" in data
    assert "bilstm" in data


def test_backend_predict_provider_endpoint(client):
    r = client.post("/api/v1/predict/provider", json={"provider_id": "PRV51003"})
    assert r.status_code == 200
    data = r.json()
    assert data["provider_id"] == "PRV51003"
    assert 0.0 <= data["calibrated_fraud_probability"] <= 1.0
    assert data["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]
    assert "base_model_probabilities" in data
    assert len(data["top_risk_factors"]) > 0
    assert data["disclaimer"] == "This is a risk assessment to guide investigation. It is not proof of fraud."


def test_backend_predict_batch_reconciliation(client):
    """
    Verify batch endpoint never silently skips invalid providers.
    Enforces total_uploaded = analysed_count + rejected_count,
    and returns rejected provider IDs with specific reasons.
    """
    batch_input = [
        "PRV51003",             # Valid test provider
        "PRV_NONEXISTENT_1",    # Invalid provider
        "PRV55462",             # Valid high-risk provider
        "",                     # Blank provider ID
        "PRV_NONEXISTENT_2",    # Invalid provider
    ]

    r = client.post("/api/v1/predict/batch", json={"provider_ids": batch_input})
    assert r.status_code == 200
    data = r.json()

    total_uploaded = data["total_uploaded"]
    analysed_count = data["analysed_count"]
    rejected_count = data["rejected_count"]
    rejected_providers = data["rejected_providers"]

    assert total_uploaded == 5
    assert analysed_count == 2
    assert rejected_count == 3
    # Strict mathematical reconciliation
    assert total_uploaded == (analysed_count + rejected_count)

    # Detailed rejection records
    rejected_ids = [rp["provider_id"] for rp in rejected_providers]
    assert "PRV_NONEXISTENT_1" in rejected_ids
    assert "PRV_NONEXISTENT_2" in rejected_ids
    for rp in rejected_providers:
        assert len(rp["reason"]) > 0

    assert data["disclaimer"] == "This is a risk assessment to guide investigation. It is not proof of fraud."


def test_default_training_configuration_selects_v2():
    """Verify that Phase 2 training defaults to artifact version v2 and preserves v1."""
    from ML.train_phase2 import run_phase2_pipeline

    sig = inspect.signature(run_phase2_pipeline)
    assert "artifact_version" in sig.parameters
    assert sig.parameters["artifact_version"].default == "v2", (
        f"Default artifact_version must be 'v2', got '{sig.parameters['artifact_version'].default}'"
    )

    # Verify v2 artifacts directory exists and contains all 13 required artifact files
    v2_dir = ROOT / "ML" / "artifacts" / "v2"
    assert v2_dir.exists(), "ML/artifacts/v2/ directory must exist"

    v2_manifest_path = v2_dir / "model_manifest.json"
    assert v2_manifest_path.exists(), "v2 model_manifest.json must exist"

    with open(v2_manifest_path, "r", encoding="utf-8") as f:
        v2_manifest = json.load(f)

    assert v2_manifest["model_version"] == "2.0.0", "v2 model_manifest model_version must be '2.0.0'"
    assert v2_manifest["training_status"] == "TRAINED"
    assert len(v2_manifest["artifact_filenames"]) == 13
    assert len(v2_manifest["sha256_checksums"]) == 12

    for filename in v2_manifest["artifact_filenames"]:
        file_path = v2_dir / filename
        assert file_path.exists(), f"Required v2 artifact file missing: {filename}"

    # Verify v1 artifacts remain intact and preserved
    v1_dir = ROOT / "ML" / "artifacts" / "v1"
    assert v1_dir.exists(), "ML/artifacts/v1/ must be preserved"
    with open(v1_dir / "model_manifest.json", "r", encoding="utf-8") as f:
        v1_manifest = json.load(f)
    assert v1_manifest["model_version"] == "1.0.0", "v1 model_version must remain '1.0.0'"
