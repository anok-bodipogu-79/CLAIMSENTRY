"""
HealthGuard AI - Phase 1 Automated Test Suite
==============================================
Validates:
  - Provider-level splitting and lack of overlap
  - Sequence and mask shape and validity
  - No data leakage across splits
  - No protected demographic features in model feature set
  - Feature specification matching actual columns
  - Comprehensive peer reference table fitted strictly on train split
  - Untrained ML model calls raising clear exceptions
  - Backend routes rejecting premature predictions with HTTP 503
  - Strict preservation of the five main project folders
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROC = ROOT / "data" / "processed"


def test_five_main_folders_preserved():
    """Verify strictly the five main project folders exist."""
    expected_folders = {"frontend", "backend", "ML", "data", "reports"}
    actual_dirs = {
        d.name for d in ROOT.iterdir()
        if d.is_dir() and not d.name.startswith((".", "__"))
    }
    assert actual_dirs == expected_folders, f"Unexpected top-level directories: {actual_dirs}"


def test_processed_files_exist():
    expected = [
        "cleaned_training_claims.csv",
        "cleaned_training_beneficiaries.csv",
        "provider_labels.csv",
        "train_provider_features.csv",
        "validation_provider_features.csv",
        "test_provider_features.csv",
        "train_provider_sequences.npy",
        "validation_provider_sequences.npy",
        "test_provider_sequences.npy",
        "train_provider_masks.npy",
        "validation_provider_masks.npy",
        "test_provider_masks.npy",
        "train_provider_ids.csv",
        "validation_provider_ids.csv",
        "test_provider_ids.csv",
        "train_sequence_provider_ids.csv",
        "validation_sequence_provider_ids.csv",
        "test_sequence_provider_ids.csv",
        "peer_reference_table.json",
        "feature_spec.json",
        "sequence_metadata.json",
    ]
    for fname in expected:
        path = PROC / fname
        assert path.exists(), f"Missing expected output: {fname}"
        assert path.stat().st_size > 0, f"Empty output file: {fname}"


def test_provider_split_no_overlap():
    train_ids = set(pd.read_csv(PROC / "train_provider_ids.csv")["ProviderID"])
    val_ids = set(pd.read_csv(PROC / "validation_provider_ids.csv")["ProviderID"])
    test_ids = set(pd.read_csv(PROC / "test_provider_ids.csv")["ProviderID"])

    assert train_ids.isdisjoint(val_ids), "Train and validation provider IDs overlap!"
    assert train_ids.isdisjoint(test_ids), "Train and test provider IDs overlap!"
    assert val_ids.isdisjoint(test_ids), "Validation and test provider IDs overlap!"


def test_feature_columns_consistent():
    train_f = pd.read_csv(PROC / "train_provider_features.csv")
    val_f = pd.read_csv(PROC / "validation_provider_features.csv")
    test_f = pd.read_csv(PROC / "test_provider_features.csv")

    assert set(train_f.columns) == set(val_f.columns), "Train and Val feature columns differ!"
    assert set(train_f.columns) == set(test_f.columns), "Train and Test feature columns differ!"


def test_no_demographic_features():
    train_f = pd.read_csv(PROC / "train_provider_features.csv")
    forbidden = {"Gender", "Race", "State", "County"}
    found = forbidden & set(train_f.columns)
    assert len(found) == 0, f"Found sensitive demographic columns in feature set: {found}"


def test_sequence_and_mask_shapes():
    for split in ["train", "validation", "test"]:
        s_arr = np.load(PROC / f"{split}_provider_sequences.npy")
        m_arr = np.load(PROC / f"{split}_provider_masks.npy")
        pids = pd.read_csv(PROC / f"{split}_sequence_provider_ids.csv")

        assert s_arr.ndim == 3, f"{split} sequences should be 3D"
        assert m_arr.ndim == 2, f"{split} masks should be 2D"
        assert s_arr.shape[0] == m_arr.shape[0] == len(pids), f"{split} row counts mismatch"
        assert s_arr.shape[1] == 12, f"{split} sequence length should be 12"
        assert s_arr.shape[2] == 6, f"{split} sequence feature count should be 6"
        assert m_arr.shape[1] == 12, f"{split} mask length should be 12"
        assert np.isin(m_arr, [0.0, 1.0]).all(), f"{split} masks should contain only 0.0 and 1.0"


def test_feature_spec_matches_features():
    train_f = pd.read_csv(PROC / "train_provider_features.csv")
    with open(PROC / "feature_spec.json") as f:
        spec = json.load(f)

    expected_cols = set(train_f.columns) - {"ProviderID", "PotentialFraud"}
    assert set(spec.keys()) == expected_cols, "feature_spec.json keys do not match feature columns!"


def test_peer_reference_table_trained_on_train_split():
    with open(PROC / "peer_reference_table.json") as f:
        ref = json.load(f)

    assert ref.get("fit_split") == "train", "Peer reference table must be fit on train split only!"
    assert ref.get("n_training_providers") == 3787, "Peer reference table should have 3787 providers"
    assert "feature_distributions" in ref, "Peer reference table missing feature_distributions"
    assert len(ref["feature_distributions"]) > 30, "Comprehensive feature distributions required"

    # Check that key percentiles are present
    dist = ref["feature_distributions"]["total_reimbursement"]
    for q in ["p10", "p25", "p50", "p75", "p90", "p95", "median", "mean", "std"]:
        assert q in dist, f"Percentile {q} missing from peer reference distribution"


def test_untrained_models_raise_error():
    """Verify that calling predict/predict_proba on untrained models raises clear RuntimeError."""
    from ML.xgboost_model import XGBoostFraudDetector
    from ML.ann_model import ANNFraudDetector
    from ML.lstm_model import LSTMFraudDetector
    from ML.ensemble import HealthGuardEnsemble

    dummy_2d = pd.DataFrame(np.zeros((2, 5)))
    dummy_3d = np.zeros((2, 12, 6))

    with pytest.raises(RuntimeError):
        XGBoostFraudDetector().predict_proba(dummy_2d)

    with pytest.raises(RuntimeError):
        ANNFraudDetector().predict_proba(dummy_2d.values)

    with pytest.raises(RuntimeError):
        LSTMFraudDetector().predict_proba(dummy_3d)

    with pytest.raises(RuntimeError):
        HealthGuardEnsemble().predict_proba({"xgboost": 0.5})


def test_backend_routes_return_service_unavailable():
    """Verify that Phase 1 prediction endpoints return HTTP 503 instead of dummy numbers."""
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)

    # Health check should be 200
    r_health = client.get("/api/v1/health")
    assert r_health.status_code == 200

    if not r_health.json()["models_trained"]:
        # When models are untrained (Phase 1 mode), prediction endpoints must return 503
        r_pred = client.post("/api/v1/predict/provider", json={"provider_id": "PRV51001", "features": {}})
        assert r_pred.status_code == 503
        assert "unavailable" in r_pred.json()["detail"].lower()

        r_batch = client.post("/api/v1/predict/batch", json={"provider_ids": ["PRV51001"]})
        assert r_batch.status_code == 503
    else:
        # Phase 2 trained models are available
        assert r_health.json()["models_trained"] is True
        assert r_health.json()["version"] == "1.0.0"

    # Summary report should return verified data without optimistic fallback defaults
    r_sum = client.get("/api/v1/reports/summary")
    assert r_sum.status_code == 200
    sum_data = r_sum.json()
    assert sum_data["status"] == "PHASE_1_VERIFIED"
    assert sum_data["checks_passed"] is True
    assert sum_data["provider_split"]["train_count"] == 3787
    assert sum_data["provider_split"]["val_count"] == 811
    assert sum_data["provider_split"]["test_count"] == 812
    assert sum_data["data_leakage_free"] is True
    assert sum_data["feature_count"] == 51


def _make_valid_verification_dict():
    """Helper creating a complete passing Phase 1 verification payload."""
    return {
        "provider_overlap": {
            "train_count": 3787,
            "val_count": 811,
            "test_count": 812,
            "train_val_overlap": 0,
            "train_test_overlap": 0,
            "val_test_overlap": 0,
            "overlap_free": True,
        },
        "fraud_distribution": {
            "train": {"total": 3787, "fraud": 354, "fraud_pct": 9.35},
            "validation": {"total": 811, "fraud": 76, "fraud_pct": 9.37},
            "test": {"total": 812, "fraud": 76, "fraud_pct": 9.36},
        },
        "leakage": {
            "train_val_claim_overlap": 0,
            "train_test_claim_overlap": 0,
            "val_test_claim_overlap": 0,
            "leakage_free": True,
        },
        "feature_spec": {
            "n_features_in_spec": 51,
            "n_features_in_data": 51,
            "spec_matches_data": True,
            "missing_in_spec": [],
            "extra_in_spec": [],
        },
        "demographic_check": {
            "clean": True,
            "found_demographics": [],
        },
    }


def test_summary_missing_verification_json(tmp_path):
    """Verify summary returns PHASE_1_UNAVAILABLE when verification JSON file does not exist."""
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    with patch("backend.routes.REP", tmp_path):
        res = client.get("/api/v1/reports/summary")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "PHASE_1_UNAVAILABLE"
        assert data["checks_passed"] is False
        assert "not found" in data["message"].lower()


def test_summary_empty_verification_json(tmp_path):
    """Verify summary returns PHASE_1_INCOMPLETE when verification JSON file is empty."""
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from backend.main import app

    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "phase1_verification.json").write_text("{}", encoding="utf-8")

    client = TestClient(app)
    with patch("backend.routes.REP", tmp_path):
        res = client.get("/api/v1/reports/summary")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "PHASE_1_INCOMPLETE"
        assert data["checks_passed"] is False
        assert "missing required verification sections" in data["message"].lower()


def test_summary_missing_required_fields(tmp_path):
    """Verify summary returns PHASE_1_INCOMPLETE when critical sections are missing."""
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from backend.main import app

    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    # Payload missing feature_spec and demographic_check
    payload = {
        "provider_overlap": {"overlap_free": True, "train_count": 3787, "val_count": 811, "test_count": 812},
        "leakage": {"leakage_free": True},
        "fraud_distribution": {"train": {}},
    }
    (eval_dir / "phase1_verification.json").write_text(json.dumps(payload), encoding="utf-8")

    client = TestClient(app)
    with patch("backend.routes.REP", tmp_path):
        res = client.get("/api/v1/reports/summary")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "PHASE_1_INCOMPLETE"
        assert data["checks_passed"] is False
        assert "feature_spec" in data["message"]
        assert "demographic_check" in data["message"]


def test_summary_leakage_detected(tmp_path):
    """Verify summary returns PHASE_1_FAILED when data leakage is detected."""
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from backend.main import app

    payload = _make_valid_verification_dict()
    payload["leakage"]["leakage_free"] = False
    payload["leakage"]["train_val_claim_overlap"] = 42

    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "phase1_verification.json").write_text(json.dumps(payload), encoding="utf-8")

    client = TestClient(app)
    with patch("backend.routes.REP", tmp_path):
        res = client.get("/api/v1/reports/summary")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "PHASE_1_FAILED"
        assert data["checks_passed"] is False
        assert any("leakage" in fail.lower() for fail in data["failures"])


def test_summary_feature_spec_mismatch(tmp_path):
    """Verify summary returns PHASE_1_FAILED when feature specification mismatches generated data."""
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from backend.main import app

    payload = _make_valid_verification_dict()
    payload["feature_spec"]["spec_matches_data"] = False
    payload["feature_spec"]["missing_in_spec"] = ["untracked_feature_x"]

    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "phase1_verification.json").write_text(json.dumps(payload), encoding="utf-8")

    client = TestClient(app)
    with patch("backend.routes.REP", tmp_path):
        res = client.get("/api/v1/reports/summary")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "PHASE_1_FAILED"
        assert data["checks_passed"] is False
        assert any("feature specification" in fail.lower() for fail in data["failures"])


def test_summary_demographic_check_failure(tmp_path):
    """Verify summary returns PHASE_1_FAILED when protected demographic attributes are detected."""
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from backend.main import app

    payload = _make_valid_verification_dict()
    payload["demographic_check"]["clean"] = False
    payload["demographic_check"]["found_demographics"] = ["Bene_Race", "Bene_Gender"]

    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "phase1_verification.json").write_text(json.dumps(payload), encoding="utf-8")

    client = TestClient(app)
    with patch("backend.routes.REP", tmp_path):
        res = client.get("/api/v1/reports/summary")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "PHASE_1_FAILED"
        assert data["checks_passed"] is False
        assert any("demographic" in fail.lower() for fail in data["failures"])


def test_summary_successful_verification_all_checks_pass():
    """Verify summary returns PHASE_1_VERIFIED strictly when all required validation checks pass."""
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    res = client.get("/api/v1/reports/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "PHASE_1_VERIFIED"
    assert data["checks_passed"] is True
    assert data["failures"] is None
    assert data["data_leakage_free"] is True
    assert data["feature_count"] == 51
    assert data["demographic_clean"] is True
    assert data["spec_matches_data"] is True
    assert data["provider_split"]["overlap_free"] is True
    assert data["provider_split"]["train_count"] == 3787
    assert data["provider_split"]["val_count"] == 811
    assert data["provider_split"]["test_count"] == 812
