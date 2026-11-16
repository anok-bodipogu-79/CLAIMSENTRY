"""
HealthGuard AI - Phase 4 ML Integration Tests
Verifies real model loading, v1 and v2 version switching, risk tier thresholds, and SHAP explanations.
"""

import pytest
from backend.app.services.prediction_service import prediction_service
from backend.app.services.explanation_service import explanation_service
from ML.predict import HealthGuardPredictor


def test_predictor_v1_and_v2_readiness():
    """Verify that both v1 and v2 models load properly with valid artifacts."""
    pred_v1 = HealthGuardPredictor(artifact_version="v1")
    assert pred_v1 is not None
    assert pred_v1.is_ready is True
    assert pred_v1.artifact_version == "v1"

    pred_v2 = HealthGuardPredictor(artifact_version="v2")
    assert pred_v2 is not None
    assert pred_v2.is_ready is True
    assert pred_v2.artifact_version == "v2"


def test_risk_tier_classification_logic():
    """Verify calibrated probabilities map to correct risk tiers: LOW, MEDIUM, HIGH."""
    pred = prediction_service.predictor
    thresholds = pred.ensemble.risk_thresholds
    low_thresh = thresholds["low_medium_cutoff"]   # 0.22
    high_thresh = thresholds["medium_high_cutoff"] # 0.70

    assert 0.0 < low_thresh < high_thresh < 1.0

    # Low tier
    assert pred.ensemble.assign_risk_category(0.10) == "LOW"
    assert pred.ensemble.assign_risk_category(low_thresh - 0.01) == "LOW"

    # Medium tier
    assert pred.ensemble.assign_risk_category(low_thresh) == "MEDIUM"
    assert pred.ensemble.assign_risk_category(0.50) == "MEDIUM"
    assert pred.ensemble.assign_risk_category(high_thresh - 0.01) == "MEDIUM"

    # High tier
    assert pred.ensemble.assign_risk_category(high_thresh) == "HIGH"
    assert pred.ensemble.assign_risk_category(0.95) == "HIGH"


def test_shap_explanation_formatting():
    """Verify explanation formatting produces standard title and narrative."""
    eval_result = {
        "calibrated_fraud_probability": 0.88,
        "risk_tier": "HIGH",
        "top_risk_factors": [
            {"feature": "monthly_claim_count_mean", "value": 150.0, "description": "High claim frequency"},
            {"feature": "reimbursement_mean", "value": 4200.0, "description": "Elevated reimbursement amount"},
        ],
    }

    formatted = explanation_service.format_explanation(eval_result)
    assert formatted["title"] == "AI Explanation \u2013 Why This Claim Received This Risk Score"
    assert "HIGH" in formatted["text"]
    assert "Key statistical risk drivers" in formatted["text"]


def test_settings_model_version_switching(client, admin_headers):
    """Verify switching active model version via /api/settings."""
    # Get current settings
    res = client.get("/api/settings", headers=admin_headers)
    assert res.status_code == 200
    current_version = res.json()["model_version"]

    target_version = "v1" if current_version == "v2" else "v2"

    # Switch version
    update_res = client.put(
        "/api/settings",
        json={"model_version": target_version},
        headers=admin_headers,
    )
    assert update_res.status_code == 200
    assert update_res.json()["model_version"] == target_version

    # Restore version
    client.put(
        "/api/settings",
        json={"model_version": current_version},
        headers=admin_headers,
    )
