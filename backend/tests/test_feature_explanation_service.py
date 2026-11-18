"""
Unit tests for Feature Explanation Service, Investigator Decision Report, and Decision Persistence.
Tests catalog coverage, severity assignment, currency formatting, model agreement, and decision recording.
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.services.feature_explanation_service import (
    FeatureExplanationService,
    format_inr_currency,
    format_usd_currency,
    FEATURE_CATALOG,
)
from backend.app.main import app
from backend.app.core.dependencies import get_current_user
from backend.app.db.models import User


def test_feature_catalog_coverage():
    """Verify all 51 features are cataloged across expected categories."""
    assert len(FEATURE_CATALOG) >= 51
    required_features = [
        "total_reimbursement",
        "avg_monthly_reimbursement",
        "reimbursement_to_peer_median",
        "total_claim_count",
        "unique_beneficiaries",
        "avg_beneficiary_age",
        "avg_inpatient_length_of_stay",
        "procedure_code_entropy",
        "diagnosis_code_entropy",
        "inpatient_claim_ratio",
    ]
    for feat in required_features:
        meta = FeatureExplanationService.get_feature_meta(feat)
        assert meta is not None
        assert "display_name" in meta
        assert "category" in meta
        assert "scope" in meta
        assert "unit" in meta


def test_inr_currency_formatting():
    """Verify Indian numbering format with correct comma groupings."""
    assert format_inr_currency(0.0) == "₹0.00"
    assert format_inr_currency(1234.56) == "₹1,234.56"
    assert format_inr_currency(123456.78) == "₹1,23,456.78"
    assert format_inr_currency(12345678.90) == "₹1,23,45,678.90"
    assert format_inr_currency(1189750.00) == "₹11,89,750.00"


def test_usd_currency_formatting():
    """Verify standard USD formatting."""
    assert format_usd_currency(0.0) == "$0.00"
    assert format_usd_currency(1234567.89) == "$1,234,567.89"


def test_severity_classification_rules():
    """Verify deterministic severity assignments for risk elevating and mitigating features."""
    # Rank 1 positive SHAP -> Major Concern
    finding_1 = FeatureExplanationService.generate_plain_english_finding(
        feature_name="total_reimbursement",
        observed_value=20000.0,
        shap_value=0.45,
        peer_median=5000.0,
        currency="INR",
        exchange_rate=83.50,
        rank=1,
    )
    assert finding_1["severity"] == "Major Concern"
    assert finding_1["effect"] == "INCREASES_RISK"
    assert "₹" in finding_1["observed_value_formatted"]

    # Rank 4 positive SHAP < 0.05 -> Informational
    finding_info = FeatureExplanationService.generate_plain_english_finding(
        feature_name="unique_beneficiaries",
        observed_value=50.0,
        shap_value=0.02,
        peer_median=45.0,
        currency="INR",
        rank=4,
    )
    assert finding_info["severity"] == "Informational"

    # Negative SHAP -> Mitigating Factor
    finding_mitigating = FeatureExplanationService.generate_plain_english_finding(
        feature_name="avg_beneficiary_age",
        observed_value=72.0,
        shap_value=-0.08,
        peer_median=71.5,
        currency="INR",
        rank=1,
    )
    assert finding_mitigating["severity"] == "Mitigating Factor"
    assert finding_mitigating["effect"] == "REDUCES_RISK"


def test_model_agreement_evaluation():
    """Verify model consensus evaluation across XGBoost, ANN, Bi-LSTM, Stacking, and Platt."""
    # Consensus HIGH
    assessment_agree = FeatureExplanationService.evaluate_model_agreement(
        base_probs={"xgboost": 0.85, "ann": 0.80, "bilstm": 0.78},
        calibrated_prob=0.83,
        thresholds={"low_medium_cutoff": 0.22, "medium_high_cutoff": 0.70},
    )
    assert assessment_agree["consensus_status"] == "Models Agree"
    assert len(assessment_agree["component_models"]) == 3

    # Partial Agreement (one disagrees)
    assessment_partial = FeatureExplanationService.evaluate_model_agreement(
        base_probs={"xgboost": 0.85, "ann": 0.50, "bilstm": 0.75},
        calibrated_prob=0.73,
        thresholds={"low_medium_cutoff": 0.22, "medium_high_cutoff": 0.70},
    )
    assert assessment_partial["consensus_status"] == "Partial Agreement"


def test_investigation_actions_generation():
    """Verify action items tailored to findings."""
    top_positive_factors = [
        {"feature_name": "total_reimbursement", "severity": "Major Concern"},
        {"feature_name": "ip_total_reimbursement", "severity": "Concern"},
    ]
    actions = FeatureExplanationService.generate_investigation_actions(
        top_positive_factors=top_positive_factors,
        risk_tier="HIGH",
        anomaly_flags=[],
    )
    assert len(actions) >= 3
    assert any("Audit high-value reimbursement" in a for a in actions)


def test_record_investigator_decision_endpoint():
    """Verify /api/analyzer/decision records case note and updates status."""
    mock_user = User(
        id=1,
        name="Jane Investigator",
        email="investigator@healthguard.ai",
        password_hash="mock",
        role="investigator",
        is_active=True,
    )
    app.dependency_overrides[get_current_user] = lambda: mock_user
    try:
        client = TestClient(app)
        payload = {
            "claim_id": "CLM_DECISION_TEST",
            "provider_id": "PRV51003",
            "decision_status": "Under Investigation",
            "notes": "Verified high inpatient reimbursement variance. Scheduled full clinical chart audit.",
        }

        res = client.post("/api/analyzer/decision", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["claim_id"] == "CLM_DECISION_TEST"
        assert data["provider_id"] == "PRV51003"
        assert data["decision_status"] == "Under Investigation"
        assert data["audit_logged"] is True
    finally:
        app.dependency_overrides.pop(get_current_user, None)
