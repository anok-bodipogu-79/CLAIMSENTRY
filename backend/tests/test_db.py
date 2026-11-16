"""
HealthGuard AI - Phase 4 Database Tests
Verifies table creation, unique constraints, foreign keys, relationships, and rollbacks.
"""

import pytest
from sqlalchemy.exc import IntegrityError
from backend.app.db.database import Base
from backend.app.db.models import (
    User, ModelVersion, Provider, Beneficiary, Claim,
    ProviderRiskScore, ClaimAnalysis, Explanation, CSVUpload,
    CSVUploadRow, CaseNote, AuditLog
)


def test_tables_created(db_session):
    """Verify that all 12 Phase 4 tables are properly registered in metadata."""
    expected_tables = {
        "users", "model_versions", "providers", "beneficiaries", "claims",
        "provider_risk_scores", "claim_analyses", "explanations",
        "csv_uploads", "csv_upload_rows", "case_notes", "audit_logs"
    }
    actual_tables = set(Base.metadata.tables.keys())
    assert expected_tables.issubset(actual_tables), f"Missing tables: {expected_tables - actual_tables}"


def test_user_unique_email_constraint(db_session):
    """Verify unique constraint on User.email."""
    u1 = User(
        email="duplicate@healthguard.ai",
        name="User One",
        password_hash="hash",
        role="INVESTIGATOR",
    )
    db_session.add(u1)
    db_session.commit()

    u2 = User(
        email="duplicate@healthguard.ai",
        name="User Two",
        password_hash="hash",
        role="INVESTIGATOR",
    )
    db_session.add(u2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_claim_unique_claim_id_constraint(db_session):
    """Verify unique constraint on Claim.claim_id."""
    p = Provider(provider_id="PRV_UNIQ_01", provider_name="Unique Clinic")
    db_session.add(p)
    db_session.commit()

    c1 = Claim(claim_id="CLM_UNIQ_999", provider_id="PRV_UNIQ_01", reimbursement_amount=100.0)
    db_session.add(c1)
    db_session.commit()

    c2 = Claim(claim_id="CLM_UNIQ_999", provider_id="PRV_UNIQ_01", reimbursement_amount=200.0)
    db_session.add(c2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_relationships_and_foreign_keys(db_session):
    """Verify ORM relationships between Provider, Claim, RiskScore, and Explanation."""
    provider = Provider(provider_id="PRV_REL_01", provider_name="Relationship Hospital")
    beneficiary = Beneficiary(beneficiary_id="BENE_REL_01", renal_disease=False, chronic_conditions_count=2)
    db_session.add_all([provider, beneficiary])
    db_session.commit()

    claim = Claim(
        claim_id="CLM_REL_01",
        provider_id=provider.provider_id,
        beneficiary_id=beneficiary.beneficiary_id,
        reimbursement_amount=5000.0,
        deductible_amount=500.0,
    )
    db_session.add(claim)
    db_session.commit()

    # Create risk score linked to provider
    risk_score = ProviderRiskScore(
        provider_id=provider.provider_id,
        risk_score=0.88,
        raw_score=0.85,
        risk_tier="HIGH",
        threshold_low=0.22,
        threshold_high=0.70,
    )
    db_session.add(risk_score)
    db_session.commit()

    # Create explanation linked to risk score
    explanation = Explanation(
        provider_risk_score_id=risk_score.id,
        explanation_title="AI Explanation \u2013 Why This Claim Received This Risk Score",
        explanation_text="High risk due to reimbursement spikes.",
        feature_contributions=[{"feature": "monthly_claim_count_mean", "shap_value": 0.24}],
    )
    db_session.add(explanation)
    db_session.commit()

    # Check relationships
    refreshed_p = db_session.query(Provider).filter_by(provider_id="PRV_REL_01").first()
    assert len(refreshed_p.claims) == 1
    assert refreshed_p.claims[0].claim_id == "CLM_REL_01"
    assert len(refreshed_p.risk_scores) == 1
    assert refreshed_p.risk_scores[0].risk_tier == "HIGH"
    assert len(refreshed_p.risk_scores[0].explanations) == 1
    assert refreshed_p.risk_scores[0].explanations[0].feature_contributions[0]["shap_value"] == 0.24


def test_transaction_rollback_preserves_state(db_session):
    """Verify that failed transactions roll back cleanly without partial persistence."""
    initial_users = db_session.query(User).count()

    try:
        u_valid = User(email="valid_trans@healthguard.ai", name="Valid", password_hash="pw")
        db_session.add(u_valid)
        db_session.flush()

        # Trigger violation
        u_invalid = User(email=None, name="Invalid", password_hash="pw")  # email is not nullable
        db_session.add(u_invalid)
        db_session.commit()
    except Exception:
        db_session.rollback()

    assert db_session.query(User).count() == initial_users
