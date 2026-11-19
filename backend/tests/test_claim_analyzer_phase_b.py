"""
Tests for Phase B: Provider-History Performance and Duplicate Query Optimization
"""

import pytest
import pandas as pd
from datetime import datetime
from backend.app.services.claim_analysis_pipeline import claim_analysis_pipeline
from backend.app.services.prediction_service import prediction_service
from backend.app.db.database import SessionLocal
from backend.app.db.models import Claim, Provider, Beneficiary


@pytest.fixture
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_cms_only_history(db_session):
    # Provider history comes from PostgreSQL database records
    c = Claim(
        claim_id="CLM_TEST_PRV51003",
        provider_id="PRV51003",
        claim_type="Outpatient",
        claim_date=datetime(2024, 1, 1),
        reimbursement_amount=100.0,
    )
    db_session.add(c)
    db_session.commit()
    hist = claim_analysis_pipeline.fetch_provider_historical_claims("PRV51003", db=db_session)
    assert not hist.empty
    assert "ClaimID" in hist.columns
    assert "ProviderID" in hist.columns
    # Ensure all claims belong to PRV51003
    assert (hist["ProviderID"] == "PRV51003").all()


def test_new_provider_no_history():
    hist = claim_analysis_pipeline.fetch_provider_historical_claims("PRV_NON_EXISTENT_99999", db=None)
    assert hist.empty


def test_case_variation_provider_id(db_session):
    c = Claim(
        claim_id="CLM_TEST_CASE_01",
        provider_id="PRV51003",
        claim_type="Outpatient",
        claim_date=datetime(2024, 1, 1),
        reimbursement_amount=100.0,
    )
    db_session.add(c)
    db_session.commit()
    hist_upper = claim_analysis_pipeline.fetch_provider_historical_claims("PRV51003", db=db_session)
    hist_lower = claim_analysis_pipeline.fetch_provider_historical_claims("prv51003", db=db_session)
    assert len(hist_upper) == len(hist_lower)
    assert sorted(hist_upper["ClaimID"].tolist()) == sorted(hist_lower["ClaimID"].tolist())


def test_postgresql_priority_and_deduplication(db_session):
    # Insert or query a provider in PostgreSQL
    prv_id = "PRV_PHASE_B_TEST"
    existing_prv = db_session.query(Provider).filter(Provider.provider_id == prv_id).first()
    if not existing_prv:
        prv = Provider(provider_id=prv_id, provider_name="Phase B Clinic")
        db_session.add(prv)
        db_session.flush()

    # Add 2 claims in PostgreSQL
    c1_id = "CLM_PB_001"
    c2_id = "CLM_PB_002"
    existing_c1 = db_session.query(Claim).filter(Claim.claim_id == c1_id).first()
    if not existing_c1:
        c1 = Claim(
            claim_id=c1_id,
            provider_id=prv_id,
            reimbursement_amount=250.0,
            currency="USD",
        )
        db_session.add(c1)
    existing_c2 = db_session.query(Claim).filter(Claim.claim_id == c2_id).first()
    if not existing_c2:
        c2 = Claim(
            claim_id=c2_id,
            provider_id=prv_id,
            reimbursement_amount=500.0,
            currency="USD",
        )
        db_session.add(c2)
    db_session.commit()

    # Fetch provider history using our authoritative loader
    hist = claim_analysis_pipeline.fetch_provider_historical_claims(prv_id, db=db_session)
    assert len(hist) >= 2
    cids = hist["ClaimID"].tolist()
    assert c1_id in cids
    assert c2_id in cids
    # Ensure no duplicates
    assert len(cids) == len(set(cids))


def test_cache_speedup_and_consistency():
    # Calling fetch_provider_historical_claims multiple times should return identical data from cache
    hist_1 = claim_analysis_pipeline.fetch_provider_historical_claims("PRV51003", db=None)
    hist_2 = claim_analysis_pipeline.fetch_provider_historical_claims("PRV51003", db=None)

    assert len(hist_1) == len(hist_2)
    assert hist_1["ClaimID"].tolist() == hist_2["ClaimID"].tolist()
    assert hist_1["InscClaimAmtReimbursed"].sum() == hist_2["InscClaimAmtReimbursed"].sum()
