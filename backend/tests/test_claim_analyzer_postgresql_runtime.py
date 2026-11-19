"""
Test Suite: Claim Analyzer PostgreSQL Runtime & CMS Isolation
=============================================================
Validates:
1. CMS Dataset Runtime Isolation: Live claim analysis never reads CMS parquet or CSV files.
   (predictor._cms_runtime_access_count == 0)
2. Four Core Real-Time Scenarios:
   - Scenario 1: Existing Provider + Existing Patient
   - Scenario 2: Existing Provider + New Patient
   - Scenario 3: New Provider + Existing Patient
   - Scenario 4: New Provider + New Patient
3. Rule-based anomaly evaluation statuses (PASSED, FLAGGED, UNAVAILABLE, NOT_APPLICABLE).
   - Repeat Beneficiary check is UNAVAILABLE for new providers with 0 prior claims.
   - Post-Death Service Date is UNAVAILABLE when DOD is not recorded.
4. Empty historical charts handling for new providers without prior claims in PostgreSQL:
   - has_historical_records: False, months: []
5. Peer benchmark decoupling:
   - Displays "Peer benchmark unavailable" instead of offline CMS training quantiles.
6. Data Availability and Completeness metadata audit.
"""

import pytest
import pandas as pd
from unittest.mock import patch
from datetime import datetime, date

from backend.app.db.database import SessionLocal
from backend.app.db.models import Provider, Beneficiary, Claim, ProviderRiskScore, ClaimAnalysis
from backend.app.db.schemas import SingleClaimRequest
from backend.app.services.claim_analysis_pipeline import claim_analysis_pipeline
from backend.app.services.prediction_service import prediction_service


@pytest.fixture(scope="module")
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_cms_runtime_isolation_access_count(db_session):
    """
    Verify that live claim inference does NOT access CMS reference parquet or CSV files.
    The predictor._cms_runtime_access_count must remain strictly 0.
    """
    predictor = prediction_service.predictor
    initial_cms_count = predictor._cms_runtime_access_count

    # Execute a single claim analysis for a provider
    payload = SingleClaimRequest(
        ClaimID="CLM_TEST_CMS_ISO_001",
        ProviderID="PRV_TEST_ISO_001",
        BeneID="BENE_TEST_ISO_001",
        InscClaimAmtReimbursed=12000.0,
        ClaimStartDt="2024-05-01",
        ClaimEndDt="2024-05-03",
        ClaimType="Outpatient",
        Currency="INR",
        AttendingPhysician="PHY_TEST_001",
        ClmDiagnosisCode_1="4019",
    )

    try:
        res = claim_analysis_pipeline.analyze_single_claim(payload=payload, db=db_session)
        assert res is not None
        assert res.claim_id == "CLM_TEST_CMS_ISO_001"
        assert predictor._cms_runtime_access_count == initial_cms_count, (
            f"CMS runtime access occurred during live claim analysis! Count grew from {initial_cms_count} to {predictor._cms_runtime_access_count}"
        )
    finally:
        # Clean up inserted test record to keep test pure
        db_session.query(ClaimAnalysis).filter(ClaimAnalysis.claim_id == "CLM_TEST_CMS_ISO_001").delete()
        db_session.query(Claim).filter(Claim.claim_id == "CLM_TEST_CMS_ISO_001").delete()
        db_session.commit()


def test_scenario_1_existing_provider_existing_patient(db_session):
    """
    Scenario 1: Provider and Patient both have prior records in PostgreSQL.
    - provider_history.status == "RECORDED", prior_claim_count > 0
    - patient_history.status == "RECORDED", prior_claim_count > 0
    - monthly_trends.has_historical_records == True
    """
    prov_id = "PRV_SCENARIO_1_EX"
    bene_id = "BENE_SCENARIO_1_EX"

    # Setup provider, beneficiary, and 2 prior claims in PostgreSQL
    prov = db_session.query(Provider).filter(Provider.provider_id == prov_id).first()
    if not prov:
        prov = Provider(provider_id=prov_id)
        db_session.add(prov)

    bene = db_session.query(Beneficiary).filter(Beneficiary.beneficiary_id == bene_id).first()
    if not bene:
        bene = Beneficiary(beneficiary_id=bene_id, chronic_conditions_count=2, renal_disease=1)
        db_session.add(bene)
    db_session.commit()

    prior_c1 = Claim(
        claim_id="CLM_SCENARIO_1_PRIOR_1",
        provider_id=prov_id,
        beneficiary_id=bene_id,
        claim_type="Outpatient",
        claim_date=datetime(2024, 1, 10),
        claim_end_date=datetime(2024, 1, 10),
        reimbursement_amount=250.0,
        attending_physician="PHY_101",
        diagnosis_codes=["4019"],
    )
    prior_c2 = Claim(
        claim_id="CLM_SCENARIO_1_PRIOR_2",
        provider_id=prov_id,
        beneficiary_id=bene_id,
        claim_type="Outpatient",
        claim_date=datetime(2024, 2, 15),
        claim_end_date=datetime(2024, 2, 15),
        reimbursement_amount=300.0,
        attending_physician="PHY_101",
        diagnosis_codes=["4019"],
    )
    db_session.add_all([prior_c1, prior_c2])
    db_session.commit()

    try:
        payload = SingleClaimRequest(
            ClaimID="CLM_SCENARIO_1_CURRENT",
            ProviderID=prov_id,
            BeneID=bene_id,
            InscClaimAmtReimbursed=45000.0,
            ClaimStartDt="2024-03-01",
            ClaimEndDt="2024-03-01",
            ClaimType="Outpatient",
            Currency="INR",
            AttendingPhysician="PHY_101",
            ClmDiagnosisCode_1="4019",
        )
        res = claim_analysis_pipeline.analyze_single_claim(payload=payload, db=db_session)

        # Assert Data Availability
        da = res.data_availability
        assert da is not None
        assert da["provider_history"]["status"] == "RECORDED"
        assert da["provider_history"]["available"] is True
        assert da["provider_history"]["prior_claim_count"] >= 2
        assert da["patient_history"]["status"] == "RECORDED"
        assert da["patient_history"]["available"] is True

        # Assert Trends
        assert res.monthly_trends is not None
        assert res.monthly_trends["has_historical_records"] is True
        assert len(res.monthly_trends["months"]) > 0

    finally:
        db_session.query(ClaimAnalysis).filter(ClaimAnalysis.claim_id.in_(["CLM_SCENARIO_1_PRIOR_1", "CLM_SCENARIO_1_PRIOR_2", "CLM_SCENARIO_1_CURRENT"])).delete()
        db_session.query(Claim).filter(Claim.claim_id.in_(["CLM_SCENARIO_1_PRIOR_1", "CLM_SCENARIO_1_PRIOR_2", "CLM_SCENARIO_1_CURRENT"])).delete()
        db_session.commit()


def test_scenario_2_existing_provider_new_patient(db_session):
    """
    Scenario 2: Provider exists in DB, but Patient has 0 prior encounters in PostgreSQL.
    - provider_history.status == "RECORDED"
    - patient_history.status == "NEW_PATIENT", prior_claim_count == 0
    """
    prov_id = "PRV_SCENARIO_2_EX"
    bene_id = "BENE_SCENARIO_2_NEW"

    prior_c = Claim(
        claim_id="CLM_SCENARIO_2_PRIOR",
        provider_id=prov_id,
        beneficiary_id="BENE_OTHER_001",
        claim_type="Outpatient",
        claim_date=datetime(2024, 1, 10),
        claim_end_date=datetime(2024, 1, 10),
        reimbursement_amount=150.0,
        attending_physician="PHY_102",
        diagnosis_codes=["4019"],
    )
    db_session.add(prior_c)
    db_session.commit()

    try:
        payload = SingleClaimRequest(
            ClaimID="CLM_SCENARIO_2_CURRENT",
            ProviderID=prov_id,
            BeneID=bene_id,
            InscClaimAmtReimbursed=25000.0,
            ClaimStartDt="2024-03-01",
            ClaimEndDt="2024-03-01",
            ClaimType="Outpatient",
            Currency="INR",
            AttendingPhysician="PHY_102",
            ClmDiagnosisCode_1="4019",
        )
        res = claim_analysis_pipeline.analyze_single_claim(payload=payload, db=db_session)

        da = res.data_availability
        assert da is not None
        assert da["provider_history"]["status"] == "RECORDED"
        assert da["patient_history"]["status"] == "NEW_PATIENT"
        assert da["patient_history"]["available"] is False
        assert da["patient_history"]["prior_claim_count"] == 0

    finally:
        db_session.query(ClaimAnalysis).filter(ClaimAnalysis.claim_id.in_(["CLM_SCENARIO_2_PRIOR", "CLM_SCENARIO_2_CURRENT"])).delete()
        db_session.query(Claim).filter(Claim.claim_id.in_(["CLM_SCENARIO_2_PRIOR", "CLM_SCENARIO_2_CURRENT"])).delete()
        db_session.commit()


def test_scenario_3_new_provider_existing_patient(db_session):
    """
    Scenario 3: Provider has 0 prior claims in DB, but Patient has existing history.
    - provider_history.status == "NEW_PROVIDER", prior_claim_count == 0
    - patient_history.status == "RECORDED", prior_claim_count > 0
    - monthly_trends.has_historical_records == False, months: []
    - Repeat Beneficiary check is UNAVAILABLE
    """
    prov_id = "PRV_SCENARIO_3_NEW"
    bene_id = "BENE_SCENARIO_3_EX"

    bene_prior_claim = Claim(
        claim_id="CLM_SCENARIO_3_PATIENT_PRIOR",
        provider_id="PRV_OTHER_PROVIDER",
        beneficiary_id=bene_id,
        claim_type="Outpatient",
        claim_date=datetime(2024, 1, 10),
        claim_end_date=datetime(2024, 1, 10),
        reimbursement_amount=500.0,
        attending_physician="PHY_OTHER",
        diagnosis_codes=["4019"],
    )
    db_session.add(bene_prior_claim)
    db_session.commit()

    try:
        payload = SingleClaimRequest(
            ClaimID="CLM_SCENARIO_3_CURRENT",
            ProviderID=prov_id,
            BeneID=bene_id,
            InscClaimAmtReimbursed=18000.0,
            ClaimStartDt="2024-03-01",
            ClaimEndDt="2024-03-01",
            ClaimType="Outpatient",
            Currency="INR",
            AttendingPhysician="PHY_103",
            ClmDiagnosisCode_1="4019",
        )
        res = claim_analysis_pipeline.analyze_single_claim(payload=payload, db=db_session)

        da = res.data_availability
        assert da is not None
        assert da["provider_history"]["status"] == "NEW_PROVIDER"
        assert da["provider_history"]["available"] is False
        assert da["provider_history"]["prior_claim_count"] == 0
        assert da["patient_history"]["status"] == "RECORDED"
        assert da["patient_history"]["available"] is True

        # Check monthly trends empty state
        assert res.monthly_trends is not None
        assert res.monthly_trends["has_historical_records"] is False
        assert res.monthly_trends["months"] == []
        assert "No historical claims available" in res.monthly_trends["message"]

        # Check evaluated checks: Repeat Beneficiary must be UNAVAILABLE
        eval_checks = res.claim_anomaly_findings[0]["evaluated_checks"]
        repeat_check = next((c for c in eval_checks if c["rule_name"] == "Repeat Beneficiary Frequency"), None)
        assert repeat_check is not None
        assert repeat_check["status"] == "UNAVAILABLE"
        assert "Provider history insufficient" in repeat_check["details"]

    finally:
        db_session.query(ClaimAnalysis).filter(ClaimAnalysis.claim_id.in_(["CLM_SCENARIO_3_PATIENT_PRIOR", "CLM_SCENARIO_3_CURRENT"])).delete()
        db_session.query(Claim).filter(Claim.claim_id.in_(["CLM_SCENARIO_3_PATIENT_PRIOR", "CLM_SCENARIO_3_CURRENT"])).delete()
        db_session.commit()


def test_scenario_4_new_provider_new_patient(db_session):
    """
    Scenario 4: Provider and Patient are completely new to HealthGuard AI.
    - provider_history.status == "NEW_PROVIDER", prior_claim_count == 0
    - patient_history.status == "NEW_PATIENT", prior_claim_count == 0
    - peer_benchmark.status == "INSUFFICIENT_DATA"
    - Model produces finite probability without NaN or crash.
    """
    prov_id = "PRV_SCENARIO_4_NEW"
    bene_id = "BENE_SCENARIO_4_NEW"

    try:
        payload = SingleClaimRequest(
            ClaimID="CLM_SCENARIO_4_CURRENT",
            ProviderID=prov_id,
            BeneID=bene_id,
            InscClaimAmtReimbursed=30000.0,
            ClaimStartDt="2024-04-10",
            ClaimEndDt="2024-04-10",
            ClaimType="Outpatient",
            Currency="INR",
            AttendingPhysician="PHY_104",
            ClmDiagnosisCode_1="4019",
        )
        res = claim_analysis_pipeline.analyze_single_claim(payload=payload, db=db_session)

        da = res.data_availability
        assert da is not None
        assert da["provider_history"]["status"] == "NEW_PROVIDER"
        assert da["patient_history"]["status"] == "NEW_PATIENT"
        assert da["peer_benchmark"]["status"] == "INSUFFICIENT_DATA"

        # Model output validity
        assert 0.0 <= res.provider_risk_score <= 1.0
        assert res.provider_risk_tier in ["LOW", "MEDIUM", "HIGH"]

        # Peer benchmark decoupling in SHAP explanations
        inv_rep = res.investigator_report
        assert inv_rep is not None
        for reason in inv_rep.get("main_reasons", []):
            assert reason["peer_reference_formatted"] in ["Peer benchmark unavailable", "Peer benchmark unavailable — insufficient application data"]
            assert reason["comparison_display"] in ["Peer benchmark unavailable", "Peer benchmark unavailable — insufficient application data"]
            assert "peer reference benchmark unavailable for live application data" in reason["finding"]

    finally:
        db_session.query(ClaimAnalysis).filter(ClaimAnalysis.claim_id == "CLM_SCENARIO_4_CURRENT").delete()
        db_session.query(Claim).filter(Claim.claim_id == "CLM_SCENARIO_4_CURRENT").delete()
        db_session.commit()


def test_post_death_check_evaluation_statuses(db_session):
    """
    Verify Post-Death Service Date rule evaluation:
    1. Beneficiary date of death not recorded -> status is UNAVAILABLE
    2. Beneficiary is recorded as deceased and claim date is after death -> status is FLAGGED
    3. Beneficiary is recorded as deceased and claim date is before death -> status is PASSED
    """
    prov_id = "PRV_DEATH_TEST"

    # Case 1: Beneficiary without DOD
    bene_alive = db_session.query(Beneficiary).filter(Beneficiary.beneficiary_id == "BENE_ALIVE_001").first()
    if not bene_alive:
        bene_alive = Beneficiary(beneficiary_id="BENE_ALIVE_001", is_deceased=False, date_of_death=None)
        db_session.add(bene_alive)

    # Case 2 & 3: Beneficiary with DOD = 2024-01-01
    bene_dead = db_session.query(Beneficiary).filter(Beneficiary.beneficiary_id == "BENE_DEAD_001").first()
    if not bene_dead:
        bene_dead = Beneficiary(beneficiary_id="BENE_DEAD_001", is_deceased=True, date_of_death=date(2024, 1, 1))
        db_session.add(bene_dead)
    else:
        bene_dead.is_deceased = True
        bene_dead.date_of_death = date(2024, 1, 1)
    db_session.commit()

    try:
        # 1. Alive beneficiary -> UNAVAILABLE
        req1 = SingleClaimRequest(
            ClaimID="CLM_DEATH_TEST_1",
            ProviderID=prov_id,
            BeneID="BENE_ALIVE_001",
            InscClaimAmtReimbursed=1000.0,
            ClaimStartDt="2024-02-01",
            ClaimEndDt="2024-02-01",
            ClaimType="Outpatient",
            Currency="USD",
            AttendingPhysician="PHY_D",
            ClmDiagnosisCode_1="4019",
        )
        res1 = claim_analysis_pipeline.analyze_single_claim(payload=req1, db=db_session)
        checks1 = res1.claim_anomaly_findings[0]["evaluated_checks"]
        c1 = next(c for c in checks1 if c["rule_name"] == "Post-Death Service Date")
        assert c1["status"] == "UNAVAILABLE"

        # 2. Deceased beneficiary, claim on 2024-03-01 (AFTER death) -> FLAGGED
        req2 = SingleClaimRequest(
            ClaimID="CLM_DEATH_TEST_2",
            ProviderID=prov_id,
            BeneID="BENE_DEAD_001",
            InscClaimAmtReimbursed=1000.0,
            ClaimStartDt="2024-03-01",
            ClaimEndDt="2024-03-01",
            ClaimType="Outpatient",
            Currency="USD",
            AttendingPhysician="PHY_D",
            ClmDiagnosisCode_1="4019",
        )
        res2 = claim_analysis_pipeline.analyze_single_claim(payload=req2, db=db_session)
        checks2 = res2.claim_anomaly_findings[0]["evaluated_checks"]
        c2 = next(c for c in checks2 if c["rule_name"] == "Post-Death Service Date")
        assert c2["status"] == "FLAGGED"
        assert "POST_DEATH_CLAIM" in res2.anomaly_flags

        # 3. Deceased beneficiary, claim on 2023-12-01 (BEFORE death) -> PASSED
        req3 = SingleClaimRequest(
            ClaimID="CLM_DEATH_TEST_3",
            ProviderID=prov_id,
            BeneID="BENE_DEAD_001",
            InscClaimAmtReimbursed=1000.0,
            ClaimStartDt="2023-12-01",
            ClaimEndDt="2023-12-01",
            ClaimType="Outpatient",
            Currency="USD",
            AttendingPhysician="PHY_D",
            ClmDiagnosisCode_1="4019",
        )
        res3 = claim_analysis_pipeline.analyze_single_claim(payload=req3, db=db_session)
        checks3 = res3.claim_anomaly_findings[0]["evaluated_checks"]
        c3 = next(c for c in checks3 if c["rule_name"] == "Post-Death Service Date")
        assert c3["status"] == "PASSED"
        assert "POST_DEATH_CLAIM" not in res3.anomaly_flags

    finally:
        db_session.query(ClaimAnalysis).filter(ClaimAnalysis.claim_id.in_(["CLM_DEATH_TEST_1", "CLM_DEATH_TEST_2", "CLM_DEATH_TEST_3"])).delete()
        db_session.query(Claim).filter(Claim.claim_id.in_(["CLM_DEATH_TEST_1", "CLM_DEATH_TEST_2", "CLM_DEATH_TEST_3"])).delete()
        db_session.query(Beneficiary).filter(Beneficiary.beneficiary_id.in_(["BENE_ALIVE_001", "BENE_DEAD_001"])).delete()
        db_session.commit()
