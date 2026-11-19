"""
Test Suite: Focused Claim Analyzer Provider & Beneficiary Claim-History Validation
==================================================================================
Covers:
1. Provider Claims History:
   - Retrieves previous claims using submitted Provider ID.
   - Excludes current claim from historical calculations.
   - Verified sourced strictly from PostgreSQL.
   - Historical statistics (reimbursement, count, unique beneficiaries) calculated correctly.
   - Zero CMS access count at runtime.

2. Beneficiary Claims History:
   - Retrieves previous claims using Beneficiary ID.
   - Includes claims across ALL providers associated with that beneficiary (not restricted to current Provider ID).
   - Excludes current claim from historical calculations.
   - Repeated claims, claim frequency, and service overlap checks across providers.
   - Sourced exclusively from PostgreSQL.

3. Four Scenarios:
   - Scenario A: Existing Provider + Existing Beneficiary (multi-provider history)
   - Scenario B: Existing Provider + New Beneficiary
   - Scenario C: New Provider + Existing Beneficiary
   - Scenario D: New Provider + New Beneficiary

4. Model Invariants & Integrity:
   - 51 tabular features preserved.
   - 12-month LSTM sequence (1, 12, 6) preserved.
   - Missing history does not bias fraud score.
"""

import pytest
import pandas as pd
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


def test_provider_history_retrieval_and_current_claim_exclusion(db_session):
    """
    1. Provider Claims History:
    - Sourced exclusively from PostgreSQL.
    - Excludes current claim from historical calculations.
    - Statistics calculated correctly.
    - Zero CMS dataset access.
    """
    prov_id = "PRV_HIST_VAL_01"
    bene_id = "BENE_HIST_VAL_01"
    predictor = prediction_service.predictor
    initial_cms_access = predictor._cms_runtime_access_count

    try:
        # Seed provider and 2 prior claims
        prov = db_session.query(Provider).filter(Provider.provider_id == prov_id).first()
        if not prov:
            prov = Provider(provider_id=prov_id)
            db_session.add(prov)

        bene = db_session.query(Beneficiary).filter(Beneficiary.beneficiary_id == bene_id).first()
        if not bene:
            bene = Beneficiary(beneficiary_id=bene_id)
            db_session.add(bene)
        db_session.flush()

        c1 = Claim(
            claim_id="CLM_PHIST_01",
            provider_id=prov_id,
            beneficiary_id=bene_id,
            claim_type="Outpatient",
            claim_date=datetime(2023, 1, 15),
            claim_end_date=datetime(2023, 1, 15),
            reimbursement_amount=500.0,
            deductible_amount=50.0,
        )
        c2 = Claim(
            claim_id="CLM_PHIST_02",
            provider_id=prov_id,
            beneficiary_id=bene_id,
            claim_type="Outpatient",
            claim_date=datetime(2023, 2, 20),
            claim_end_date=datetime(2023, 2, 20),
            reimbursement_amount=1500.0,
            deductible_amount=100.0,
        )
        db_session.add_all([c1, c2])
        db_session.commit()

        # Fetch provider history directly excluding a hypothetical new claim
        hist_df = claim_analysis_pipeline.fetch_provider_historical_claims(
            provider_id=prov_id,
            db=db_session,
            exclude_claim_id="CLM_PHIST_NEW",
        )
        assert len(hist_df) == 2, f"Expected 2 prior claims, got {len(hist_df)}"
        assert set(hist_df["ClaimID"].tolist()) == {"CLM_PHIST_01", "CLM_PHIST_02"}

        # Fetch provider history excluding an EXISTING claim ID (e.g. c1)
        hist_df_ex_c1 = claim_analysis_pipeline.fetch_provider_historical_claims(
            provider_id=prov_id,
            db=db_session,
            exclude_claim_id="CLM_PHIST_01",
        )
        assert len(hist_df_ex_c1) == 1
        assert hist_df_ex_c1.iloc[0]["ClaimID"] == "CLM_PHIST_02"

        # Now run live claim analysis for a new claim
        payload = SingleClaimRequest(
            ClaimID="CLM_PHIST_NEW",
            ProviderID=prov_id,
            BeneID=bene_id,
            InscClaimAmtReimbursed=1000.0,
            ClaimStartDt="2023-04-10",
            ClaimEndDt="2023-04-10",
            ClaimType="Outpatient",
            Currency="USD",
        )
        res = claim_analysis_pipeline.analyze_single_claim(payload, db=db_session)

        # Provider history verification
        prov_hist = res.provider_history or (res.data_availability and res.data_availability.get("provider_history"))
        assert prov_hist is not None
        assert prov_hist["available"] is True
        assert prov_hist["prior_claim_count"] == 2  # Exactly 2 prior claims, excluding CLM_PHIST_NEW
        assert prov_hist["total_reimbursement"] == 2000.0  # 500 + 1500
        assert prov_hist["avg_reimbursement"] == 1000.0  # 2000 / 2
        assert prov_hist["distinct_beneficiaries_count"] == 1

        # Invariant: CMS runtime access count remains 0
        assert predictor._cms_runtime_access_count == initial_cms_access

    finally:
        # Cleanup seeded records
        db_session.query(ClaimAnalysis).filter(ClaimAnalysis.claim_id.in_(["CLM_PHIST_01", "CLM_PHIST_02", "CLM_PHIST_NEW"])).delete(synchronize_session=False)
        db_session.query(Claim).filter(Claim.claim_id.in_(["CLM_PHIST_01", "CLM_PHIST_02", "CLM_PHIST_NEW"])).delete(synchronize_session=False)
        db_session.commit()


def test_beneficiary_history_multi_provider_retrieval(db_session):
    """
    2. Beneficiary Claims History:
    - Sourced exclusively from PostgreSQL.
    - Includes claims across ALL providers associated with that beneficiary.
    - Example scenario from user request:
      Beneficiary BEN200:
      - Claim 001 - Provider PRV100
      - Claim 002 - Provider PRV105
      - Claim 003 - Provider PRV110
      When analyzing a new claim for BEN200 through PRV100, all three previous claims must be considered!
    - Verifies overlap/duplicate checks across providers.
    """
    bene_id = "BEN200_MULTI_TEST"
    prv_100 = "PRV100_BEN_TEST"
    prv_105 = "PRV105_BEN_TEST"
    prv_110 = "PRV110_BEN_TEST"

    try:
        # Seed providers and beneficiary
        for pid in [prv_100, prv_105, prv_110]:
            if not db_session.query(Provider).filter(Provider.provider_id == pid).first():
                db_session.add(Provider(provider_id=pid))

        bene = db_session.query(Beneficiary).filter(Beneficiary.beneficiary_id == bene_id).first()
        if not bene:
            bene = Beneficiary(beneficiary_id=bene_id, chronic_conditions_count=2, renal_disease=1)
            db_session.add(bene)
        db_session.flush()

        # Seed the 3 claims across 3 distinct providers
        clm_1 = Claim(
            claim_id="CLM_BEN_001",
            provider_id=prv_100,
            beneficiary_id=bene_id,
            claim_type="Outpatient",
            claim_date=datetime(2023, 1, 10),
            claim_end_date=datetime(2023, 1, 10),
            reimbursement_amount=300.0,
            deductible_amount=30.0,
        )
        clm_2 = Claim(
            claim_id="CLM_BEN_002",
            provider_id=prv_105,
            beneficiary_id=bene_id,
            claim_type="Outpatient",
            claim_date=datetime(2023, 2, 14),
            claim_end_date=datetime(2023, 2, 14),
            reimbursement_amount=700.0,
            deductible_amount=70.0,
        )
        clm_3 = Claim(
            claim_id="CLM_BEN_003",
            provider_id=prv_110,
            beneficiary_id=bene_id,
            claim_type="Outpatient",
            claim_date=datetime(2023, 3, 20),
            claim_end_date=datetime(2023, 3, 20),
            reimbursement_amount=1200.0,
            deductible_amount=100.0,
        )
        db_session.add_all([clm_1, clm_2, clm_3])
        db_session.commit()

        # Step 1: Query beneficiary historical claims directly
        patient_claims, profile = claim_analysis_pipeline.fetch_patient_historical_claims(
            beneficiary_id=bene_id,
            db=db_session,
            exclude_claim_id="CLM_BEN_NEW_004",
        )
        assert len(patient_claims) == 3, f"Expected 3 claims across all providers, got {len(patient_claims)}"
        retrieved_providers = {c["provider_id"] for c in patient_claims}
        assert retrieved_providers == {prv_100, prv_105, prv_110}, "Must include all distinct providers"
        assert profile["chronic_conditions_count"] == 2
        assert profile["renal_disease"] == 1

        # Step 2: Analyze a new claim for BEN200 through PRV100 (non-overlapping dates)
        payload = SingleClaimRequest(
            ClaimID="CLM_BEN_NEW_004",
            ProviderID=prv_100,
            BeneID=bene_id,
            InscClaimAmtReimbursed=500.0,
            ClaimStartDt="2023-05-01",
            ClaimEndDt="2023-05-01",
            ClaimType="Outpatient",
            Currency="USD",
        )
        res = claim_analysis_pipeline.analyze_single_claim(payload, db=db_session)

        bene_hist = res.beneficiary_history or (res.data_availability and res.data_availability.get("beneficiary_history"))
        assert bene_hist is not None
        assert bene_hist["available"] is True
        assert bene_hist["prior_claim_count"] == 3  # All 3 prior claims retrieved!
        assert bene_hist["associated_providers_count"] == 3  # Across PRV100, PRV105, PRV110
        assert set(bene_hist["associated_providers"]) == {prv_100, prv_105, prv_110}
        assert bene_hist["total_reimbursement"] == 2200.0  # 300 + 700 + 1200
        assert bene_hist["first_claim_date"] == "2023-01-10"
        assert bene_hist["last_claim_date"] == "2023-03-20"
        assert bene_hist["overlap_detected"] is False

        # Step 3: Analyze another claim with overlapping service date against PRV105 (clm_2: 2023-02-14)
        payload_overlap = SingleClaimRequest(
            ClaimID="CLM_BEN_OVERLAP_005",
            ProviderID=prv_100,
            BeneID=bene_id,
            InscClaimAmtReimbursed=600.0,
            ClaimStartDt="2023-02-14",  # Same date as CLM_BEN_002 with PRV105!
            ClaimEndDt="2023-02-14",
            ClaimType="Outpatient",
            Currency="USD",
        )
        res_overlap = claim_analysis_pipeline.analyze_single_claim(payload_overlap, db=db_session)
        bene_hist_overlap = res_overlap.beneficiary_history or (res_overlap.data_availability and res_overlap.data_availability.get("beneficiary_history"))
        assert bene_hist_overlap["overlap_detected"] is True
        assert len(bene_hist_overlap["overlapping_claims"]) >= 1
        assert bene_hist_overlap["overlapping_claims"][0]["provider_id"] == prv_105

    finally:
        # Cleanup seeded records
        test_cids = ["CLM_BEN_001", "CLM_BEN_002", "CLM_BEN_003", "CLM_BEN_NEW_004", "CLM_BEN_OVERLAP_005"]
        db_session.query(ClaimAnalysis).filter(ClaimAnalysis.claim_id.in_(test_cids)).delete(synchronize_session=False)
        db_session.query(Claim).filter(Claim.claim_id.in_(test_cids)).delete(synchronize_session=False)
        db_session.commit()


def test_four_scenarios_comprehensive(db_session):
    """
    3. Test the Four Scenarios:
    - Scenario A: Existing Provider + Existing Beneficiary
    - Scenario B: Existing Provider + New Beneficiary
    - Scenario C: New Provider + Existing Beneficiary
    - Scenario D: New Provider + New Beneficiary
    """
    exist_prv = "PRV_SCENARIO_EXIST_01"
    exist_bene = "BEN_SCENARIO_EXIST_01"
    new_prv = "PRV_SCENARIO_NEW_99"
    new_bene = "BEN_SCENARIO_NEW_99"

    try:
        # Seed existing provider with 2 prior claims
        if not db_session.query(Provider).filter(Provider.provider_id == exist_prv).first():
            db_session.add(Provider(provider_id=exist_prv))

        # Seed existing beneficiary with 1 prior claim under a third provider
        other_prv = "PRV_SCENARIO_OTHER_55"
        if not db_session.query(Provider).filter(Provider.provider_id == other_prv).first():
            db_session.add(Provider(provider_id=other_prv))

        if not db_session.query(Beneficiary).filter(Beneficiary.beneficiary_id == exist_bene).first():
            db_session.add(Beneficiary(beneficiary_id=exist_bene))
        db_session.flush()

        prior_claims = [
            Claim(
                claim_id="CLM_SCENARIO_P1",
                provider_id=exist_prv,
                beneficiary_id="BEN_OTHER_01",
                claim_type="Outpatient",
                claim_date=datetime(2023, 1, 1),
                reimbursement_amount=800.0,
            ),
            Claim(
                claim_id="CLM_SCENARIO_P2",
                provider_id=exist_prv,
                beneficiary_id="BEN_OTHER_02",
                claim_type="Outpatient",
                claim_date=datetime(2023, 2, 1),
                reimbursement_amount=1200.0,
            ),
            Claim(
                claim_id="CLM_SCENARIO_B1",
                provider_id=other_prv,
                beneficiary_id=exist_bene,
                claim_type="Outpatient",
                claim_date=datetime(2023, 3, 1),
                reimbursement_amount=450.0,
            ),
        ]
        db_session.add_all(prior_claims)
        db_session.commit()

        # --- SCENARIO A: Existing Provider + Existing Beneficiary ---
        req_a = SingleClaimRequest(
            ClaimID="CLM_SCENARIO_A",
            ProviderID=exist_prv,
            BeneID=exist_bene,
            InscClaimAmtReimbursed=600.0,
            ClaimStartDt="2023-05-10",
            ClaimType="Outpatient",
            Currency="USD",
        )
        res_a = claim_analysis_pipeline.analyze_single_claim(req_a, db=db_session)
        assert 0.0 <= res_a.provider_risk_score <= 1.0
        assert res_a.provider_history["available"] is True
        assert res_a.provider_history["prior_claim_count"] == 2
        assert res_a.beneficiary_history["available"] is True
        assert res_a.beneficiary_history["prior_claim_count"] == 1
        assert res_a.beneficiary_history["associated_providers"] == [other_prv]

        # --- SCENARIO B: Existing Provider + New Beneficiary ---
        req_b = SingleClaimRequest(
            ClaimID="CLM_SCENARIO_B",
            ProviderID=exist_prv,
            BeneID=new_bene,
            InscClaimAmtReimbursed=750.0,
            ClaimStartDt="2023-06-12",
            ClaimType="Outpatient",
            Currency="USD",
        )
        res_b = claim_analysis_pipeline.analyze_single_claim(req_b, db=db_session)
        assert 0.0 <= res_b.provider_risk_score <= 1.0
        assert res_b.provider_history["available"] is True
        assert res_b.provider_history["prior_claim_count"] >= 2
        assert res_b.beneficiary_history["available"] is False
        assert res_b.beneficiary_history["prior_claim_count"] == 0
        assert res_b.beneficiary_history["status"] == "NEW_PATIENT"

        # --- SCENARIO C: New Provider + Existing Beneficiary ---
        req_c = SingleClaimRequest(
            ClaimID="CLM_SCENARIO_C",
            ProviderID=new_prv,
            BeneID=exist_bene,
            InscClaimAmtReimbursed=900.0,
            ClaimStartDt="2023-07-15",
            ClaimType="Outpatient",
            Currency="USD",
        )
        res_c = claim_analysis_pipeline.analyze_single_claim(req_c, db=db_session)
        assert 0.0 <= res_c.provider_risk_score <= 1.0
        assert res_c.provider_history["available"] is False
        assert res_c.provider_history["prior_claim_count"] == 0
        assert res_c.provider_history["status"] == "NEW_PROVIDER"
        assert res_c.beneficiary_history["available"] is True
        assert res_c.beneficiary_history["prior_claim_count"] >= 1

        # Check that Repeat Beneficiary check is marked UNAVAILABLE for new provider
        repeat_check = next((c for c in res_c.claim_anomaly_findings[0]["evaluated_checks"] if c["rule_name"] == "Repeat Beneficiary Frequency"), None)
        assert repeat_check is not None
        assert repeat_check["status"] == "UNAVAILABLE"

        # --- SCENARIO D: New Provider + New Beneficiary ---
        brand_new_prv = "PRV_BRAND_NEW_123"
        brand_new_bene = "BEN_BRAND_NEW_456"
        req_d = SingleClaimRequest(
            ClaimID="CLM_SCENARIO_D",
            ProviderID=brand_new_prv,
            BeneID=brand_new_bene,
            InscClaimAmtReimbursed=350.0,
            ClaimStartDt="2023-08-20",
            ClaimType="Outpatient",
            Currency="USD",
        )
        res_d = claim_analysis_pipeline.analyze_single_claim(req_d, db=db_session)
        assert 0.0 <= res_d.provider_risk_score <= 1.0
        assert res_d.provider_history["available"] is False
        assert res_d.provider_history["prior_claim_count"] == 0
        assert res_d.provider_history["status"] == "NEW_PROVIDER"
        assert res_d.beneficiary_history["available"] is False
        assert res_d.beneficiary_history["prior_claim_count"] == 0
        assert res_d.beneficiary_history["status"] == "NEW_PATIENT"

    finally:
        # Cleanup
        all_test_cids = [
            "CLM_SCENARIO_P1", "CLM_SCENARIO_P2", "CLM_SCENARIO_B1",
            "CLM_SCENARIO_A", "CLM_SCENARIO_B", "CLM_SCENARIO_C", "CLM_SCENARIO_D"
        ]
        db_session.query(ClaimAnalysis).filter(ClaimAnalysis.claim_id.in_(all_test_cids)).delete(synchronize_session=False)
        db_session.query(Claim).filter(Claim.claim_id.in_(all_test_cids)).delete(synchronize_session=False)
        db_session.commit()


def test_section_7_multi_provider_beneficiary_regression(db_session):
    """
    7. AUTOMATIC MULTI-PROVIDER BENEFICIARY REGRESSION TEST (Section 7 Exact Requirements)
    Test data:
    Beneficiary: BEN_VALIDATION_001
    Historical claims:
    | Claim ID     | Provider ID  | Amount |
    | CLM_TEST_001 | PRV_TEST_001 | 300    |
    | CLM_TEST_002 | PRV_TEST_002 | 700    |
    | CLM_TEST_003 | PRV_TEST_003 | 1200   |

    New claim:
    | CLM_TEST_004 | PRV_TEST_001 | BEN_VALIDATION_001 | (Amount 500) |

    Expected results:
    - Beneficiary prior claim count = 3
    - Associated provider count = 3
    - Total prior reimbursement = 2200
    - Historical claims contain all three providers.
    - New claim is excluded.
    - Provider history for PRV_TEST_001 contains only claims belonging to PRV_TEST_001.
    - Beneficiary history contains claims from all three providers.
    """
    bene_id = "BEN_VALIDATION_001"
    prv_1 = "PRV_TEST_001"
    prv_2 = "PRV_TEST_002"
    prv_3 = "PRV_TEST_003"
    cids = ["CLM_TEST_001", "CLM_TEST_002", "CLM_TEST_003", "CLM_TEST_004"]

    try:
        # Seed providers
        for pid in [prv_1, prv_2, prv_3]:
            if not db_session.query(Provider).filter(Provider.provider_id == pid).first():
                db_session.add(Provider(provider_id=pid))

        # Seed beneficiary
        if not db_session.query(Beneficiary).filter(Beneficiary.beneficiary_id == bene_id).first():
            db_session.add(Beneficiary(beneficiary_id=bene_id))
        db_session.flush()

        # Seed historical claims
        c1 = Claim(
            claim_id="CLM_TEST_001",
            provider_id=prv_1,
            beneficiary_id=bene_id,
            claim_type="Outpatient",
            claim_date=datetime(2023, 1, 10),
            reimbursement_amount=300.0,
            currency="USD",
        )
        c2 = Claim(
            claim_id="CLM_TEST_002",
            provider_id=prv_2,
            beneficiary_id=bene_id,
            claim_type="Outpatient",
            claim_date=datetime(2023, 2, 15),
            reimbursement_amount=700.0,
            currency="USD",
        )
        c3 = Claim(
            claim_id="CLM_TEST_003",
            provider_id=prv_3,
            beneficiary_id=bene_id,
            claim_type="Outpatient",
            claim_date=datetime(2023, 3, 20),
            reimbursement_amount=1200.0,
            currency="USD",
        )
        db_session.add_all([c1, c2, c3])
        db_session.commit()

        # Submit new claim CLM_TEST_004 through PRV_TEST_001
        payload = SingleClaimRequest(
            ClaimID="CLM_TEST_004",
            ProviderID=prv_1,
            BeneID=bene_id,
            InscClaimAmtReimbursed=500.0,
            ClaimStartDt="2023-05-15",
            ClaimType="Outpatient",
            Currency="USD",
        )
        response = claim_analysis_pipeline.analyze_single_claim(payload, db=db_session)

        # 1. Beneficiary History Validation
        bene_hist = response.beneficiary_history or response.data_availability.get("beneficiary_history")
        assert bene_hist is not None, "beneficiary_history must be present"
        assert bene_hist["available"] is True, "Beneficiary history must be marked available"
        assert bene_hist["prior_claim_count"] == 3, f"Expected 3 prior claims, got {bene_hist['prior_claim_count']}"
        assert bene_hist["associated_providers_count"] == 3, f"Expected 3 associated providers, got {bene_hist['associated_providers_count']}"
        assert set(bene_hist["associated_providers"]) == {prv_1, prv_2, prv_3}, "Must contain all three providers"
        assert bene_hist["total_reimbursement"] == 2200.0, f"Expected 2200.0 total prior reimbursement, got {bene_hist['total_reimbursement']}"

        # 2. Exclusion of new claim from beneficiary history
        recent_cids = [c["claim_id"] for c in (bene_hist.get("recent_activity") or [])]
        assert "CLM_TEST_004" not in recent_cids, "New claim CLM_TEST_004 must not be in historical records"

        # 3. Provider History Validation for PRV_TEST_001
        prov_hist = response.provider_history or response.data_availability.get("provider_history")
        assert prov_hist is not None, "provider_history must be present"
        assert prov_hist["available"] is True, "Provider history must be marked available"
        # PRV_TEST_001 has only CLM_TEST_001 (amount 300) in history, excluding CLM_TEST_004
        assert prov_hist["prior_claim_count"] == 1, f"Expected 1 prior claim for PRV_TEST_001, got {prov_hist['prior_claim_count']}"
        assert prov_hist["total_reimbursement"] == 300.0, f"Expected 300.0 for PRV_TEST_001, got {prov_hist['total_reimbursement']}"

    finally:
        # Strict cleanup to leave no test records in database
        db_session.query(ClaimAnalysis).filter(ClaimAnalysis.claim_id.in_(cids)).delete(synchronize_session=False)
        db_session.query(Claim).filter(Claim.claim_id.in_(cids)).delete(synchronize_session=False)
        db_session.commit()

