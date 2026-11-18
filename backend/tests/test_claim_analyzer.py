"""
HealthGuard AI - Phase 4 Single Claim Analyzer API Tests
Verifies claim analysis endpoint, validation, ML execution, anomaly detection, DB persistence, and disclaimer.
"""

from backend.app.db.models import Provider, Claim, ProviderRiskScore, ClaimAnalysis


def test_single_claim_analyzer_missing_fields(client, auth_headers):
    """Verify validation error when required fields are missing."""
    payload = {
        "ClaimID": "CLM_ERR_001",
        # missing ProviderID, InscClaimAmtReimbursed, ClaimStartDt
    }
    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 422


def test_single_claim_analyzer_full_pipeline(client, auth_headers, db_session):
    """Verify full end-to-end execution of single claim analysis with DB persistence."""
    payload = {
        "ClaimID": "CLM_ANALYSIS_999",
        "ProviderID": "PRV_ANALYSIS_001",
        "BeneID": "BENE_ANALYSIS_001",
        "ClaimType": "Inpatient",
        "ClaimStartDt": "2009-01-15",
        "ClaimEndDt": "2009-01-20",
        "InscClaimAmtReimbursed": 18500.0,
        "DeductibleAmtPaid": 1068.0,
        "ClmDiagnosisCode_1": "4019",
        "AttendingPhysician": "PHY_TEST_101",
    }

    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()

    # Core required response fields
    assert data["claim_id"] == "CLM_ANALYSIS_999"
    assert data["provider_id"] == "PRV_ANALYSIS_001"
    assert "provider_risk_score" in data
    assert 0.0 <= data["provider_risk_score"] <= 1.0
    assert data["provider_risk_tier"] in ["LOW", "MEDIUM", "HIGH"]
    assert "investigation_recommendation" in data
    assert "explanation" in data
    assert "claim_anomaly_findings" in data

    # Statutory disclaimer check
    disclaimer = data.get("disclaimer", "")
    assert "This is a risk assessment to guide investigation. It is not proof of fraud." in disclaimer

    # DB persistence verification
    saved_claim = db_session.query(Claim).filter_by(claim_id="CLM_ANALYSIS_999").first()
    assert saved_claim is not None
    assert saved_claim.provider_id == "PRV_ANALYSIS_001"
    assert saved_claim.reimbursement_amount == 18500.0

    saved_risk = db_session.query(ProviderRiskScore).filter_by(provider_id="PRV_ANALYSIS_001").first()
    assert saved_risk is not None
    assert saved_risk.risk_tier == data["provider_risk_tier"]


def test_inpatient_claim_with_all_additional_fields(client, auth_headers, db_session):
    """Verify Inpatient claim properly persists AdmissionDt, DischargeDt, LOS, Operating/Other physicians and ProcedureCode."""
    payload = {
        "ClaimID": "CLM_INP_FULL_001",
        "ProviderID": "PRV_INP_001",
        "BeneID": "BENE_INP_001",
        "ClaimType": "Inpatient",
        "ClaimStartDt": "2009-05-01",
        "ClaimEndDt": "2009-05-10",
        "AdmissionDt": "2009-05-01",
        "DischargeDt": "2009-05-10",
        "InscClaimAmtReimbursed": 24000.0,
        "DeductibleAmtPaid": 1068.0,
        "ClmDiagnosisCode_1": "4019",
        "ClmProcedureCode_1": "3722",
        "AttendingPhysician": "PHY_ATT_101",
        "OperatingPhysician": "PHY_OP_201",
        "OtherPhysician": "PHY_OTH_301",
    }
    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["claim_id"] == "CLM_INP_FULL_001"

    saved_claim = db_session.query(Claim).filter_by(claim_id="CLM_INP_FULL_001").first()
    assert saved_claim is not None
    assert saved_claim.claim_type == "Inpatient"
    assert saved_claim.admission_date is not None
    assert saved_claim.discharge_date is not None
    assert saved_claim.inpatient_los == 9
    assert saved_claim.operating_physician == "PHY_OP_201"
    assert saved_claim.other_physician == "PHY_OTH_301"
    assert saved_claim.procedure_codes == ["3722"]


def test_outpatient_claim_without_inpatient_fields(client, auth_headers, db_session):
    """Verify Outpatient claims succeed cleanly without admission/discharge dates and set LOS to 0."""
    payload = {
        "ClaimID": "CLM_OUTP_CLEAN_001",
        "ProviderID": "PRV_OUTP_001",
        "BeneID": "BENE_OUTP_001",
        "ClaimType": "Outpatient",
        "ClaimStartDt": "2009-06-01",
        "ClaimEndDt": "2009-06-01",
        "InscClaimAmtReimbursed": 450.0,
        "DeductibleAmtPaid": 50.0,
        "ClmDiagnosisCode_1": "25000",
        "AttendingPhysician": "PHY_ATT_102",
        # AdmissionDt and DischargeDt omitted entirely
    }
    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["claim_id"] == "CLM_OUTP_CLEAN_001"

    saved_claim = db_session.query(Claim).filter_by(claim_id="CLM_OUTP_CLEAN_001").first()
    assert saved_claim is not None
    assert saved_claim.claim_type == "Outpatient"
    assert saved_claim.admission_date is None
    assert saved_claim.discharge_date is None
    assert saved_claim.inpatient_los == 0

    saved_analysis = db_session.query(ClaimAnalysis).filter_by(claim_id="CLM_OUTP_CLEAN_001").first()
    assert saved_analysis is not None
    assert saved_analysis.anomaly_score >= 0.0
