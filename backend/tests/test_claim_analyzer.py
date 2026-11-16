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

    saved_analysis = db_session.query(ClaimAnalysis).filter_by(claim_id="CLM_ANALYSIS_999").first()
    assert saved_analysis is not None
    assert saved_analysis.anomaly_score >= 0.0
