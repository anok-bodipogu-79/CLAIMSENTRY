"""
HealthGuard AI - Phase 4 Providers and Claims API Tests
Verifies provider listing, details, risk scoring, claims pagination, and filtering.
"""

from backend.app.db.models import Provider, Claim, ProviderRiskScore, Explanation


def setup_sample_data(db_session):
    """Seed sample providers, claims, and risk scores."""
    p1 = Provider(provider_id="PRV_TEST_A", provider_name="Alpha Medical Center")
    p2 = Provider(provider_id="PRV_TEST_B", provider_name="Beta Health Clinic")
    db_session.add_all([p1, p2])
    db_session.commit()

    c1 = Claim(
        claim_id="CLM_TEST_001",
        provider_id="PRV_TEST_A",
        claim_type="Inpatient",
        reimbursement_amount=15000.0,
        deductible_amount=1000.0,
        inpatient_los=5,
    )
    c2 = Claim(
        claim_id="CLM_TEST_002",
        provider_id="PRV_TEST_A",
        claim_type="Outpatient",
        reimbursement_amount=250.0,
        deductible_amount=50.0,
    )
    c3 = Claim(
        claim_id="CLM_TEST_003",
        provider_id="PRV_TEST_B",
        claim_type="Outpatient",
        reimbursement_amount=350.0,
    )
    db_session.add_all([c1, c2, c3])
    db_session.commit()

    score = ProviderRiskScore(
        provider_id="PRV_TEST_A",
        risk_score=0.85,
        raw_score=0.82,
        risk_tier="HIGH",
        threshold_low=0.22,
        threshold_high=0.70,
    )
    db_session.add(score)
    db_session.commit()

    expl = Explanation(
        provider_risk_score_id=score.id,
        explanation_title="AI Explanation \u2013 Why This Claim Received This Risk Score",
        explanation_text="High reimbursement total observed.",
        feature_contributions={"top_risk_factors": [{"feature": "reimbursement_total", "value": 15250.0}]},
    )
    db_session.add(expl)
    db_session.commit()


def test_list_providers_pagination_and_search(client, auth_headers, db_session):
    """Verify provider listing supports pagination and text search."""
    setup_sample_data(db_session)

    # All providers
    res = client.get("/api/providers?page=1&page_size=10", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 2
    assert len(data["providers"]) == 2

    # Search filter
    res_search = client.get("/api/providers?search=Alpha", headers=auth_headers)
    assert res_search.status_code == 200
    data_search = res_search.json()
    assert data_search["total"] == 1
    assert data_search["providers"][0]["provider_id"] == "PRV_TEST_A"


def test_get_provider_detail(client, auth_headers, db_session):
    """Verify retrieval of individual provider details and latest risk score."""
    setup_sample_data(db_session)

    res = client.get("/api/providers/PRV_TEST_A", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["provider_id"] == "PRV_TEST_A"
    assert data["total_claims"] == 2
    assert data["risk_score"] == 0.85
    assert data["risk_tier"] == "HIGH"


def test_get_provider_claims(client, auth_headers, db_session):
    """Verify retrieval of claims belonging to a specific provider."""
    setup_sample_data(db_session)

    res = client.get("/api/providers/PRV_TEST_A/claims", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["total"] == 2
    claim_ids = [c["claim_id"] for c in data["claims"]]
    assert "CLM_TEST_001" in claim_ids
    assert "CLM_TEST_002" in claim_ids


def test_get_provider_risk_and_explanation(client, auth_headers, db_session):
    """Verify provider risk history and SHAP explanation endpoint."""
    setup_sample_data(db_session)

    # Risk endpoint
    res_risk = client.get("/api/providers/PRV_TEST_A/risk", headers=auth_headers)
    assert res_risk.status_code == 200
    data_risk = res_risk.json()
    assert data_risk["risk_tier"] == "HIGH"
    assert data_risk["risk_score"] == 0.85

    # Explanation endpoint
    res_expl = client.get("/api/providers/PRV_TEST_A/explanation", headers=auth_headers)
    assert res_expl.status_code == 200
    data_expl = res_expl.json()
    assert "top_risk_factors" in data_expl
    assert len(data_expl["top_risk_factors"]) == 1
    assert data_expl["top_risk_factors"][0]["feature"] == "reimbursement_total"


def test_list_claims_filtering_and_pagination(client, auth_headers, db_session):
    """Verify claims listing with provider filtering and pagination."""
    setup_sample_data(db_session)

    # Filter by provider
    res_prv = client.get("/api/claims?provider_id=PRV_TEST_A", headers=auth_headers)
    assert res_prv.status_code == 200
    data = res_prv.json()
    assert data["total"] == 2
    assert len(data["claims"]) == 2

    # Filter all
    res_all = client.get("/api/claims", headers=auth_headers)
    assert res_all.status_code == 200
    assert res_all.json()["total"] == 3


def test_get_claim_detail_not_found(client, auth_headers):
    """Verify 404 for non-existent claim."""
    res = client.get("/api/claims/CLM_NONEXISTENT", headers=auth_headers)
    assert res.status_code == 404
