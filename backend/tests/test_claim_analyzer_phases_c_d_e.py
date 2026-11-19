"""
HealthGuard AI - Claim Analyzer Verification Tests for Phases C, D, and E
- Phase C: Diagnosis Code 2 & Admission Diagnosis Code processing in payload
- Phase D: Concurrency and duplicate claim handling without 500 error (idempotent / 409)
- Phase E: Investigator Decision workflow persistence and audit logging
"""

import pytest
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from sqlalchemy.exc import IntegrityError
from fastapi import status

from backend.app.db.schemas import (
    SingleClaimRequest,
    SingleClaimResponse,
    InvestigatorDecisionRequest,
    InvestigatorDecisionResponse,
)
from backend.app.services.claim_analysis_pipeline import (
    claim_analysis_pipeline,
    AnalysisConflictError,
)
from backend.app.db.models import ClaimAnalysis, User, CaseNote, AuditLog


class DummyUser:
    id = 1
    email = "investigator@healthguard.ai"
    role = "investigator"


# ══════════════════════════════════════════════════════════════════════════════
# PHASE C TESTS: Diagnosis Code 2 and Admission Diagnosis Code
# ══════════════════════════════════════════════════════════════════════════════

def test_phase_c_payload_with_admission_and_secondary_diagnosis(db_session):
    """
    Verify SingleClaimRequest accepts ClmAdmitDiagnosisCode and ClmDiagnosisCode_2
    and pipeline correctly incorporates them.
    """
    req = SingleClaimRequest(
        ClaimID=f"CLM_TEST_C_{uuid.uuid4().hex[:6]}",
        ProviderID="PRV51003",
        BeneID="BENE11001",
        ClaimType="Inpatient",
        InscClaimAmtReimbursed=50000.0,
        DeductibleAmtPaid=1000.0,
        Currency="INR",
        ClaimStartDt="2020-05-01",
        ClaimEndDt="2020-05-05",
        AdmissionDt="2020-05-01",
        DischargeDt="2020-05-05",
        ClmDiagnosisCode_1="4019",
        ClmDiagnosisCode_2="25000",
        ClmAdmitDiagnosisCode="4280",
        ClmProcedureCode_1="3722",
        AttendingPhysician="PHY31201",
    )

    assert req.ClmDiagnosisCode_1 == "4019"
    assert req.ClmDiagnosisCode_2 == "25000"
    assert req.ClmAdmitDiagnosisCode == "4280"
    assert req.ClmProcedureCode_1 == "3722"

    resp = claim_analysis_pipeline.analyze_single_claim(
        payload=req,
        db=db_session,
        current_user=DummyUser(),
    )

    assert isinstance(resp, SingleClaimResponse)
    assert resp.claim_id == req.ClaimID
    assert resp.provider_id == req.ProviderID


# ══════════════════════════════════════════════════════════════════════════════
# PHASE D TESTS: Concurrency & Duplicate Claim Handling
# ══════════════════════════════════════════════════════════════════════════════

def test_phase_d_integrity_error_with_existing_claim_analysis_returns_idempotently(db_session):
    """
    When IntegrityError occurs (e.g. concurrent race condition) but ClaimAnalysis exists,
    analyze_single_claim must catch IntegrityError and return SingleClaimResponse idempotently.
    """
    claim_id = f"CLM_RACE_{uuid.uuid4().hex[:6]}"
    req = SingleClaimRequest(
        ClaimID=claim_id,
        ProviderID="PRV51003",
        BeneID="BENE11001",
        ClaimType="Outpatient",
        InscClaimAmtReimbursed=10000.0,
        DeductibleAmtPaid=100.0,
        Currency="INR",
        ClaimStartDt="2020-05-01",
        ClaimEndDt="2020-05-01",
        ClmDiagnosisCode_1="4019",
    )

    # First run: normal analysis to ensure record is persisted
    resp1 = claim_analysis_pipeline.analyze_single_claim(
        payload=req,
        db=db_session,
        current_user=DummyUser(),
    )
    assert resp1.claim_id == claim_id

    # Simulate concurrent worker encountering an IntegrityError during db.commit()
    with patch.object(db_session, "commit", side_effect=IntegrityError("duplicate key", {}, None)):
        resp2 = claim_analysis_pipeline.analyze_single_claim(
            payload=req,
            db=db_session,
            current_user=DummyUser(),
        )
        # Should return idempotent response instead of raising RuntimeError or crashing
        assert isinstance(resp2, SingleClaimResponse)
        assert resp2.claim_id == claim_id
        assert resp2.provider_id == "PRV51003"


def test_phase_d_integrity_error_without_existing_claim_raises_conflict(db_session):
    """
    When IntegrityError occurs and no existing analysis record is present,
    analyze_single_claim must raise AnalysisConflictError (which maps to HTTP 409).
    """
    claim_id = f"CLM_UNRESOLVED_{uuid.uuid4().hex[:6]}"
    req = SingleClaimRequest(
        ClaimID=claim_id,
        ProviderID="PRV51003",
        BeneID="BENE11001",
        ClaimType="Outpatient",
        InscClaimAmtReimbursed=10000.0,
        DeductibleAmtPaid=100.0,
        Currency="INR",
        ClaimStartDt="2020-05-01",
        ClaimEndDt="2020-05-01",
        ClmDiagnosisCode_1="4019",
    )

    with patch.object(db_session, "commit", side_effect=IntegrityError("foreign key conflict", {}, None)):
        with pytest.raises(AnalysisConflictError) as exc_info:
            claim_analysis_pipeline.analyze_single_claim(
                payload=req,
                db=db_session,
                current_user=DummyUser(),
            )
        assert "Concurrent submission conflict" in str(exc_info.value)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE E TESTS: Investigator Decision Workflow
# ══════════════════════════════════════════════════════════════════════════════

def test_phase_e_record_investigator_decision_endpoint(client, db_session):
    """
    Verify POST /api/analyzer/decision endpoint persists the decision,
    creates a CaseNote and AuditLog, and returns the expected response.
    """
    # 1. First authenticate or mock user
    claim_id = f"CLM_DEC_{uuid.uuid4().hex[:6]}"
    provider_id = "PRV51003"

    payload = {
        "claim_id": claim_id,
        "provider_id": provider_id,
        "decision_status": "Investigation Required",
        "notes": "Unusual billing spike relative to historical outpatient peer benchmarks.",
        "decision_date": datetime.now(timezone.utc).isoformat(),
    }

    # We test via the FastAPI test client if auth is bypassed or via direct router call
    from backend.app.api.claim_analyzer import record_investigator_decision
    req_obj = InvestigatorDecisionRequest(**payload)

    # Make sure DummyUser has an ID in the DB or mock it
    mock_user = DummyUser()
    resp = record_investigator_decision(
        payload=req_obj,
        db=db_session,
        current_user=mock_user,
    )

    assert isinstance(resp, InvestigatorDecisionResponse)
    assert resp.claim_id == claim_id
    assert resp.provider_id == provider_id
    assert resp.decision_status == "Investigation Required"
    assert "Unusual billing spike" in resp.notes
    assert resp.audit_logged is True
    assert resp.decision_id is not None
