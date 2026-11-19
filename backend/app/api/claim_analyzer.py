"""
HealthGuard AI - Single Claim Analyzer Endpoint
"""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.core.dependencies import get_current_user
from backend.app.db.database import get_db
from backend.app.db.models import User, CaseNote, AuditLog, ClaimAnalysis
from backend.app.db.schemas import (
    SingleClaimRequest,
    SingleClaimResponse,
    InvestigatorDecisionRequest,
    InvestigatorDecisionResponse,
)
from backend.app.services.claim_analysis_pipeline import (
    claim_analysis_pipeline,
    AnalysisValidationError,
    AnalysisConflictError,
)

router = APIRouter(prefix="/analyzer", tags=["Claim Analyzer"])


@router.post("/single", response_model=SingleClaimResponse)
def analyze_single_claim(
    payload: SingleClaimRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Analyze a single healthcare claim against provider history and training peer benchmarks.
    Executes the genuine multi-modal ML ensemble, runs rule-based anomaly detection,
    validates response invariants, and returns explanations and investigation guidance
    via the shared universal analysis pipeline.
    """
    try:
        return claim_analysis_pipeline.analyze_single_claim(
            payload=payload,
            db=db,
            current_user=current_user,
        )
    except AnalysisValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except AnalysisConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Claim analysis failed: {e}",
        )


@router.post("/decision", response_model=InvestigatorDecisionResponse)
def record_investigator_decision(
    payload: InvestigatorDecisionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Persist an investigator decision and case notes for an analyzed claim and provider.
    Updates the ClaimAnalysis status, records a CaseNote, and writes an AuditLog entry.
    """
    allowed_statuses = [
        "Pending Review",
        "Investigation Required",
        "Under Investigation",
        "No Issue Identified",
        "Suspected Irregularity – Escalation Required",
        "Closed",
    ]
    if payload.decision_status not in allowed_statuses:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid decision status '{payload.decision_status}'. Allowed values: {allowed_statuses}",
        )

    now = datetime.now(timezone.utc)
    decision_date_str = payload.decision_date or now.isoformat()

    try:
        # 1. Update latest ClaimAnalysis status if available
        ca = (
            db.query(ClaimAnalysis)
            .filter(ClaimAnalysis.claim_id == payload.claim_id)
            .order_by(ClaimAnalysis.id.desc())
            .first()
        )
        if ca:
            ca.analysis_status = payload.decision_status

        # 2. Add CaseNote record for audit history
        note = CaseNote(
            provider_id=payload.provider_id,
            user_id=current_user.id,
            note_text=f"[{payload.decision_status}] Claim {payload.claim_id}: {payload.notes or 'No notes entered.'}",
            created_at=now,
        )
        db.add(note)
        db.flush()

        # 3. Write compliance AuditLog
        audit = AuditLog(
            user_id=current_user.id,
            action="INVESTIGATOR_DECISION_RECORDED",
            resource_type="CLAIM",
            resource_id=payload.claim_id,
            details={
                "decision_status": payload.decision_status,
                "provider_id": payload.provider_id,
                "claim_id": payload.claim_id,
                "notes": payload.notes,
                "investigator_email": current_user.email,
                "decision_date": decision_date_str,
            },
            timestamp=now,
        )
        db.add(audit)
        db.commit()

        return InvestigatorDecisionResponse(
            decision_id=note.id,
            claim_id=payload.claim_id,
            provider_id=payload.provider_id,
            decision_status=payload.decision_status,
            notes=payload.notes or "",
            decision_date=decision_date_str,
            investigator_email=current_user.email,
            audit_logged=True,
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record investigator decision: {e}",
        )
