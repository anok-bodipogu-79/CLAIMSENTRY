"""
HealthGuard AI - Insurance Claim Exploration Endpoints
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.app.core.config import settings
from backend.app.core.dependencies import get_current_user
from backend.app.db.database import get_db
from backend.app.db.models import User, Claim, ClaimAnalysis, ProviderRiskScore
from backend.app.db.schemas import (
    PaginatedClaimsResponse,
    ClaimListItem,
    ClaimAnalysisDetail,
)

router = APIRouter(prefix="/claims", tags=["Claims"])


@router.get("", response_model=PaginatedClaimsResponse)
def list_claims(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    provider_id: Optional[str] = Query(None, description="Filter by ProviderID"),
    search: Optional[str] = Query(None, description="Search by ClaimID, ProviderID, or BeneficiaryID"),
    claim_type: Optional[str] = Query(None, description="Filter by Claim Type (Inpatient/Outpatient)"),
    start_date: Optional[str] = Query(None, description="Filter claims from start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Filter claims up to end date (YYYY-MM-DD)"),
    has_anomaly: Optional[bool] = Query(None, description="Filter for claims with anomaly flags"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List claims with pagination, search, type, date and anomaly filters."""
    from datetime import datetime
    query = db.query(Claim)

    if provider_id:
        query = query.filter(Claim.provider_id == provider_id.strip())

    if search:
        s = f"%{search.strip()}%"
        query = query.filter((Claim.claim_id.ilike(s)) | (Claim.provider_id.ilike(s)) | (Claim.beneficiary_id.ilike(s)))

    if claim_type:
        query = query.filter(Claim.claim_type.ilike(f"%{claim_type.strip()}%"))

    if start_date:
        try:
            dt = datetime.strptime(start_date.strip(), "%Y-%m-%d")
            query = query.filter(Claim.claim_date >= dt)
        except ValueError:
            pass

    if end_date:
        try:
            dt = datetime.strptime(end_date.strip(), "%Y-%m-%d")
            query = query.filter(Claim.claim_date <= dt)
        except ValueError:
            pass

    if has_anomaly is not None:
        if has_anomaly:
            query = query.join(ClaimAnalysis).filter(ClaimAnalysis.analysis_status == "FLAGGED")
        else:
            query = query.join(ClaimAnalysis).filter(ClaimAnalysis.analysis_status == "COMPLETED")

    total = query.count()
    offset = (page - 1) * page_size
    claims = query.order_by(desc(Claim.created_at)).offset(offset).limit(page_size).all()

    items = []
    for c in claims:
        analysis = c.analyses[0] if c.analyses else None
        raw_flags = analysis.anomaly_flags if analysis else []
        flags = []
        if isinstance(raw_flags, list):
            for f in raw_flags:
                if isinstance(f, str):
                    flags.append(f)
                elif isinstance(f, dict):
                    flags.append(f.get("flag", f.get("description", f.get("rule", f.get("reason", str(f))))))
                else:
                    flags.append(str(f))
        elif isinstance(raw_flags, str):
            flags = [raw_flags]

        items.append(
            ClaimListItem(
                claim_id=c.claim_id,
                provider_id=c.provider_id,
                beneficiary_id=c.beneficiary_id,
                claim_type=c.claim_type,
                claim_date=c.claim_date,
                reimbursement_amount=c.reimbursement_amount,
                anomaly_flags=flags,
                has_anomalies=bool(flags),
            )
        )

    total_pages = (total + page_size - 1) // page_size if total > 0 else 1
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "claims": items,
    }


@router.get("/{claim_id}", response_model=ClaimListItem)
def get_claim(
    claim_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve individual claim summary."""
    c = db.query(Claim).filter(Claim.claim_id == claim_id).first()
    if not c:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim '{claim_id}' not found.",
        )

    analysis = c.analyses[0] if c.analyses else None
    raw_flags = analysis.anomaly_flags if analysis else []
    flags = []
    if isinstance(raw_flags, list):
        for f in raw_flags:
            if isinstance(f, str):
                flags.append(f)
            elif isinstance(f, dict):
                flags.append(f.get("flag", f.get("description", f.get("rule", f.get("reason", str(f))))))
            else:
                flags.append(str(f))
    elif isinstance(raw_flags, str):
        flags = [raw_flags]

    return ClaimListItem(
        claim_id=c.claim_id,
        provider_id=c.provider_id,
        beneficiary_id=c.beneficiary_id,
        claim_type=c.claim_type,
        claim_date=c.claim_date,
        reimbursement_amount=c.reimbursement_amount,
        anomaly_flags=flags,
        has_anomalies=bool(flags),
    )


@router.get("/{claim_id}/analysis", response_model=ClaimAnalysisDetail)
def get_claim_analysis(
    claim_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve detailed anomaly detection findings and associated provider risk score."""
    c = db.query(Claim).filter(Claim.claim_id == claim_id).first()
    if not c:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Claim '{claim_id}' not found.",
        )

    analysis = db.query(ClaimAnalysis).filter(ClaimAnalysis.claim_id == claim_id).first()
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No analysis record found for claim '{claim_id}'.",
        )

    provider_score = None
    provider_tier = None
    if analysis.provider_risk_score_id:
        rs = db.query(ProviderRiskScore).filter(ProviderRiskScore.id == analysis.provider_risk_score_id).first()
        if rs:
            provider_score = rs.risk_score
            provider_tier = rs.risk_tier

    return ClaimAnalysisDetail(
        claim_id=c.claim_id,
        provider_id=c.provider_id,
        reimbursement_amount=c.reimbursement_amount,
        anomaly_score=analysis.anomaly_score or 0.0,
        anomaly_flags=analysis.anomaly_flags or [],
        rules_triggered=analysis.anomaly_rules_triggered or [],
        analysis_status=analysis.analysis_status,
        data_availability_warnings=[],
        provider_risk_score=provider_score,
        provider_risk_tier=provider_tier,
        disclaimer=settings.REGULATORY_DISCLAIMER,
    )
