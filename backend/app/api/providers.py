"""
HealthGuard AI - Provider Exploration & Evaluation Endpoints
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import desc, func

from backend.app.core.config import settings
from backend.app.core.dependencies import get_current_user
from backend.app.db.database import get_db
from backend.app.db.models import User, Provider, Claim, ProviderRiskScore, Explanation
from backend.app.db.schemas import (
    PaginatedProvidersResponse,
    ProviderListItem,
    ProviderRiskDetail,
    ProviderExplanationResponse,
    PaginatedClaimsResponse,
    ClaimListItem,
    ProviderMonthlyTrendsResponse,
)
from backend.app.services.claim_analysis_pipeline import claim_analysis_pipeline

router = APIRouter(prefix="/providers", tags=["Providers"])


@router.get("", response_model=PaginatedProvidersResponse)
def list_providers(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    risk_tier: Optional[str] = Query(None, description="Filter by risk tier: LOW, MEDIUM, HIGH"),
    search: Optional[str] = Query(None, description="Search by ProviderID"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List healthcare providers with pagination, tier filtering, and ID searching."""
    query = db.query(Provider)

    if search:
        s = f"%{search.strip()}%"
        query = query.filter(
            (Provider.provider_id.ilike(s)) | (Provider.provider_name.ilike(s))
        )

    # Join with latest risk score if filtering by risk_tier or for display
    latest_score_subq = (
        db.query(
            ProviderRiskScore.provider_id,
            func.max(ProviderRiskScore.id).label("max_id")
        )
        .group_by(ProviderRiskScore.provider_id)
        .subquery()
    )

    if risk_tier:
        query = (
            query.join(latest_score_subq, Provider.provider_id == latest_score_subq.c.provider_id)
            .join(ProviderRiskScore, ProviderRiskScore.id == latest_score_subq.c.max_id)
            .filter(ProviderRiskScore.risk_tier == risk_tier.upper())
        )

    total = query.count()
    offset = (page - 1) * page_size
    providers = query.order_by(Provider.provider_id).offset(offset).limit(page_size).all()

    items = []
    for p in providers:
        # Fetch latest risk score for this provider
        latest_rs = (
            db.query(ProviderRiskScore)
            .filter(ProviderRiskScore.provider_id == p.provider_id)
            .order_by(desc(ProviderRiskScore.id))
            .first()
        )
        total_claims = db.query(func.count(Claim.id)).filter(Claim.provider_id == p.provider_id).scalar() or 0
        total_reimb = db.query(func.sum(Claim.reimbursement_amount)).filter(Claim.provider_id == p.provider_id).scalar() or 0.0

        items.append(
            ProviderListItem(
                provider_id=p.provider_id,
                provider_name=p.provider_name or f"Healthcare Provider {p.provider_id}",
                risk_score=latest_rs.risk_score if latest_rs else None,
                risk_tier=latest_rs.risk_tier if latest_rs else None,
                total_claims=int(total_claims),
                total_reimbursement=float(total_reimb),
                last_assessed=latest_rs.created_at if latest_rs else None,
            )
        )

    total_pages = (total + page_size - 1) // page_size if total > 0 else 1
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "providers": items,
    }


@router.get("/{provider_id}", response_model=ProviderListItem)
def get_provider(
    provider_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve detailed provider summary."""
    p = db.query(Provider).filter(Provider.provider_id == provider_id).first()
    if not p:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider '{provider_id}' not found.",
        )

    latest_rs = (
        db.query(ProviderRiskScore)
        .filter(ProviderRiskScore.provider_id == p.provider_id)
        .order_by(desc(ProviderRiskScore.id))
        .first()
    )
    total_claims = db.query(func.count(Claim.id)).filter(Claim.provider_id == p.provider_id).scalar() or 0
    total_reimb = db.query(func.sum(Claim.reimbursement_amount)).filter(Claim.provider_id == p.provider_id).scalar() or 0.0

    return ProviderListItem(
        provider_id=p.provider_id,
        provider_name=p.provider_name or f"Healthcare Provider {p.provider_id}",
        risk_score=latest_rs.risk_score if latest_rs else None,
        risk_tier=latest_rs.risk_tier if latest_rs else None,
        total_claims=int(total_claims),
        total_reimbursement=float(total_reimb),
        last_assessed=latest_rs.created_at if latest_rs else None,
    )


@router.get("/{provider_id}/risk", response_model=ProviderRiskDetail)
@router.get("/{provider_id}/risk-score", response_model=ProviderRiskDetail)
def get_provider_risk(
    provider_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve latest risk evaluation score, tier, and base model probabilities."""
    p = db.query(Provider).filter(Provider.provider_id == provider_id).first()
    if not p:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider '{provider_id}' not found.",
        )

    latest_rs = (
        db.query(ProviderRiskScore)
        .filter(ProviderRiskScore.provider_id == provider_id)
        .order_by(desc(ProviderRiskScore.id))
        .first()
    )
    if not latest_rs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No risk evaluation found for provider '{provider_id}'.",
        )

    base_preds = (latest_rs.analysis_snapshot or {}).get("base_model_probabilities")

    return ProviderRiskDetail(
        provider_id=latest_rs.provider_id,
        risk_score=latest_rs.risk_score,
        raw_score=latest_rs.raw_score,
        risk_tier=latest_rs.risk_tier,
        threshold_low=latest_rs.threshold_low,
        threshold_high=latest_rs.threshold_high,
        base_model_probabilities=base_preds,
        model_version=settings.MODEL_VERSION,
        created_at=latest_rs.created_at,
        disclaimer=settings.REGULATORY_DISCLAIMER,
    )


@router.get("/{provider_id}/claims", response_model=PaginatedClaimsResponse)
def get_provider_claims(
    provider_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve all claims submitted by this healthcare provider."""
    p = db.query(Provider).filter(Provider.provider_id == provider_id).first()
    if not p:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider '{provider_id}' not found.",
        )

    query = db.query(Claim).filter(Claim.provider_id == provider_id)
    total = query.count()
    offset = (page - 1) * page_size
    claims = query.order_by(desc(Claim.claim_date)).offset(offset).limit(page_size).all()

    items = []
    for c in claims:
        analysis = c.analyses[0] if c.analyses else None
        flags = analysis.anomaly_flags if analysis else []
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


@router.get("/{provider_id}/explanation", response_model=ProviderExplanationResponse)
def get_provider_explanation(
    provider_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve AI explainability attributions and top risk factors for provider."""
    latest_rs = (
        db.query(ProviderRiskScore)
        .filter(ProviderRiskScore.provider_id == provider_id)
        .order_by(desc(ProviderRiskScore.id))
        .first()
    )
    if not latest_rs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No risk evaluation score found for provider '{provider_id}'.",
        )

    expl = (
        db.query(Explanation)
        .filter(Explanation.provider_risk_score_id == latest_rs.id)
        .order_by(desc(Explanation.id))
        .first()
    )

    contribs = expl.feature_contributions if expl and expl.feature_contributions else {}
    top_factors = contribs.get("top_risk_factors", [])
    pos_contribs = contribs.get("positive_contributors", [])
    neg_contribs = contribs.get("negative_contributors", [])

    return ProviderExplanationResponse(
        provider_id=provider_id,
        risk_score=latest_rs.risk_score,
        risk_tier=latest_rs.risk_tier,
        explanation_title=expl.explanation_title if expl else settings.EXPLANATION_TITLE,
        explanation_text=expl.explanation_text if expl else f"Provider assessed at {latest_rs.risk_score:.1%} fraud probability ({latest_rs.risk_tier} risk).",
        top_risk_factors=top_factors,
        positive_contributors=pos_contribs,
        negative_contributors=neg_contribs,
        temporal_explanation=None,
        disclaimer=settings.REGULATORY_DISCLAIMER,
    )


@router.get("/{provider_id}/monthly-trends", response_model=ProviderMonthlyTrendsResponse)
def get_provider_monthly_trends(
    provider_id: str,
    period: str = Query("12_months", description="Time period: 6_months, 12_months, all"),
    currency: str = Query("INR", description="Reimbursement currency: INR or USD"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieve authentic monthly reimbursement and claim count aggregations for a provider.
    Used for investigator visual analysis (Reimbursement Trend & Claim Activity charts).
    """
    return claim_analysis_pipeline.get_provider_monthly_trends(
        provider_id=provider_id,
        db=db,
        currency=currency,
        period=period,
    )
