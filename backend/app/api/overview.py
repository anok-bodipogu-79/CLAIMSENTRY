"""
HealthGuard AI - Overview Dashboard Endpoints
"""

from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from backend.app.core.config import settings
from backend.app.core.dependencies import get_current_user
from backend.app.db.database import get_db
from backend.app.db.models import User, Provider, Claim, ProviderRiskScore, ClaimAnalysis, CSVUpload
from backend.app.db.schemas import (
    OverviewSummaryResponse,
    RecentActivityResponse,
    RecentActivityItem,
    RiskDistributionResponse,
    RiskDistributionTier,
)

router = APIRouter(prefix="/overview", tags=["Overview Dashboard"])


@router.get("/summary", response_model=OverviewSummaryResponse)
def get_overview_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Overview summary delivering exactly four core KPIs calculated dynamically from database records:
      1. total_providers
      2. total_claims
      3. high_risk_providers
      4. claims_requiring_investigation
    """
    # 1. Total providers
    total_providers = db.query(func.count(Provider.id)).scalar() or 0

    # 2. Total claims
    total_claims = db.query(func.count(Claim.id)).scalar() or 0

    # 3. High-risk providers (latest assessment score >= 0.70 or risk_tier == 'HIGH')
    # Using subquery for latest provider risk score
    latest_subq = (
        db.query(
            ProviderRiskScore.provider_id,
            func.max(ProviderRiskScore.id).label("max_id")
        )
        .group_by(ProviderRiskScore.provider_id)
        .subquery()
    )

    high_risk_providers = (
        db.query(func.count(ProviderRiskScore.id))
        .join(latest_subq, ProviderRiskScore.id == latest_subq.c.max_id)
        .filter(ProviderRiskScore.risk_tier == "HIGH")
        .scalar() or 0
    )

    # 4. Claims requiring investigation (either flagged by anomaly detection OR associated with high-risk provider)
    high_risk_provider_ids = (
        db.query(ProviderRiskScore.provider_id)
        .join(latest_subq, ProviderRiskScore.id == latest_subq.c.max_id)
        .filter(ProviderRiskScore.risk_tier == "HIGH")
        .subquery()
    )

    claims_with_anomalies = (
        db.query(Claim.claim_id)
        .join(ClaimAnalysis, Claim.claim_id == ClaimAnalysis.claim_id)
        .filter(ClaimAnalysis.analysis_status == "FLAGGED")
    )

    claims_under_high_risk_providers = (
        db.query(Claim.claim_id)
        .filter(Claim.provider_id.in_(high_risk_provider_ids))
    )

    claims_requiring_investigation = (
        claims_with_anomalies.union(claims_under_high_risk_providers).count()
    )

    return {
        "total_providers": int(total_providers),
        "total_claims": int(total_claims),
        "high_risk_providers": int(high_risk_providers),
        "claims_requiring_investigation": int(claims_requiring_investigation),
        "model_version": settings.MODEL_VERSION,
        "risk_thresholds": {
            "low_medium_cutoff": settings.DEFAULT_LOW_MEDIUM_THRESHOLD,
            "medium_high_cutoff": settings.DEFAULT_MEDIUM_HIGH_THRESHOLD,
        },
        "disclaimer": settings.REGULATORY_DISCLAIMER,
    }


@router.get("/recent-activity", response_model=RecentActivityResponse)
def get_recent_activity(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    limit: int = 10,
):
    """Retrieve chronologically recent provider risk evaluations and batch uploads."""
    activities: List[RecentActivityItem] = []

    # Recent provider risk assessments
    recent_scores = (
        db.query(ProviderRiskScore)
        .order_by(desc(ProviderRiskScore.created_at))
        .limit(limit)
        .all()
    )
    for score in recent_scores:
        activities.append(
            RecentActivityItem(
                id=score.id,
                activity_type="PROVIDER_ASSESSMENT",
                description=f"Provider {score.provider_id} assessed with {score.risk_score:.1%} fraud probability",
                entity_id=score.provider_id,
                risk_tier=score.risk_tier,
                timestamp=score.created_at,
            )
        )

    # Recent CSV uploads
    recent_uploads = (
        db.query(CSVUpload)
        .order_by(desc(CSVUpload.created_at))
        .limit(limit)
        .all()
    )
    for up in recent_uploads:
        activities.append(
            RecentActivityItem(
                id=up.id,
                activity_type="CSV_UPLOAD",
                description=f"CSV upload '{up.filename}' processed ({up.successful_rows} analysed, {up.rejected_rows} rejected)",
                entity_id=f"UPLOAD_{up.id}",
                risk_tier=None,
                timestamp=up.created_at,
            )
        )

    # Sort combined activity chronologically descending
    activities.sort(key=lambda x: x.timestamp, reverse=True)
    res_list = activities[:limit]
    return {"activities": res_list, "recent_activities": res_list}


@router.get("/risk-distribution", response_model=RiskDistributionResponse)
def get_risk_distribution(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve provider population distribution across LOW, MEDIUM, and HIGH risk tiers."""
    latest_subq = (
        db.query(
            ProviderRiskScore.provider_id,
            func.max(ProviderRiskScore.id).label("max_id")
        )
        .group_by(ProviderRiskScore.provider_id)
        .subquery()
    )

    tier_counts = (
        db.query(
            ProviderRiskScore.risk_tier,
            func.count(ProviderRiskScore.id).label("cnt")
        )
        .join(latest_subq, ProviderRiskScore.id == latest_subq.c.max_id)
        .group_by(ProviderRiskScore.risk_tier)
        .all()
    )

    count_map = {tier: count for tier, count in tier_counts}
    low_cnt = count_map.get("LOW", 0)
    med_cnt = count_map.get("MEDIUM", 0)
    high_cnt = count_map.get("HIGH", 0)
    total = low_cnt + med_cnt + high_cnt

    def pct(c):
        return round((c / total) * 100, 2) if total > 0 else 0.0

    distribution = [
        RiskDistributionTier(
            tier="LOW",
            count=low_cnt,
            percentage=pct(low_cnt),
            description=f"Fraud probability below {settings.DEFAULT_LOW_MEDIUM_THRESHOLD:.2f}",
        ),
        RiskDistributionTier(
            tier="MEDIUM",
            count=med_cnt,
            percentage=pct(med_cnt),
            description=f"Fraud probability between {settings.DEFAULT_LOW_MEDIUM_THRESHOLD:.2f} and {settings.DEFAULT_MEDIUM_HIGH_THRESHOLD:.2f}",
        ),
        RiskDistributionTier(
            tier="HIGH",
            count=high_cnt,
            percentage=pct(high_cnt),
            description=f"Fraud probability {settings.DEFAULT_MEDIUM_HIGH_THRESHOLD:.2f} and above",
        ),
    ]

    return {
        "total_assessed": total,
        "distribution": distribution,
    }
