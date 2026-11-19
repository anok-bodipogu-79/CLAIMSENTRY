"""
HealthGuard AI - Overview Dashboard Endpoints
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from backend.app.core.config import settings
from backend.app.core.dependencies import get_current_user
from backend.app.db.database import get_db
from backend.app.db.models import User, Provider, Beneficiary, Claim, ProviderRiskScore, ClaimAnalysis, CSVUpload
from backend.app.db.schemas import (
    OverviewSummaryResponse,
    OverviewChartsResponse,
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
    Overview summary delivering core and extended KPIs calculated dynamically from database records:
      1. total_providers
      2. total_claims
      3. high_risk_providers
      4. claims_requiring_investigation
      5. total_beneficiaries
      6. high_risk_claims, medium_risk_claims, low_risk_claims
      7. total_reimbursement, avg_claim_amount
      8. claims_analyzed_today, claims_analyzed_this_month
    """
    # 1. Total providers & claims & beneficiaries
    total_providers = db.query(func.count(Provider.id)).scalar() or 0
    total_claims = db.query(func.count(Claim.id)).scalar() or 0
    total_beneficiaries = db.query(func.count(Beneficiary.id)).scalar() or 0

    # 2. High-risk providers (latest assessment score >= 0.70 or risk_tier == 'HIGH')
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

    # 3. Claims requiring investigation (either flagged by anomaly detection OR associated with high-risk provider)
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
        .filter(Claim.provider_id.in_(high_risk_provider_ids.select()))
    )

    claims_requiring_investigation = (
        claims_with_anomalies.union(claims_under_high_risk_providers).count()
    )

    # 4. Extended claim risk and financial statistics from PostgreSQL
    high_risk_claims = (
        db.query(func.count(ClaimAnalysis.id))
        .filter((ClaimAnalysis.analysis_status == "FLAGGED") | (ClaimAnalysis.anomaly_score >= 0.70))
        .scalar() or 0
    )
    medium_risk_claims = (
        db.query(func.count(ClaimAnalysis.id))
        .filter(
            ClaimAnalysis.anomaly_score >= 0.22,
            ClaimAnalysis.anomaly_score < 0.70,
            ClaimAnalysis.analysis_status != "FLAGGED",
        )
        .scalar() or 0
    )
    low_risk_claims = max(0, int(total_claims) - int(high_risk_claims) - int(medium_risk_claims))

    total_reimb = float(db.query(func.sum(Claim.reimbursement_amount)).scalar() or 0.0)
    avg_reimb = round(total_reimb / int(total_claims), 2) if total_claims > 0 else 0.0

    # 5. Temporal analysis counts
    now_utc = datetime.now(timezone.utc)
    today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    month_start = now_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    analyzed_today = (
        db.query(func.count(Claim.id))
        .filter(Claim.created_at >= today_start)
        .scalar() or 0
    )
    analyzed_this_month = (
        db.query(func.count(Claim.id))
        .filter(Claim.created_at >= month_start)
        .scalar() or 0
    )

    return {
        "total_providers": int(total_providers),
        "total_claims": int(total_claims),
        "total_beneficiaries": int(total_beneficiaries),
        "high_risk_providers": int(high_risk_providers),
        "claims_requiring_investigation": int(claims_requiring_investigation),
        "high_risk_claims": int(high_risk_claims),
        "medium_risk_claims": int(medium_risk_claims),
        "low_risk_claims": int(low_risk_claims),
        "total_reimbursement": round(total_reimb, 2),
        "avg_claim_amount": avg_reimb,
        "claims_analyzed_today": int(analyzed_today),
        "claims_analyzed_this_month": int(analyzed_this_month),
        "model_version": settings.MODEL_VERSION,
        "risk_thresholds": {
            "low_medium_cutoff": settings.DEFAULT_LOW_MEDIUM_THRESHOLD,
            "medium_high_cutoff": settings.DEFAULT_MEDIUM_HIGH_THRESHOLD,
        },
        "disclaimer": settings.REGULATORY_DISCLAIMER,
    }


@router.get("/charts", response_model=OverviewChartsResponse)
def get_overview_charts(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieve PostgreSQL-aggregated time-series and comparative chart data for Dashboard Overview.
    Guaranteed: returns genuine PostgreSQL aggregates; returns clean empty arrays when no records exist.
    """
    claims = db.query(Claim).order_by(Claim.created_at.desc()).all()
    if not claims:
        return {
            "claims_by_month": [],
            "claims_over_time": [],
            "provider_activity": [],
            "recent_claims": [],
        }

    # 1. Group claims by month (using claim_date, falling back to created_at)
    monthly_map: Dict[str, Dict[str, Any]] = {}
    temporal_map: Dict[str, Dict[str, Any]] = {}
    provider_activity_map: Dict[str, Dict[str, Any]] = {}

    for c in claims:
        dt = c.claim_date or c.created_at
        month_key = dt.strftime("%Y-%m") if dt else "Unknown"
        date_key = dt.strftime("%Y-%m-%d") if dt else "Unknown"
        created_date_key = c.created_at.strftime("%Y-%m-%d") if c.created_at else date_key
        amt = float(c.reimbursement_amount or 0.0)

        # Monthly aggregation
        if month_key not in monthly_map:
            monthly_map[month_key] = {"month": month_key, "claims_count": 0, "total_reimbursement": 0.0}
        monthly_map[month_key]["claims_count"] += 1
        monthly_map[month_key]["total_reimbursement"] = round(monthly_map[month_key]["total_reimbursement"] + amt, 2)

        # Temporal analysis vs service date
        if created_date_key not in temporal_map:
            temporal_map[created_date_key] = {
                "date": created_date_key,
                "analysis_count": 0,
                "service_count": 0,
            }
        temporal_map[created_date_key]["analysis_count"] += 1
        if date_key in temporal_map:
            temporal_map[date_key]["service_count"] += 1
        else:
            temporal_map[date_key] = {"date": date_key, "analysis_count": 0, "service_count": 1}

        # Provider activity
        pid = c.provider_id or "Unknown"
        if pid not in provider_activity_map:
            provider_activity_map[pid] = {"provider_id": pid, "claim_count": 0, "total_reimbursement": 0.0}
        provider_activity_map[pid]["claim_count"] += 1
        provider_activity_map[pid]["total_reimbursement"] = round(provider_activity_map[pid]["total_reimbursement"] + amt, 2)

    # Sort monthly trends chronologically
    claims_by_month = sorted(list(monthly_map.values()), key=lambda x: x["month"])
    claims_over_time = sorted(list(temporal_map.values()), key=lambda x: x["date"])[-30:]

    # Top 10 providers by claim volume
    provider_activity = sorted(
        list(provider_activity_map.values()),
        key=lambda x: x["claim_count"],
        reverse=True,
    )[:10]

    # Recent claims list (top 10)
    recent_claims = []
    for c in claims[:10]:
        analysis = c.analyses[0] if c.analyses else None
        raw_flags = analysis.anomaly_flags if analysis else []
        flags = []
        if isinstance(raw_flags, list):
            for f in raw_flags:
                flags.append(f if isinstance(f, str) else str(f))
        elif isinstance(raw_flags, str):
            flags = [raw_flags]

        score = float(analysis.anomaly_score) if (analysis and analysis.anomaly_score is not None) else None
        tier = "HIGH" if (score and score >= 0.70) or (analysis and analysis.analysis_status == "FLAGGED") else ("MEDIUM" if score and score >= 0.22 else "LOW")

        recent_claims.append({
            "claim_id": c.claim_id,
            "provider_id": c.provider_id,
            "beneficiary_id": c.beneficiary_id,
            "claim_type": c.claim_type,
            "claim_date": c.claim_date.strftime("%Y-%m-%d") if c.claim_date else None,
            "reimbursement_amount": float(c.reimbursement_amount or 0.0),
            "currency": c.currency or "USD",
            "original_reimbursement": float(c.original_reimbursement or c.reimbursement_amount or 0.0),
            "risk_score": score,
            "risk_tier": tier,
            "has_anomalies": bool(flags),
            "anomaly_flags": flags,
            "analysis_status": analysis.analysis_status if analysis else "COMPLETED",
            "created_at": c.created_at.isoformat() if c.created_at else None,
        })

    return {
        "claims_by_month": claims_by_month,
        "claims_over_time": claims_over_time,
        "provider_activity": provider_activity,
        "recent_claims": recent_claims,
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
                provider_id=score.provider_id,
                risk_score=float(score.risk_score),
                risk_tier=score.risk_tier,
                timestamp=score.created_at,
                created_at=score.created_at,
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
