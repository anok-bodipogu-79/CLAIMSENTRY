"""
HealthGuard AI - Claim Anomaly Detection Service
"""

from typing import List, Dict, Any, Optional
import pandas as pd
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.models import Claim, ClaimAnalysis
from ML.claim_anomaly import detect_claim_anomalies, DEFAULT_PEER_REIMBURSEMENT_P95


class AnomalyService:
    """Manages claim anomaly rules and persistence of claim-level evaluations."""

    def run_claim_anomalies(
        self,
        claims: Any,
        historical_claims_df: Optional[pd.DataFrame] = None,
        peer_reference_data: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Run the 8-rule claim anomaly detection."""
        return detect_claim_anomalies(
            claims_df=claims,
            provider_reference_data=peer_reference_data,
            peer_reimbursement_p95=DEFAULT_PEER_REIMBURSEMENT_P95,
            historical_claims_df=historical_claims_df,
        )

    def persist_claim_analysis(
        self,
        db: Session,
        claim_id: str,
        provider_risk_score_id: Optional[int],
        anomaly_record: Dict[str, Any],
    ) -> ClaimAnalysis:
        """Store anomaly detection results for an individual claim."""
        flags = anomaly_record.get("anomaly_flags", [])
        rules_triggered = anomaly_record.get("rules_triggered", [])
        status_str = "FLAGGED" if flags else "COMPLETED"

        analysis = ClaimAnalysis(
            claim_id=claim_id,
            provider_risk_score_id=provider_risk_score_id,
            anomaly_score=float(len(flags)),
            anomaly_flags=flags,
            anomaly_rules_triggered=rules_triggered,
            analysis_status=status_str,
        )
        db.add(analysis)
        db.flush()
        return analysis


anomaly_service = AnomalyService()
