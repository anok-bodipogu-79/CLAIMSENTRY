"""
HealthGuard AI - Model Explanation Service (SHAP & Temporal Occlusion)
"""

from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.models import Explanation


class ExplanationService:
    """Manages explainability presentation and persistence."""

    TITLE = settings.EXPLANATION_TITLE

    def format_explanation(
        self,
        eval_result: Dict[str, Any],
        claim_anomalies: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Format an explanation report for human investigators."""
        top_factors = eval_result.get("top_risk_factors", [])
        positive_factors = eval_result.get("positive_contributors") or eval_result.get("top_positive_risk_factors") or [
            f for f in top_factors if f.get("shap_value", 0.0) > 0
        ]
        negative_factors = eval_result.get("negative_contributors") or eval_result.get("top_negative_risk_factors") or [
            f for f in top_factors if f.get("shap_value", 0.0) < 0
        ]
        risk_tier = eval_result.get("risk_tier", "LOW")
        prob = float(eval_result.get("calibrated_fraud_probability", eval_result.get("fraud_probability", 0.0)))

        # Build natural language narrative dynamically and accurately
        prob_pct = f"{prob * 100:.1f}%"
        lines = [
            f"The multi-modal stacking ensemble assigned this provider a calibrated fraud risk score of {prob_pct}, "
            f"placing them in the {risk_tier} risk tier."
        ]

        if positive_factors:
            lines.append("Key statistical risk drivers elevating fraud risk (XGBoost SHAP attribution):")
            for f in positive_factors[:5]:
                fname = f.get("feature", "Unknown")
                fval = f.get("actual_value", f.get("feature_value", f.get("value", "N/A")))
                sval = f.get("shap_value", 0.0)
                bench = f.get("benchmark_summary", "")
                bench_str = f" [{bench}]" if bench else ""
                lines.append(f" • {fname}: observed value {fval} (SHAP impact: {sval:+.4f}){bench_str}")
        elif top_factors:
            lines.append("Key statistical risk drivers identified by XGBoost SHAP attribution include:")
            for f in top_factors[:5]:
                fname = f.get("feature", "")
                val = f.get("value", f.get("actual_value", f.get("feature_value", "")))
                desc = f.get("description", "")
                lines.append(f" • {desc or fname}: observed value {val}")

        if negative_factors:
            lines.append("Key statistical risk drivers moderating fraud risk (XGBoost SHAP attribution):")
            for f in negative_factors[:3]:
                fname = f.get("feature", "Unknown")
                fval = f.get("actual_value", f.get("feature_value", f.get("value", "N/A")))
                sval = f.get("shap_value", 0.0)
                bench = f.get("benchmark_summary", "")
                bench_str = f" [{bench}]" if bench else ""
                lines.append(f" • {fname}: observed value {fval} (SHAP impact: {sval:+.4f}){bench_str}")

        if claim_anomalies:
            flag_count = sum(len(ca.get("anomaly_flags", [])) for ca in claim_anomalies)
            if flag_count > 0:
                lines.append(f"In addition, {flag_count} operational/clinical claim anomaly flag(s) were detected across submitted claims.")
            else:
                lines.append("No operational or clinical claim anomalies were detected on submitted claims.")

        temporal_expl = eval_result.get("temporal_explanation")
        if temporal_expl and isinstance(temporal_expl, dict) and "trajectory_summary" in temporal_expl:
            lines.append(f"Bi-LSTM Temporal Dynamics: {temporal_expl['trajectory_summary']}")

        explanation_text = "\n".join(lines)

        return {
            "title": self.TITLE,
            "text": explanation_text,
            "top_risk_factors": top_factors,
            "positive_contributors": positive_factors,
            "negative_contributors": negative_factors,
            "temporal_explanation": temporal_expl,
            "disclaimer": settings.REGULATORY_DISCLAIMER,
        }

    def persist_explanation(
        self,
        db: Session,
        provider_risk_score_id: int,
        claim_analysis_id: Optional[int],
        explanation_data: Dict[str, Any],
    ) -> Explanation:
        """Store explanation record in database."""
        expl = Explanation(
            claim_analysis_id=claim_analysis_id,
            provider_risk_score_id=provider_risk_score_id,
            explanation_type="SHAP_FEATURE_IMPORTANCE",
            explanation_title=self.TITLE,
            explanation_text=explanation_data.get("text", ""),
            feature_contributions={
                "top_risk_factors": explanation_data.get("top_risk_factors", []),
                "positive_contributors": explanation_data.get("positive_contributors", []),
                "negative_contributors": explanation_data.get("negative_contributors", []),
            },
        )
        db.add(expl)
        db.flush()
        return expl


explanation_service = ExplanationService()
