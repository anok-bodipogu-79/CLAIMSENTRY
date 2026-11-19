"""
HealthGuard AI - Machine Learning Prediction Service
"""

from typing import Dict, Any, Optional, List
import pandas as pd
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.models import Provider, Claim, ProviderRiskScore, ModelVersion
from ML.predict import HealthGuardPredictor, get_predictor, predict_claims


class PredictionService:
    """Manages ML inference, model versioning, and persistence of provider risk assessments."""

    def __init__(self):
        self._current_version = settings.MODEL_VERSION
        self._predictor = HealthGuardPredictor(artifact_version=self._current_version)

    @property
    def predictor(self) -> HealthGuardPredictor:
        if self._predictor.artifact_version != settings.MODEL_VERSION:
            self._current_version = settings.MODEL_VERSION
            self._predictor = HealthGuardPredictor(artifact_version=self._current_version)
        return self._predictor

    def get_readiness(self) -> Dict[str, Any]:
        """Check artifact verification and readiness."""
        pred = self.predictor
        return {
            "is_ready": pred.is_ready,
            "version": pred.artifact_version,
            "error": pred.verification_error,
            "manifest": pred.manifest if pred.is_ready else None,
        }

    def set_model_version(self, version: str) -> bool:
        """Switch active model version (e.g. 'v1' or 'v2')."""
        if version not in ["v1", "v2"]:
            raise ValueError(f"Unsupported model version: {version}. Expected 'v1' or 'v2'.")
        new_pred = HealthGuardPredictor(artifact_version=version)
        if not new_pred.is_ready:
            raise RuntimeError(f"Model version {version} failed verification: {new_pred.verification_error}")
        settings.MODEL_VERSION = version
        self._current_version = version
        self._predictor = new_pred
        return True

    def predict_claims_pipeline(
        self,
        claims_input: Any,
        provider_id: Optional[str] = None,
        historical_claims_df: Optional[pd.DataFrame] = None,
        patient_historical_claims: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute the real shared Phase 3 prediction and anomaly pipeline."""
        return self.predictor.predict_claims(
            claims_input=claims_input,
            provider_id=provider_id,
            historical_claims_df=historical_claims_df,
            patient_historical_claims=patient_historical_claims,
            include_shap=True,
            include_temporal=True,
            include_anomalies=True,
        )

    def persist_provider_assessment(
        self,
        db: Session,
        provider_id: str,
        eval_result: Dict[str, Any],
        model_version_str: str,
    ) -> ProviderRiskScore:
        """
        Store or update a provider risk score snapshot.
        Enforces: one score snapshot per evaluation, linked to the provider and model version.
        """
        # Ensure Provider record exists
        provider = db.query(Provider).filter(Provider.provider_id == provider_id).first()
        if not provider:
            provider = Provider(provider_id=provider_id)
            db.add(provider)
            db.flush()

        # Ensure ModelVersion record exists
        model_ver = db.query(ModelVersion).filter(ModelVersion.version == model_version_str).first()
        if not model_ver:
            model_ver = ModelVersion(
                version=model_version_str,
                model_type="MultiModalStackingEnsemble",
                training_status="TRAINED",
                manifest_path=f"ML/artifacts/{model_version_str}/model_manifest.json",
                is_active=True,
            )
            db.add(model_ver)
            db.flush()

        # Create new ProviderRiskScore record
        risk_score = ProviderRiskScore(
            provider_id=provider_id,
            model_version_id=model_ver.id,
            risk_score=float(eval_result.get("calibrated_fraud_probability", eval_result.get("fraud_probability", 0.0))),
            raw_score=float(eval_result.get("raw_stacked_probability", 0.0)),
            risk_tier=str(eval_result.get("risk_tier", "LOW")),
            threshold_low=float(eval_result.get("risk_thresholds_applied", {}).get("low_medium_cutoff", 0.22)),
            threshold_high=float(eval_result.get("risk_thresholds_applied", {}).get("medium_high_cutoff", 0.70)),
            analysis_snapshot={
                "base_model_probabilities": eval_result.get("base_model_probabilities", {}),
                "total_claims_analyzed": eval_result.get("total_claims_analyzed", 0),
                "has_limited_history": eval_result.get("has_limited_history", False),
                "warnings": eval_result.get("warnings", []),
            },
        )
        db.add(risk_score)
        db.flush()
        return risk_score


prediction_service = PredictionService()
