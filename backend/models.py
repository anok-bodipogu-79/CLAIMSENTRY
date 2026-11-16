"""
HealthGuard AI - Database Models and Pydantic Schemas
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from backend.database import Base
from backend.app.db.models import Provider as ProviderRecord, AuditLog


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class ProviderFeaturePayload(BaseModel):
    """Provider features payload for fraud scoring."""
    provider_id: str
    features: Optional[Dict[str, Any]] = None
    sequence: Optional[List[List[float]]] = None
    sequence_mask: Optional[List[int]] = None


class RiskFactor(BaseModel):
    feature: str
    feature_value: float
    shap_value: float
    direction: str
    peer_percentile: Optional[float] = None
    peer_median: Optional[float] = None
    benchmark_summary: Optional[str] = None


class PredictionResponse(BaseModel):
    """Fraud prediction output schema."""
    provider_id: str
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    calibrated_fraud_probability: float = Field(..., ge=0.0, le=1.0)
    raw_stacked_probability: float = Field(..., ge=0.0, le=1.0)
    predicted_fraud: int = Field(..., ge=0, le=1)
    risk_tier: str  # LOW, MEDIUM, HIGH
    base_model_probabilities: Dict[str, float]
    risk_thresholds_applied: Dict[str, Any]
    has_limited_history: bool = False
    warnings: Optional[List[str]] = None
    top_risk_factors: List[Dict[str, Any]] = []
    scope_disclosure: Optional[str] = None
    model_version: str = "1.0.0"
    timestamp: str
    disclaimer: str = "This is a risk assessment to guide investigation. It is not proof of fraud."


class BatchPredictionRequest(BaseModel):
    """Batch prediction request containing list of provider IDs."""
    provider_ids: List[str]


class RejectedProvider(BaseModel):
    provider_id: str
    reason: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response with full audit reconciliation."""
    predictions: List[PredictionResponse]
    total_uploaded: int
    analysed_count: int
    rejected_count: int
    rejected_providers: List[RejectedProvider] = []
    total_evaluated: Optional[int] = None  # Backwards compatibility alias for analysed_count
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    disclaimer: str = "This is a risk assessment to guide investigation. It is not proof of fraud."


class ModelReadinessResponse(BaseModel):
    """Readiness status of trained machine learning artifacts."""
    status: str
    is_ready: bool
    model_version: str
    artifacts_directory: str
    manifest_present: bool
    base_models: Dict[str, Any]
    ensemble: Dict[str, Any]
    risk_thresholds: Dict[str, Any]
    sha256_checksums: Dict[str, str]
    message: str
