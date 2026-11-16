"""
HealthGuard AI - Database Models and Pydantic Schemas
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Boolean
from backend.database import Base


# ══════════════════════════════════════════════════════════════════════════════
# SQLALCHEMY ORM MODELS
# ══════════════════════════════════════════════════════════════════════════════

class ProviderRecord(Base):
    """Provider database record storing fraud prediction results and metadata."""
    __tablename__ = "providers"

    id = Column(Integer, primary_key=True, index=True)
    provider_id = Column(String(50), unique=True, index=True, nullable=False)
    potential_fraud = Column(Integer, nullable=True)  # Ground truth label (0 or 1)
    predicted_probability = Column(Float, nullable=True)
    predicted_label = Column(Integer, nullable=True)
    risk_level = Column(String(20), default="LOW")  # LOW, MEDIUM, HIGH, CRITICAL
    total_reimbursement = Column(Float, default=0.0)
    total_claims = Column(Integer, default=0)
    unique_patients = Column(Integer, default=0)
    audit_status = Column(String(30), default="PENDING_REVIEW")
    notes = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AuditLog(Base):
    """Investigation audit trail for regulatory compliance."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    provider_id = Column(String(50), index=True, nullable=False)
    investigator_id = Column(String(50), nullable=False)
    action = Column(String(100), nullable=False)
    details = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)


# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class ProviderFeaturePayload(BaseModel):
    """Provider features payload for fraud scoring."""
    provider_id: str
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    """Fraud prediction output schema."""
    provider_id: str
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    predicted_fraud: int = Field(..., ge=0, le=1)
    risk_tier: str  # LOW, MEDIUM, HIGH, CRITICAL
    top_risk_factors: List[Dict[str, Any]] = []
    model_version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchPredictionRequest(BaseModel):
    """Batch prediction request containing list of provider IDs."""
    provider_ids: List[str]


class ModelMetricsResponse(BaseModel):
    """Performance evaluation metrics response."""
    model_name: str
    roc_auc: float
    pr_auc: float
    f1_score: float
    precision: float
    recall: float
    confusion_matrix: Dict[str, int]
    optimal_threshold: float
