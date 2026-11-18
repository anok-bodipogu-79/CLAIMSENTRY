"""
HealthGuard AI - PostgreSQL / SQLAlchemy Database Models
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, JSON, ForeignKey, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship

from backend.app.db.database import Base


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    """System users with role-based authorization and bcrypt password hash."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default="investigator", nullable=False)  # admin, investigator, auditor
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    csv_uploads = relationship("CSVUpload", back_populates="user", cascade="all, delete-orphan")
    case_notes = relationship("CaseNote", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")


class ModelVersion(Base):
    """Registry of trained machine learning model iterations and metadata."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String(50), unique=True, index=True, nullable=False)  # e.g., "v1", "v2"
    model_type = Column(String(100), default="MultiModalStackingEnsemble", nullable=False)
    training_status = Column(String(50), default="TRAINED", nullable=False)
    manifest_path = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    risk_scores = relationship("ProviderRiskScore", back_populates="model_version")


class Provider(Base):
    """Healthcare providers evaluated for anomalous billing and potential fraud."""
    __tablename__ = "providers"

    id = Column(Integer, primary_key=True, index=True)
    provider_id = Column(String(50), unique=True, index=True, nullable=False)
    provider_name = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    claims = relationship("Claim", back_populates="provider", cascade="all, delete-orphan")
    risk_scores = relationship("ProviderRiskScore", back_populates="provider", cascade="all, delete-orphan")
    case_notes = relationship("CaseNote", back_populates="provider", cascade="all, delete-orphan")


class Beneficiary(Base):
    """Healthcare beneficiaries (non-sensitive fields only; zero protected attributes used in ML)."""
    __tablename__ = "beneficiaries"

    id = Column(Integer, primary_key=True, index=True)
    beneficiary_id = Column(String(50), unique=True, index=True, nullable=False)
    renal_disease = Column(Boolean, default=False)
    chronic_conditions_count = Column(Integer, default=0)
    is_deceased = Column(Boolean, default=False)
    date_of_death = Column(DateTime, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    claims = relationship("Claim", back_populates="beneficiary")


class Claim(Base):
    """Individual insurance claims submitted by healthcare providers."""
    __tablename__ = "claims"

    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(String(50), unique=True, index=True, nullable=False)
    provider_id = Column(String(50), ForeignKey("providers.provider_id", ondelete="CASCADE"), index=True, nullable=False)
    beneficiary_id = Column(String(50), ForeignKey("beneficiaries.beneficiary_id", ondelete="SET NULL"), index=True, nullable=True)
    claim_type = Column(String(50), default="Outpatient", nullable=False)
    claim_date = Column(DateTime, nullable=True)
    claim_end_date = Column(DateTime, nullable=True)
    admission_date = Column(DateTime, nullable=True)
    discharge_date = Column(DateTime, nullable=True)
    reimbursement_amount = Column(Float, default=0.0, nullable=True)
    deductible_amount = Column(Float, default=0.0, nullable=True)
    inpatient_los = Column(Integer, default=0)
    currency = Column(String(10), default="USD", server_default="USD", nullable=True)
    original_reimbursement = Column(Float, nullable=True)
    original_deductible = Column(Float, nullable=True)
    attending_physician = Column(String(50), nullable=True)
    operating_physician = Column(String(50), nullable=True)
    other_physician = Column(String(50), nullable=True)
    diagnosis_codes = Column(JSON, nullable=True)
    procedure_codes = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    provider = relationship("Provider", back_populates="claims")
    beneficiary = relationship("Beneficiary", back_populates="claims")
    analyses = relationship("ClaimAnalysis", back_populates="claim", cascade="all, delete-orphan")


class ProviderRiskScore(Base):
    """Specific snapshot assessment of a healthcare provider's fraud risk."""
    __tablename__ = "provider_risk_scores"

    id = Column(Integer, primary_key=True, index=True)
    provider_id = Column(String(50), ForeignKey("providers.provider_id", ondelete="CASCADE"), index=True, nullable=False)
    model_version_id = Column(Integer, ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True)
    risk_score = Column(Float, nullable=False)  # Calibrated probability [0.0, 1.0]
    raw_score = Column(Float, nullable=True)   # Raw stacked probability
    risk_tier = Column(String(20), index=True, nullable=False)  # LOW, MEDIUM, HIGH
    threshold_low = Column(Float, default=0.22, nullable=False)
    threshold_high = Column(Float, default=0.70, nullable=False)
    analysis_snapshot = Column(JSON, nullable=True)  # Snapshot of probabilities, features, warnings
    created_at = Column(DateTime(timezone=True), default=utcnow, index=True, nullable=False)

    provider = relationship("Provider", back_populates="risk_scores")
    model_version = relationship("ModelVersion", back_populates="risk_scores")
    claim_analyses = relationship("ClaimAnalysis", back_populates="provider_risk_score")
    explanations = relationship("Explanation", back_populates="provider_risk_score", cascade="all, delete-orphan")


class ClaimAnalysis(Base):
    """Claim-level anomaly detection findings evaluated against training peer benchmarks."""
    __tablename__ = "claim_analyses"

    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(String(50), ForeignKey("claims.claim_id", ondelete="CASCADE"), index=True, nullable=False)
    provider_risk_score_id = Column(Integer, ForeignKey("provider_risk_scores.id", ondelete="SET NULL"), index=True, nullable=True)
    anomaly_score = Column(Float, default=0.0)
    anomaly_flags = Column(JSON, nullable=True)  # List of triggered flags
    anomaly_rules_triggered = Column(JSON, nullable=True)  # Rule details
    analysis_status = Column(String(30), default="COMPLETED", nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    claim = relationship("Claim", back_populates="analyses")
    provider_risk_score = relationship("ProviderRiskScore", back_populates="claim_analyses")
    explanations = relationship("Explanation", back_populates="claim_analysis", cascade="all, delete-orphan")


class Explanation(Base):
    """Explainability attributions (SHAP feature values, Bi-LSTM temporal occlusion)."""
    __tablename__ = "explanations"

    id = Column(Integer, primary_key=True, index=True)
    claim_analysis_id = Column(Integer, ForeignKey("claim_analyses.id", ondelete="CASCADE"), index=True, nullable=True)
    provider_risk_score_id = Column(Integer, ForeignKey("provider_risk_scores.id", ondelete="CASCADE"), index=True, nullable=True)
    explanation_type = Column(String(50), default="SHAP_FEATURE_IMPORTANCE", nullable=False)
    explanation_title = Column(String(255), default="AI Explanation – Why This Claim Received This Risk Score", nullable=False)
    explanation_text = Column(Text, nullable=False)
    feature_contributions = Column(JSON, nullable=True)  # Positive & negative top risk factors
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    claim_analysis = relationship("ClaimAnalysis", back_populates="explanations")
    provider_risk_score = relationship("ProviderRiskScore", back_populates="explanations")


class CSVUpload(Base):
    """Batch CSV claim upload processing record."""
    __tablename__ = "csv_uploads"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    filename = Column(String(255), nullable=False)
    total_rows = Column(Integer, default=0, nullable=False)
    successful_rows = Column(Integer, default=0, nullable=False)
    rejected_rows = Column(Integer, default=0, nullable=False)
    status = Column(String(30), default="PENDING", index=True, nullable=False)  # PENDING, PROCESSING, COMPLETED, FAILED
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User", back_populates="csv_uploads")
    rows = relationship("CSVUploadRow", back_populates="csv_upload", cascade="all, delete-orphan")


class CSVUploadRow(Base):
    """Itemized per-row tracking of batch CSV processing with explicit rejection reasons."""
    __tablename__ = "csv_upload_rows"

    id = Column(Integer, primary_key=True, index=True)
    csv_upload_id = Column(Integer, ForeignKey("csv_uploads.id", ondelete="CASCADE"), index=True, nullable=False)
    row_number = Column(Integer, nullable=False)
    claim_id = Column(String(50), nullable=True)
    processing_status = Column(String(20), nullable=False)  # SUCCESS, REJECTED
    rejection_reason = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    csv_upload = relationship("CSVUpload", back_populates="rows")


class CaseNote(Base):
    """Investigator notes for provider fraud review."""
    __tablename__ = "case_notes"

    id = Column(Integer, primary_key=True, index=True)
    provider_id = Column(String(50), ForeignKey("providers.provider_id", ondelete="CASCADE"), index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    note_text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    provider = relationship("Provider", back_populates="case_notes")
    user = relationship("User", back_populates="case_notes")


class AuditLog(Base):
    """System activity audit log for compliance and traceability."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    action = Column(String(100), index=True, nullable=False)
    resource_type = Column(String(50), nullable=True)
    resource_id = Column(String(50), nullable=True)
    details = Column(JSON, nullable=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    user = relationship("User", back_populates="audit_logs")


class RevokedToken(Base):
    """Blacklisted JWT tokens revoked upon user logout."""
    __tablename__ = "revoked_tokens"

    id = Column(Integer, primary_key=True, index=True)
    token = Column(String(512), unique=True, index=True, nullable=False)
    revoked_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)

