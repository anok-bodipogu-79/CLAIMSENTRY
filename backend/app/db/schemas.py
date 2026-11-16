"""
HealthGuard AI - Pydantic Request & Response Schemas
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, EmailStr, Field


# ══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class UserRegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, example="Jane Investigator")
    email: EmailStr = Field(..., example="jane.investigator@healthguard.ai")
    password: str = Field(..., min_length=6, max_length=100, example="SecurePassword123")


class UserRoleUpdateRequest(BaseModel):
    role: str = Field(..., example="admin")


class UserLoginRequest(BaseModel):
    email: EmailStr = Field(..., example="jane.investigator@healthguard.ai")
    password: str = Field(..., example="SecurePassword123")


class UserResponse(BaseModel):
    id: int
    name: str
    email: EmailStr
    role: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class TokenPayload(BaseModel):
    sub: Optional[str] = None
    role: Optional[str] = None
    exp: Optional[int] = None


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW DASHBOARD SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class OverviewSummaryResponse(BaseModel):
    total_providers: int = Field(..., description="Total unique healthcare providers in database")
    total_claims: int = Field(..., description="Total insurance claims stored in database")
    high_risk_providers: int = Field(..., description="Providers with fraud risk score >= 0.70")
    claims_requiring_investigation: int = Field(..., description="Claims flagged by anomaly rules or linked to high-risk providers")
    model_version: str
    risk_thresholds: Dict[str, float]
    disclaimer: str


class RecentActivityItem(BaseModel):
    id: int
    activity_type: str  # "PROVIDER_ASSESSMENT", "CSV_UPLOAD"
    description: str
    entity_id: str
    risk_tier: Optional[str] = None
    timestamp: datetime


class RecentActivityResponse(BaseModel):
    activities: List[RecentActivityItem]


class RiskDistributionTier(BaseModel):
    tier: str
    count: int
    percentage: float
    description: str


class RiskDistributionResponse(BaseModel):
    total_assessed: int
    distribution: List[RiskDistributionTier]


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class ProviderListItem(BaseModel):
    provider_id: str
    provider_name: Optional[str] = None
    risk_score: Optional[float] = None
    risk_tier: Optional[str] = None
    total_claims: int = 0
    total_reimbursement: float = 0.0
    last_assessed: Optional[datetime] = None

    class Config:
        from_attributes = True


class PaginatedProvidersResponse(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    providers: List[ProviderListItem]


class ProviderRiskDetail(BaseModel):
    provider_id: str
    risk_score: float
    raw_score: Optional[float] = None
    risk_tier: str
    threshold_low: float
    threshold_high: float
    base_model_probabilities: Optional[Dict[str, float]] = None
    model_version: Optional[str] = None
    created_at: datetime
    disclaimer: str

    class Config:
        from_attributes = True


class ExplanationFeatureContribution(BaseModel):
    feature: str
    value: Union[float, int, str]
    attribution: float
    description: Optional[str] = None


class ProviderExplanationResponse(BaseModel):
    provider_id: str
    risk_score: float
    risk_tier: str
    explanation_title: str
    explanation_text: str
    top_risk_factors: List[Dict[str, Any]]
    positive_contributors: Optional[List[Dict[str, Any]]] = None
    negative_contributors: Optional[List[Dict[str, Any]]] = None
    temporal_explanation: Optional[Dict[str, Any]] = None
    disclaimer: str


# ══════════════════════════════════════════════════════════════════════════════
# CLAIM SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class ClaimListItem(BaseModel):
    claim_id: str
    provider_id: str
    beneficiary_id: Optional[str] = None
    claim_type: str
    claim_date: Optional[datetime] = None
    reimbursement_amount: float
    anomaly_flags: Optional[List[str]] = None
    has_anomalies: bool = False

    class Config:
        from_attributes = True


class PaginatedClaimsResponse(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    claims: List[ClaimListItem]


class ClaimAnalysisDetail(BaseModel):
    claim_id: str
    provider_id: str
    reimbursement_amount: float
    anomaly_score: float
    anomaly_flags: List[str]
    rules_triggered: List[Dict[str, Any]]
    analysis_status: str
    data_availability_warnings: Optional[List[str]] = None
    provider_risk_score: Optional[float] = None
    provider_risk_tier: Optional[str] = None
    disclaimer: str


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE CLAIM ANALYZER SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class SingleClaimRequest(BaseModel):
    ClaimID: str = Field(..., example="CLM100234")
    ProviderID: str = Field(..., example="PRV51003")
    BeneID: Optional[str] = Field(default=None, example="BENE11001")
    InscClaimAmtReimbursed: float = Field(..., ge=0.0, example=3500.0)
    ClaimStartDt: str = Field(..., example="2009-03-01")
    ClaimEndDt: Optional[str] = Field(default=None, example="2009-03-05")
    AdmissionDt: Optional[str] = Field(default=None, example="2009-03-01")
    DischargeDt: Optional[str] = Field(default=None, example="2009-03-05")
    ClaimType: Optional[str] = Field(default="Outpatient", example="Outpatient")
    DeductibleAmtPaid: Optional[float] = Field(default=0.0, example=100.0)
    AttendingPhysician: Optional[str] = Field(default=None, example="PHY31201")
    OperatingPhysician: Optional[str] = Field(default=None)
    OtherPhysician: Optional[str] = Field(default=None)
    ClmDiagnosisCode_1: Optional[str] = Field(default=None, example="4019")
    ClmDiagnosisCode_2: Optional[str] = Field(default=None)
    ClmAdmitDiagnosisCode: Optional[str] = Field(default=None)
    ClmProcedureCode_1: Optional[str] = Field(default=None)
    DOD: Optional[str] = Field(default=None)


class SingleClaimResponse(BaseModel):
    claim_id: str
    provider_id: str
    provider_risk_score: float
    provider_risk_tier: str
    model_version: str
    claim_anomaly_findings: List[Dict[str, Any]]
    anomaly_flags: List[str]
    has_anomalies: bool
    explanation_title: str
    explanation: Dict[str, Any]
    investigation_recommendation: str
    disclaimer: str


# ══════════════════════════════════════════════════════════════════════════════
# CSV UPLOAD SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class CSVUploadResponse(BaseModel):
    upload_id: int
    filename: str
    status: str
    total_rows: int
    successful_rows: int
    rejected_rows: int
    message: str


class CSVUploadDetail(BaseModel):
    upload_id: int
    filename: str
    status: str
    total_rows: int
    successful_rows: int
    rejected_rows: int
    reconciliation_verified: bool
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class CSVUploadRowResponse(BaseModel):
    row_number: int
    claim_id: Optional[str] = None
    processing_status: str
    rejection_reason: Optional[str] = None


class PaginatedCSVRowsResponse(BaseModel):
    upload_id: int
    total_rejected: int
    rejected_rows: List[CSVUploadRowResponse]


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class SettingsResponse(BaseModel):
    model_version: str
    low_medium_threshold: float
    medium_high_threshold: float
    database_connected: bool
    database_type: str
    models_ready: bool
    disclaimer: str


class SettingsUpdateRequest(BaseModel):
    model_version: Optional[str] = Field(default=None, example="v2")
    low_medium_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    medium_high_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ModelVersionSwitchRequest(BaseModel):
    version: str = Field(..., example="v2")
