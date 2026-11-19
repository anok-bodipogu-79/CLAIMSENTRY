"""
HealthGuard AI - Pydantic Request & Response Schemas
"""

import math
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any, Union
import pandas as pd
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator


class Currency(str, Enum):
    INR = "INR"
    USD = "USD"



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
    total_beneficiaries: Optional[int] = 0
    high_risk_claims: Optional[int] = 0
    medium_risk_claims: Optional[int] = 0
    low_risk_claims: Optional[int] = 0
    total_reimbursement: Optional[float] = 0.0
    avg_claim_amount: Optional[float] = 0.0
    claims_analyzed_today: Optional[int] = 0
    claims_analyzed_this_month: Optional[int] = 0


class OverviewChartsResponse(BaseModel):
    claims_by_month: List[Dict[str, Any]] = []
    claims_over_time: List[Dict[str, Any]] = []
    provider_activity: List[Dict[str, Any]] = []
    recent_claims: List[Dict[str, Any]] = []


class RecentActivityItem(BaseModel):
    id: int
    activity_type: str  # "PROVIDER_ASSESSMENT", "CSV_UPLOAD"
    description: str
    entity_id: str
    provider_id: Optional[str] = None
    risk_score: Optional[float] = None
    risk_tier: Optional[str] = None
    timestamp: datetime
    created_at: Optional[datetime] = None


class RecentActivityResponse(BaseModel):
    activities: List[RecentActivityItem]
    recent_activities: Optional[List[RecentActivityItem]] = None


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
    currency: Optional[str] = "USD"
    original_reimbursement: Optional[float] = None
    provider_risk_score: Optional[float] = None
    provider_risk_tier: Optional[str] = None
    investigation_priority: Optional[str] = None
    processing_status: Optional[str] = "SUCCESS"

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
    ClaimID: str = Field(..., min_length=2, max_length=50, example="CLM_1001")
    ProviderID: str = Field(..., min_length=2, max_length=50, example="PRV_1001")
    BeneID: Optional[str] = Field(default=None, max_length=50, example="BEN_1001")
    InscClaimAmtReimbursed: float = Field(..., ge=0.0, example=3500.0)
    Currency: Optional[str] = Field(default=None, example="INR", description="Input currency: INR or USD. Defaults to USD for legacy CMS records, or INR from user application.")
    ClaimStartDt: str = Field(..., example="2009-03-01")
    ClaimEndDt: Optional[str] = Field(default=None, example="2009-03-05")
    AdmissionDt: Optional[str] = Field(default=None, example="2009-03-01")
    DischargeDt: Optional[str] = Field(default=None, example="2009-03-05")
    ClaimType: Optional[str] = Field(default="Outpatient", example="Outpatient")
    DeductibleAmtPaid: Optional[float] = Field(default=0.0, ge=0.0, example=100.0)
    AttendingPhysician: Optional[str] = Field(default=None, example="PHY31201")
    OperatingPhysician: Optional[str] = Field(default=None)
    OtherPhysician: Optional[str] = Field(default=None)
    ClmDiagnosisCode_1: Optional[str] = Field(default=None, example="4019")
    ClmDiagnosisCode_2: Optional[str] = Field(default=None)
    ClmAdmitDiagnosisCode: Optional[str] = Field(default=None)
    ClmProcedureCode_1: Optional[str] = Field(default=None)
    DOD: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def validate_amounts_and_currency(self) -> "SingleClaimRequest":
        cur = (self.Currency or "USD").strip().upper()
        reimb = self.InscClaimAmtReimbursed
        ded = self.DeductibleAmtPaid or 0.0

        if reimb is None or not math.isfinite(reimb):
            raise ValueError("Reimbursement amount must be a finite numerical value.")
        if reimb < 0.0:
            raise ValueError("Reimbursement amount cannot be negative.")

        if ded is None or not math.isfinite(ded):
            raise ValueError("Deductible amount must be a finite numerical value.")
        if ded < 0.0:
            raise ValueError("Deductible amount cannot be negative.")

        if cur == "INR":
            max_reimb = 41750000.0  # 500000 USD * 83.50 INR/USD
            max_ded = 2087500.0     # 25000 USD * 83.50 INR/USD
            if reimb > max_reimb:
                raise ValueError(
                    f"Reimbursement amount exceeds maximum plausible clinical limit "
                    f"(₹4,17,50,000.00, equivalent to $500000.00 USD clinical cap)."
                )
            if ded > max_ded:
                raise ValueError(
                    f"Deductible amount exceeds maximum plausible limit "
                    f"(₹20,87,500.00, equivalent to $25000.00 USD cap)."
                )
        else:
            if reimb > 500000.0:
                raise ValueError("Reimbursement amount exceeds maximum plausible clinical limit ($500000.00).")
            if ded > 25000.0:
                raise ValueError("Deductible amount exceeds maximum plausible Medicare deductible ($25000.00).")

        self.InscClaimAmtReimbursed = round(reimb, 2)
        self.DeductibleAmtPaid = round(ded, 2)
        return self

    @field_validator("ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt")
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return None
        v_clean = str(v).strip()
        try:
            pd.to_datetime(v_clean)
        except Exception:
            raise ValueError(f"Invalid date format '{v}'. Expected valid ISO date (YYYY-MM-DD).")
        return v_clean

    @model_validator(mode="after")
    def validate_dates_consistency(self) -> "SingleClaimRequest":
        today = datetime.now(timezone.utc).date()
        date_checks = [
            ("ClaimStartDt", self.ClaimStartDt),
            ("ClaimEndDt", self.ClaimEndDt),
            ("AdmissionDt", self.AdmissionDt),
            ("DischargeDt", self.DischargeDt),
        ]
        for field_name, val in date_checks:
            if val:
                parsed_dt = pd.to_datetime(val).date()
                if parsed_dt > today:
                    raise ValueError(
                        f"{field_name} ({val}) cannot be in the future. "
                        f"Service dates must be on or before current date ({today.isoformat()})."
                    )

        if self.ClaimStartDt and self.ClaimEndDt:
            dt_start = pd.to_datetime(self.ClaimStartDt)
            dt_end = pd.to_datetime(self.ClaimEndDt)
            if dt_end < dt_start:
                raise ValueError(f"ClaimEndDt ({self.ClaimEndDt}) cannot be earlier than ClaimStartDt ({self.ClaimStartDt}).")
            if (dt_end - dt_start).days > 365:
                raise ValueError("Claim duration between ClaimStartDt and ClaimEndDt cannot exceed 365 days.")

        if self.AdmissionDt and self.DischargeDt:
            dt_adm = pd.to_datetime(self.AdmissionDt)
            dt_dis = pd.to_datetime(self.DischargeDt)
            if dt_dis < dt_adm:
                raise ValueError(f"DischargeDt ({self.DischargeDt}) cannot be earlier than AdmissionDt ({self.AdmissionDt}).")
            if (dt_dis - dt_adm).days > 365:
                raise ValueError("Inpatient length of stay cannot exceed 365 days.")
        return self


class SingleClaimResponse(BaseModel):
    claim_id: str
    provider_id: str
    provider_risk_score: float
    provider_risk_tier: str
    model_version: str
    claim_anomaly_findings: List[Dict[str, Any]]
    anomaly_flags: List[str]
    has_anomalies: bool
    anomaly_score: Optional[float] = None
    explanation_title: str
    explanation: Dict[str, Any]
    investigation_recommendation: str
    disclaimer: str
    currency: Optional[str] = "INR"
    original_reimbursement: Optional[float] = None
    original_deductible: Optional[float] = None
    reimbursement_inr: Optional[float] = None
    reimbursement_usd: Optional[float] = None
    exchange_rate_applied: Optional[float] = None
    exchange_rate_source: Optional[str] = None
    investigator_report: Optional[Dict[str, Any]] = None
    base_model_probabilities: Optional[Dict[str, float]] = None
    monthly_trends: Optional[Dict[str, Any]] = None
    data_availability: Optional[Dict[str, Any]] = None
    provider_history: Optional[Dict[str, Any]] = None
    beneficiary_history: Optional[Dict[str, Any]] = None


class ProviderMonthlyPoint(BaseModel):
    month: str
    reimbursement: float
    claim_count: int


class ProviderMonthlyTrendsResponse(BaseModel):
    provider_id: str
    period: str
    currency: str
    total_reimbursement: float
    total_claims: int
    months: List[ProviderMonthlyPoint]
    has_historical_records: Optional[bool] = True
    message: Optional[str] = None


class InvestigatorDecisionRequest(BaseModel):
    claim_id: str
    provider_id: str
    decision_status: str  # "Pending Review", "Investigation Required", "Under Investigation", "No Issue Identified", "Suspected Irregularity – Escalation Required", "Closed"
    notes: Optional[str] = ""
    decision_date: Optional[str] = None


class InvestigatorDecisionResponse(BaseModel):
    decision_id: int
    claim_id: str
    provider_id: str
    decision_status: str
    notes: str
    decision_date: str
    investigator_email: str
    audit_logged: bool


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
    processed_rows: int = 0
    progress_pct: float = 0.0
    reconciliation_verified: bool
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


class CSVPreviewRow(BaseModel):
    row_number: int
    claim_id: Optional[str] = None
    provider_id: Optional[str] = None
    beneficiary_id: Optional[str] = None
    claim_type: Optional[str] = None
    reimbursement: Optional[float] = None
    currency: Optional[str] = None
    claim_start_date: Optional[str] = None
    status: str  # "VALID", "INVALID", "DUPLICATE"
    issues: List[str] = []


class CSVPreviewResponse(BaseModel):
    filename: str
    total_rows: int
    valid_rows_count: int
    invalid_rows_count: int
    duplicate_rows_count: int
    detected_columns: List[str]
    missing_required_columns: List[str]
    preview_rows: List[CSVPreviewRow]
    warnings: List[str] = []


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
