"""
Tests for Phase A: Date Validation & Monthly Chart Correctness
"""

from datetime import datetime, timezone, timedelta
import pytest
from pydantic import ValidationError
from backend.app.db.schemas import SingleClaimRequest
from backend.app.services.claim_analysis_pipeline import claim_analysis_pipeline
import pandas as pd


def test_valid_historical_claim_dates():
    req = SingleClaimRequest(
        ClaimID="CLM_HIST_01",
        ProviderID="PRV51003",
        InscClaimAmtReimbursed=1000.0,
        ClaimStartDt="2009-03-01",
        ClaimEndDt="2009-03-05",
        Currency="INR",
    )
    assert req.ClaimStartDt == "2009-03-01"
    assert req.ClaimEndDt == "2009-03-05"


def test_future_claim_start_date_rejected():
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%d")
    with pytest.raises(ValidationError) as exc_info:
        SingleClaimRequest(
            ClaimID="CLM_FUT_01",
            ProviderID="PRV51003",
            InscClaimAmtReimbursed=1000.0,
            ClaimStartDt=tomorrow,
            Currency="INR",
        )
    assert "cannot be in the future" in str(exc_info.value)


def test_future_claim_end_date_rejected():
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%d")
    with pytest.raises(ValidationError) as exc_info:
        SingleClaimRequest(
            ClaimID="CLM_FUT_02",
            ProviderID="PRV51003",
            InscClaimAmtReimbursed=1000.0,
            ClaimStartDt="2020-01-01",
            ClaimEndDt=tomorrow,
            Currency="INR",
        )
    assert "ClaimEndDt" in str(exc_info.value)
    assert "cannot be in the future" in str(exc_info.value)


def test_future_admission_date_rejected():
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%d")
    with pytest.raises(ValidationError) as exc_info:
        SingleClaimRequest(
            ClaimID="CLM_FUT_03",
            ProviderID="PRV51003",
            InscClaimAmtReimbursed=1000.0,
            ClaimStartDt="2020-01-01",
            ClaimEndDt="2020-01-05",
            AdmissionDt=tomorrow,
            DischargeDt=tomorrow,
            ClaimType="Inpatient",
            Currency="INR",
        )
    assert "AdmissionDt" in str(exc_info.value)
    assert "cannot be in the future" in str(exc_info.value)


def test_future_discharge_date_rejected():
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%d")
    with pytest.raises(ValidationError) as exc_info:
        SingleClaimRequest(
            ClaimID="CLM_FUT_04",
            ProviderID="PRV51003",
            InscClaimAmtReimbursed=1000.0,
            ClaimStartDt="2020-01-01",
            ClaimEndDt="2020-01-05",
            AdmissionDt="2020-01-01",
            DischargeDt=tomorrow,
            ClaimType="Inpatient",
            Currency="INR",
        )
    assert "DischargeDt" in str(exc_info.value)
    assert "cannot be in the future" in str(exc_info.value)


def test_claim_start_after_end_rejected():
    with pytest.raises(ValidationError) as exc_info:
        SingleClaimRequest(
            ClaimID="CLM_INV_01",
            ProviderID="PRV51003",
            InscClaimAmtReimbursed=1000.0,
            ClaimStartDt="2009-03-10",
            ClaimEndDt="2009-03-05",
            Currency="INR",
        )
    assert "cannot be earlier than ClaimStartDt" in str(exc_info.value)


def test_admission_after_discharge_rejected():
    with pytest.raises(ValidationError) as exc_info:
        SingleClaimRequest(
            ClaimID="CLM_INV_02",
            ProviderID="PRV51003",
            InscClaimAmtReimbursed=1000.0,
            ClaimStartDt="2009-03-01",
            ClaimEndDt="2009-03-15",
            AdmissionDt="2009-03-10",
            DischargeDt="2009-03-05",
            ClaimType="Inpatient",
            Currency="INR",
        )
    assert "cannot be earlier than AdmissionDt" in str(exc_info.value)


def test_monthly_trends_zero_fill_missing_months():
    # Construct mock p_history with claims in Jan 2009 and Mar 2009 (Feb missing)
    fake_history = pd.DataFrame([
        {
            "ClaimID": "CLM_M1",
            "ProviderID": "PRV_MOCK_1",
            "ClaimStartDt": "2009-01-10",
            "InscClaimAmtReimbursed": 100.0,
        },
        {
            "ClaimID": "CLM_M2",
            "ProviderID": "PRV_MOCK_1",
            "ClaimStartDt": "2009-03-15",
            "InscClaimAmtReimbursed": 200.0,
        },
    ])

    trends = claim_analysis_pipeline.get_provider_monthly_trends(
        provider_id="PRV_MOCK_1",
        db=None,
        currency="USD",
        period="12_months",
        p_history=fake_history,
    )

    months_data = trends["months"]
    assert len(months_data) == 12, f"Expected 12 months, got {len(months_data)}"

    # Ending month must be 2009-03
    assert months_data[-1]["month"] == "2009-03"
    assert months_data[-1]["claim_count"] == 1
    assert months_data[-1]["reimbursement"] == 200.0

    # 2009-02 must be present with 0 activity
    feb_entry = next((m for m in months_data if m["month"] == "2009-02"), None)
    assert feb_entry is not None, "February 2009 should be present in the 12-month calendar series"
    assert feb_entry["claim_count"] == 0
    assert feb_entry["reimbursement"] == 0.0

    # 2009-01 must have 1 claim
    jan_entry = next((m for m in months_data if m["month"] == "2009-01"), None)
    assert jan_entry is not None
    assert jan_entry["claim_count"] == 1
    assert jan_entry["reimbursement"] == 100.0


def test_monthly_trends_current_claim_included_once():
    fake_history = pd.DataFrame([
        {
            "ClaimID": "CLM_EXISTING",
            "ProviderID": "PRV_MOCK_2",
            "ClaimStartDt": "2009-05-01",
            "InscClaimAmtReimbursed": 500.0,
        }
    ])

    # Case A: new claim not in history
    trends_new = claim_analysis_pipeline.get_provider_monthly_trends(
        provider_id="PRV_MOCK_2",
        db=None,
        currency="USD",
        period="12_months",
        current_claim={
            "claim_id": "CLM_NEW",
            "date": "2009-05-10",
            "reimb_usd": 150.0,
        },
        p_history=fake_history,
    )
    may_new = next(m for m in trends_new["months"] if m["month"] == "2009-05")
    assert may_new["claim_count"] == 2
    assert may_new["reimbursement"] == 650.0

    # Case B: claim ID already in history (idempotent duplicate submission)
    trends_dup = claim_analysis_pipeline.get_provider_monthly_trends(
        provider_id="PRV_MOCK_2",
        db=None,
        currency="USD",
        period="12_months",
        current_claim={
            "claim_id": "CLM_EXISTING",
            "date": "2009-05-01",
            "reimb_usd": 500.0,
        },
        p_history=fake_history,
    )
    may_dup = next(m for m in trends_dup["months"] if m["month"] == "2009-05")
    assert may_dup["claim_count"] == 1
    assert may_dup["reimbursement"] == 500.0
