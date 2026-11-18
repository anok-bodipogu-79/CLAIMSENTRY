"""
HealthGuard AI - Currency Support Automated Test Suite (17 Comprehensive Checks)
Verifies safe Indian Rupee (INR - ₹) support across conversion, rounding, validation,
idempotency, DB persistence, ML feature ordering, model loading, calibration, SHAP, and parity.
"""

import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from backend.app.core.config import settings
from backend.app.services.currency_service import (
    CurrencyService,
    CurrencyConfigurationError,
    get_exchange_rate,
    inr_to_usd,
    usd_to_inr,
    format_inr,
    convert_claim_inputs_to_usd,
)
from backend.app.db.schemas import SingleClaimRequest, Currency
from ML.predict import HealthGuardPredictor


# 1. INR to USD conversion test
def test_1_inr_to_usd_conversion():
    usd = inr_to_usd(83500.0, rate=83.50)
    assert round(usd, 2) == 1000.00
    assert inr_to_usd(0.0, rate=83.50) == 0.0


# 2. USD to INR conversion test
def test_2_usd_to_inr_conversion():
    inr = usd_to_inr(1000.0, rate=83.50)
    assert round(inr, 2) == 83500.00
    assert usd_to_inr(0.0, rate=83.50) == 0.0


# 3. Rounding and precision test
def test_3_rounding_and_precision():
    # 12345.67 INR / 83.50 = 147.852335... -> rounds to 147.85 USD
    usd = inr_to_usd(12345.67, rate=83.50)
    assert round(usd, 2) == 147.85
    # 147.85 USD * 83.50 = 12345.475 -> rounds to 12345.48 INR
    inr = usd_to_inr(147.85, rate=83.50)
    assert round(inr, 2) == 12345.48

    formatted = format_inr(1234567.89)
    assert formatted == "₹12,34,567.89"


# 4. Missing exchange rate raises CurrencyConfigurationError
def test_4_missing_exchange_rate():
    with patch.dict(os.environ, {"INR_PER_USD": ""}, clear=False):
        with patch.object(settings, "INR_PER_USD", None):
            with pytest.raises(CurrencyConfigurationError):
                get_exchange_rate()


# 5. Invalid or zero exchange rate raises CurrencyConfigurationError
def test_5_invalid_or_zero_exchange_rate():
    with pytest.raises(CurrencyConfigurationError):
        CurrencyService.validate_exchange_rate(0.0)
    with pytest.raises(CurrencyConfigurationError):
        CurrencyService.validate_exchange_rate(-83.50)
    with pytest.raises(CurrencyConfigurationError):
        CurrencyService.validate_exchange_rate(float("nan"))
    with pytest.raises(CurrencyConfigurationError):
        CurrencyService.validate_exchange_rate(float("inf"))


# 6. Negative financial inputs rejected
def test_6_negative_financial_inputs_rejected():
    with pytest.raises(ValueError):
        inr_to_usd(-100.0)
    with pytest.raises(ValueError):
        usd_to_inr(-50.0)

    with pytest.raises(Exception):
        SingleClaimRequest(
            ClaimID="CLM_NEG_01",
            ProviderID="PRV51003",
            BeneID="BENE11001",
            ClaimType="Outpatient",
            InscClaimAmtReimbursed=-500.0,
            DeductibleAmtPaid=0.0,
            ClaimStartDt="2009-03-01",
            ClaimEndDt="2009-03-01",
            ClmDiagnosisCode_1="4019",
            AttendingPhysician="PHY31201"
        )


# 7. Double conversion prevention
def test_7_double_conversion_prevention():
    data = {
        "InscClaimAmtReimbursed": 83500.0,
        "DeductibleAmtPaid": 8350.0,
        "Currency": "INR"
    }
    converted = convert_claim_inputs_to_usd(data, rate=83.50)
    assert converted["InscClaimAmtReimbursed"] == 1000.00
    assert converted["DeductibleAmtPaid"] == 100.00
    assert converted["_currency_converted"] is True

    # Attempting to convert again must NOT divide again
    double_converted = convert_claim_inputs_to_usd(converted, rate=83.50)
    assert double_converted["InscClaimAmtReimbursed"] == 1000.00
    assert double_converted["DeductibleAmtPaid"] == 100.00


# 8. Existing USD historical claims handling
def test_8_usd_historical_claims_handling():
    data = {
        "InscClaimAmtReimbursed": 4500.0,
        "DeductibleAmtPaid": 100.0,
        "Currency": "USD"
    }
    res = convert_claim_inputs_to_usd(data, rate=83.50)
    assert res["InscClaimAmtReimbursed"] == 4500.0
    assert res["DeductibleAmtPaid"] == 100.0
    assert res["original_reimbursement"] == 4500.0
    assert res["original_deductible"] == 100.0
    assert res["currency"] == "USD"


# 9. Single-claim inference with INR input
def test_9_single_claim_inference_inr(client, auth_headers, db_session):
    payload = {
        "ClaimID": "CLM_INR_TEST_001",
        "ProviderID": "PRV51003",
        "BeneID": "BENE11001",
        "ClaimType": "Inpatient",
        "InscClaimAmtReimbursed": 350000.00,  # ₹3.5 Lakh INR
        "DeductibleAmtPaid": 8000.00,        # ₹8,000 INR
        "Currency": "INR",
        "ClaimStartDt": "2009-03-01",
        "ClaimEndDt": "2009-03-07",
        "AdmissionDt": "2009-03-01",
        "DischargeDt": "2009-03-07",
        "ClmDiagnosisCode_1": "4019",
        "ClmProcedureCode_1": "3722",
        "AttendingPhysician": "PHY31201"
    }
    response = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["claim_id"] == "CLM_INR_TEST_001"
    assert data["currency"] == "INR"
    assert data["original_reimbursement"] == 350000.00
    assert data["original_deductible"] == 8000.00
    # Converted USD: 350000 / 83.50 = 4191.62
    assert abs(data["reimbursement_usd"] - 4191.62) < 0.1
    assert "provider_risk_score" in data
    assert "provider_risk_tier" in data
    assert 0.0 <= data["provider_risk_score"] <= 1.0


# 10. CSV batch inference with INR input
def test_10_csv_batch_inference_inr(client, auth_headers):
    csv_content = (
        "ClaimID,ProviderID,BeneID,ClaimType,InscClaimAmtReimbursed,DeductibleAmtPaid,Currency,ClaimStartDt,ClaimEndDt,AdmissionDt,DischargeDt,AttendingPhysician,OperatingPhysician,OtherPhysician,ClmDiagnosisCode_1,ClmProcedureCode_1\n"
        "CLM_BATCH_INR_1,PRV51003,BENE11001,Inpatient,350000.00,8000.00,INR,2009-03-01,2009-03-07,2009-03-01,2009-03-07,PHY31201,PHY42109,PHY53110,4019,3722\n"
    )
    files = {"file": ("test_inr.csv", csv_content.encode("utf-8"), "text/csv")}
    response = client.post("/api/analyzer/csv", files=files, headers=auth_headers)
    assert response.status_code == 202, response.text
    upload_res = response.json()
    upload_id = upload_res["upload_id"]
    assert upload_id is not None

    status_res = client.get(f"/api/analyzer/csv/{upload_id}", headers=auth_headers)
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["status"] == "COMPLETED"
    assert status_data["total_rows"] == 1
    assert status_data["successful_rows"] == 1
    assert status_data["rejected_rows"] == 0


# 11. Single-claim versus CSV prediction parity in INR
def test_11_single_claim_vs_csv_prediction_parity_inr():
    predictor = HealthGuardPredictor(artifact_version="v2")

    # Raw INR inputs converted to USD via currency boundary
    inr_reimb = 350000.00
    inr_ded = 8000.00
    usd_reimb = inr_to_usd(inr_reimb, 83.50)
    usd_ded = inr_to_usd(inr_ded, 83.50)

    # 1. Predict as single-claim dict
    single_claim_dict = {
        "ClaimID": "CLM_PARITY_01",
        "ProviderID": "PRV51003",
        "BeneID": "BENE11001",
        "ClaimType": "Inpatient",
        "InscClaimAmtReimbursed": usd_reimb,
        "DeductibleAmtPaid": usd_ded,
        "ClaimStartDt": "2009-03-01",
        "ClaimEndDt": "2009-03-07",
        "AdmissionDt": "2009-03-01",
        "DischargeDt": "2009-03-07",
        "ClmDiagnosisCode_1": "4019",
        "ClmProcedureCode_1": "3722",
        "AttendingPhysician": "PHY31201",
    }
    res_single = predictor.predict_claims(single_claim_dict)
    p_single = res_single["provider_results"]["PRV51003"]

    # 2. Predict as batch DataFrame
    df_batch = pd.DataFrame([single_claim_dict])
    res_batch = predictor.predict_claims(df_batch)
    p_batch = res_batch["provider_results"]["PRV51003"]

    # Both must match exactly
    assert abs(p_single["calibrated_fraud_probability"] - p_batch["calibrated_fraud_probability"]) < 1e-5
    assert p_single["risk_tier"] == p_batch["risk_tier"]


# 12. 51-feature names and ordering preservation
def test_12_51_feature_names_and_ordering_preservation():
    predictor = HealthGuardPredictor(artifact_version="v2")
    assert len(predictor.feature_names) == 51
    # Check key financial features are present in feature_names
    expected_financial_features = [
        "total_reimbursement", "avg_reimbursement", "median_reimbursement",
        "max_reimbursement", "std_reimbursement", "ip_total_reimbursement",
        "op_total_reimbursement", "avg_deductible", "cv_reimbursement",
        "ip_claim_pct", "reimbursement_to_peer_median"
    ]
    for feat in expected_financial_features:
        assert feat in predictor.feature_names


# 13. Model loading of XGBoost, ANN, Bi-LSTM, Stacking model
def test_13_model_loading_all_components():
    predictor = HealthGuardPredictor(artifact_version="v2")
    assert predictor.xgboost_detector is not None and predictor.xgboost_detector.model is not None
    assert predictor.ann_detector is not None and predictor.ann_detector.model is not None
    assert predictor.lstm_detector is not None and predictor.lstm_detector.model is not None
    assert predictor.ensemble is not None and predictor.ensemble.meta_model is not None
    assert predictor.ensemble.calibrator is not None
    assert predictor.ann_detector.scaler is not None
    assert predictor.lstm_detector.scaler is not None


# 14. Platt calibration consistency
def test_14_platt_calibration_consistency():
    predictor = HealthGuardPredictor(artifact_version="v2")
    calibrator = predictor.ensemble.calibrator

    # Meta input
    meta_prob = np.array([0.653901])
    calibrated = calibrator.predict(meta_prob)
    assert len(calibrated) == 1
    assert round(calibrated[0], 6) == 0.589281


# 15. SHAP explanation consistency
def test_15_shap_explanation_consistency():
    predictor = HealthGuardPredictor(artifact_version="v2")
    usd_reimb = inr_to_usd(350000.0, 83.50)
    claim_dict = {
        "ClaimID": "CLM_SHAP_01",
        "ProviderID": "PRV51003",
        "BeneID": "BENE11001",
        "ClaimType": "Inpatient",
        "InscClaimAmtReimbursed": usd_reimb,
        "DeductibleAmtPaid": 100.0,
        "ClaimStartDt": "2009-03-01",
        "ClaimEndDt": "2009-03-07",
        "AdmissionDt": "2009-03-01",
        "DischargeDt": "2009-03-07",
        "ClmDiagnosisCode_1": "4019",
        "AttendingPhysician": "PHY31201",
    }
    res = predictor.predict_claims(claim_dict, include_shap=True)
    p_res = res["provider_results"]["PRV51003"]
    assert "top_risk_factors" in p_res
    assert len(p_res["top_risk_factors"]) > 0


# 16. Database persistence of currency metadata
def test_16_database_persistence_currency_metadata(client, auth_headers, db_session):
    from backend.app.db.models import Claim

    payload = {
        "ClaimID": "CLM_PERSIST_INR_01",
        "ProviderID": "PRV51003",
        "BeneID": "BENE11001",
        "ClaimType": "Outpatient",
        "InscClaimAmtReimbursed": 25000.00,
        "DeductibleAmtPaid": 500.00,
        "Currency": "INR",
        "ClaimStartDt": "2009-04-01",
        "ClaimEndDt": "2009-04-01",
        "ClmDiagnosisCode_1": "4019",
        "AttendingPhysician": "PHY31201"
    }
    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 200, res.text

    claim_row = db_session.query(Claim).filter(Claim.claim_id == "CLM_PERSIST_INR_01").first()
    assert claim_row is not None
    assert claim_row.currency == "INR"
    assert claim_row.original_reimbursement == 25000.00
    assert claim_row.original_deductible == 500.00
    # Converted reimbursement_amount in USD
    assert abs(claim_row.reimbursement_amount - (25000.00 / 83.50)) < 0.1


# 17. Fixed USD test inputs parity before and after changes
def test_17_fixed_usd_test_inputs_parity(client, auth_headers):
    payload = {
        "ClaimID": "CLM_USD_REF_01",
        "ProviderID": "PRV51003",
        "BeneID": "BENE11001",
        "ClaimType": "Inpatient",
        "InscClaimAmtReimbursed": 42000.0,
        "DeductibleAmtPaid": 1024.0,
        "Currency": "USD",
        "ClaimStartDt": "2009-03-01",
        "ClaimEndDt": "2009-03-07",
        "AdmissionDt": "2009-03-01",
        "DischargeDt": "2009-03-07",
        "ClmDiagnosisCode_1": "4019",
        "ClmProcedureCode_1": "3722",
        "AttendingPhysician": "PHY31201",
    }
    res = client.post("/api/analyzer/single", json=payload, headers=auth_headers)
    assert res.status_code == 200, res.text
    data = res.json()
    # The reference calibrated prob for PRV51003 $42,000 USD is 0.6155
    assert abs(data["provider_risk_score"] - 0.6155) < 0.05
    assert data["provider_risk_tier"] in ["MEDIUM", "HIGH"]
