"""
HealthGuard AI - REST API Routes
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import (
    ProviderFeaturePayload,
    PredictionResponse,
    BatchPredictionRequest,
    ModelMetricsResponse,
)

router = APIRouter(prefix="/api/v1", tags=["HealthGuard AI"])

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
REP  = ROOT / "reports"


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Service health and readiness check."""
    return {
        "status": "healthy",
        "service": "HealthGuard AI Fraud Detection API",
        "version": "1.0.0",
        "phase": "Phase 1 - Data Preparation & Feature Engineering Completed",
        "models_trained": False,
    }


@router.post("/predict/provider", response_model=PredictionResponse)
def predict_provider_fraud(payload: ProviderFeaturePayload, db: Session = Depends(get_db)):
    """
    Score a single healthcare provider for potential fraudulent activity.
    In Phase 1, trained models are not yet available; returns HTTP 503.
    """
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Prediction model unavailable: Phase 2 model training (XGBoost, ANN, Bi-LSTM, Ensemble) "
               "must be completed before serving live fraud predictions.",
    )


@router.post("/predict/batch")
def predict_batch_providers(request: BatchPredictionRequest, db: Session = Depends(get_db)):
    """
    Run batch fraud evaluation across multiple providers.
    In Phase 1, trained models are not yet available; returns HTTP 503.
    """
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Batch evaluation unavailable: Trained Phase 2 ensemble model artifacts are required.",
    )


@router.get("/reports/summary")
def get_dataset_summary():
    """
    Retrieve Phase 1 dataset validation and preprocessing summary.
    Returns PHASE_1_VERIFIED strictly when all required validation checks are present and pass:
      1. Provider splits have zero overlap and overlap_free is True.
      2. Data leakage checks pass with 0 claim overlap across all splits.
      3. Feature specification matches the generated feature columns.
      4. Demographic feature check is clean with 0 protected attributes.
      5. Required provider counts and feature counts are strictly positive.
    If any check fails or is missing, returns an appropriate failed/incomplete status.
    """
    ver_path = REP / "evaluation" / "phase1_verification.json"
    if not ver_path.exists():
        return {
            "status": "PHASE_1_UNAVAILABLE",
            "message": "Phase 1 pipeline verification artifacts not found.",
            "provider_split": None,
            "fraud_distribution": None,
            "data_leakage_free": None,
            "feature_count": None,
            "demographic_clean": None,
            "checks_passed": False,
        }

    try:
        with open(ver_path, "r", encoding="utf-8") as f:
            ver = json.load(f)
    except Exception as e:
        return {
            "status": "PHASE_1_UNAVAILABLE",
            "message": f"Failed to parse Phase 1 verification artifacts: {str(e)}",
            "provider_split": None,
            "fraud_distribution": None,
            "data_leakage_free": None,
            "feature_count": None,
            "demographic_clean": None,
            "checks_passed": False,
        }

    # Verify presence of all required sections
    required_sections = ["provider_overlap", "leakage", "feature_spec", "demographic_check", "fraud_distribution"]
    missing_sections = [s for s in required_sections if not isinstance(ver.get(s), dict) or not ver.get(s)]
    if missing_sections:
        return {
            "status": "PHASE_1_INCOMPLETE",
            "message": f"Missing required verification sections: {', '.join(missing_sections)}",
            "provider_split": ver.get("provider_overlap") if isinstance(ver.get("provider_overlap"), dict) else None,
            "fraud_distribution": ver.get("fraud_distribution") if isinstance(ver.get("fraud_distribution"), dict) else None,
            "data_leakage_free": None,
            "feature_count": None,
            "demographic_clean": None,
            "checks_passed": False,
        }

    provider_split = ver["provider_overlap"]
    leakage = ver["leakage"]
    feature_spec = ver["feature_spec"]
    demographic = ver["demographic_check"]
    fraud_dist = ver["fraud_distribution"]

    # Check individual criteria
    failures = []

    # 1. Provider overlap & valid counts
    train_cnt = provider_split.get("train_count", 0)
    val_cnt = provider_split.get("val_count", 0)
    test_cnt = provider_split.get("test_count", 0)
    if train_cnt <= 0 or val_cnt <= 0 or test_cnt <= 0:
        failures.append(f"Invalid provider split counts: train={train_cnt}, val={val_cnt}, test={test_cnt}")

    if (
        provider_split.get("overlap_free") is not True
        or provider_split.get("train_val_overlap", 1) != 0
        or provider_split.get("train_test_overlap", 1) != 0
        or provider_split.get("val_test_overlap", 1) != 0
    ):
        failures.append("Provider split overlap detected or overlap_free is not True")

    # 2. Data leakage check
    if (
        leakage.get("leakage_free") is not True
        or leakage.get("train_val_claim_overlap", 1) != 0
        or leakage.get("train_test_claim_overlap", 1) != 0
        or leakage.get("val_test_claim_overlap", 1) != 0
    ):
        failures.append("Data leakage detected in claim records across splits")

    # 3. Feature specification matches data
    n_feat = feature_spec.get("n_features_in_data", 0)
    if n_feat <= 0:
        failures.append(f"Invalid feature count in data: {n_feat}")
    if (
        feature_spec.get("spec_matches_data") is not True
        or len(feature_spec.get("missing_in_spec", [])) > 0
        or len(feature_spec.get("extra_in_spec", [])) > 0
    ):
        failures.append("Feature specification does not match generated feature columns")

    # 4. Demographic check clean
    found_demographics = demographic.get("found_demographics", [])
    if demographic.get("clean") is not True or len(found_demographics) > 0:
        failures.append(f"Demographic check failed; protected attributes found: {found_demographics}")

    # Determine final status
    if failures:
        status_str = "PHASE_1_FAILED"
        msg = f"Phase 1 verification failed: {'; '.join(failures)}"
        checks_passed = False
    else:
        status_str = "PHASE_1_VERIFIED"
        msg = "All Phase 1 preprocessing, feature engineering, and data integrity checks passed."
        checks_passed = True

    return {
        "status": status_str,
        "message": msg,
        "provider_split": provider_split,
        "fraud_distribution": fraud_dist,
        "data_leakage_free": leakage.get("leakage_free"),
        "feature_count": n_feat,
        "demographic_clean": demographic.get("clean"),
        "spec_matches_data": feature_spec.get("spec_matches_data"),
        "checks_passed": checks_passed,
        "failures": failures if failures else None,
    }
