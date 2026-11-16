"""
HealthGuard AI - REST API Routes
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import (
    ProviderFeaturePayload,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelReadinessResponse,
)
from ML.predict import get_predictor

router = APIRouter(prefix="/api/v1", tags=["HealthGuard AI"])

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
REP  = ROOT / "reports"


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Service health and readiness check."""
    predictor = get_predictor()
    return {
        "status": "healthy",
        "service": "HealthGuard AI Fraud Detection API",
        "version": "1.0.0",
        "phase": "Phase 2 - Model Training & Evaluation Completed",
        "models_trained": predictor.is_ready,
        "model_version": predictor.manifest.get("model_version", "1.0.0") if predictor.is_ready else None,
        "calibrated": True,
    }


@router.get("/models/readiness", response_model=ModelReadinessResponse)
def get_model_readiness():
    """Check readiness and load status of all Phase 2 trained artifacts."""
    predictor = get_predictor()
    if not predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not ready or could not be loaded.",
        )
    manifest = predictor.manifest
    return {
        "status": "READY",
        "is_ready": True,
        "model_version": manifest.get("model_version", "1.0.0"),
        "artifacts_directory": str(predictor.artifacts_dir),
        "manifest_present": True,
        "base_models": manifest.get("base_models", {}),
        "ensemble": manifest.get("ensemble", {}),
        "risk_thresholds": manifest.get("risk_thresholds", {}),
        "sha256_checksums": manifest.get("sha256_checksums", {}),
        "message": "All Phase 2 base models, stacking meta-model, and calibrator are ready for inference.",
    }


@router.get("/models/metrics")
def get_model_metrics():
    """Retrieve comprehensive evaluation metrics on untouched test set."""
    metrics_path = REP / "evaluation" / "final_metrics.json"
    if not metrics_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation metrics report not found.",
        )
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics


@router.post("/predict/provider", response_model=PredictionResponse)
def predict_provider_fraud(payload: ProviderFeaturePayload, db: Session = Depends(get_db)):
    """
    Score a single healthcare provider for potential fraudulent activity
    using the calibrated stacking ensemble and XGBoost SHAP explainability.
    """
    predictor = get_predictor()
    if not predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Prediction models are not ready. Verification error: {predictor.verification_error}",
        )

    features = payload.features
    seq = payload.sequence
    mask = payload.sequence_mask

    # If any required modality is omitted from payload, retrieve from stored dataset
    if features is None or seq is None or mask is None:
        data = predictor.get_provider_data(payload.provider_id)
        if data is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider '{payload.provider_id}' not found in processed records and complete features/sequence/mask were not provided.",
            )
        stored_feat, stored_seq, stored_mask = data
        if features is None:
            features = stored_feat
        if seq is None:
            seq = stored_seq
        if mask is None:
            mask = stored_mask

    try:
        result = predictor.predict_provider(
            provider_id=payload.provider_id,
            features=features,
            sequence=seq,
            mask=mask,
            include_shap=True,
        )
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch_providers(request: BatchPredictionRequest, db: Session = Depends(get_db)):
    """
    Run batch fraud evaluation across multiple providers.
    Explicitly reconciles total_uploaded = analysed_count + rejected_count,
    returning individual rejected provider IDs and reasons.
    """
    predictor = get_predictor()
    if not predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Prediction models are not ready. Verification error: {predictor.verification_error}",
        )

    total_uploaded = len(request.provider_ids)
    predictions = []
    rejected_providers = []

    for pid in request.provider_ids:
        if not pid or not str(pid).strip():
            rejected_providers.append({
                "provider_id": str(pid) if pid is not None else "",
                "reason": "Empty or blank Provider ID.",
            })
            continue

        clean_pid = str(pid).strip()
        data = predictor.get_provider_data(clean_pid)
        if data is None:
            rejected_providers.append({
                "provider_id": clean_pid,
                "reason": f"Provider '{clean_pid}' not found in processed records and no features/sequence supplied.",
            })
            continue

        feat, seq, mask = data
        if seq is None or mask is None:
            rejected_providers.append({
                "provider_id": clean_pid,
                "reason": f"Provider '{clean_pid}' lacks valid 12-month temporal sequence or mask.",
            })
            continue

        try:
            pred = predictor.predict_provider(
                provider_id=clean_pid,
                features=feat,
                sequence=seq,
                mask=mask,
                include_shap=False,
            )
            predictions.append(pred)
        except Exception as e:
            rejected_providers.append({
                "provider_id": clean_pid,
                "reason": f"Prediction failed during inference: {str(e)}",
            })

    analysed_count = len(predictions)
    rejected_count = len(rejected_providers)

    # Enforce exact reconciliation
    if total_uploaded != (analysed_count + rejected_count):
        raise RuntimeError(
            f"Batch reconciliation error: total_uploaded ({total_uploaded}) != "
            f"analysed_count ({analysed_count}) + rejected_count ({rejected_count})"
        )

    high = sum(1 for p in predictions if p["risk_tier"] == "HIGH")
    med = sum(1 for p in predictions if p["risk_tier"] == "MEDIUM")
    low = sum(1 for p in predictions if p["risk_tier"] == "LOW")

    return {
        "predictions": predictions,
        "total_uploaded": total_uploaded,
        "analysed_count": analysed_count,
        "rejected_count": rejected_count,
        "rejected_providers": rejected_providers,
        "total_evaluated": analysed_count,
        "high_risk_count": high,
        "medium_risk_count": med,
        "low_risk_count": low,
        "disclaimer": predictor.DISCLAIMER,
    }


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
