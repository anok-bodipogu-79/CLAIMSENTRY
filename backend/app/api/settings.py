"""
HealthGuard AI - System Settings Endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text

from backend.app.core.config import settings
from backend.app.core.dependencies import get_current_user, get_current_admin
from backend.app.db.database import get_db, engine
from backend.app.db.models import User
from backend.app.db.schemas import SettingsResponse, SettingsUpdateRequest, ModelVersionSwitchRequest
from backend.app.services.prediction_service import prediction_service

router = APIRouter(prefix="/settings", tags=["Settings"])


@router.get("", response_model=SettingsResponse)
def get_settings(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve current system configuration, active ML model version, and database connectivity."""
    db_connected = False
    db_type = "unknown"
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            db_connected = True
            db_type = engine.name  # 'postgresql' or 'sqlite'
    except Exception:
        db_connected = False

    readiness = prediction_service.get_readiness()

    return SettingsResponse(
        model_version=settings.MODEL_VERSION,
        low_medium_threshold=settings.DEFAULT_LOW_MEDIUM_THRESHOLD,
        medium_high_threshold=settings.DEFAULT_MEDIUM_HIGH_THRESHOLD,
        database_connected=db_connected,
        database_type=db_type,
        models_ready=readiness["is_ready"],
        disclaimer=settings.REGULATORY_DISCLAIMER,
    )


@router.put("", response_model=SettingsResponse)
def update_settings(
    payload: SettingsUpdateRequest,
    db: Session = Depends(get_db),
    current_admin: User = Depends(get_current_admin),
):
    """Safely update application settings and risk thresholds. Restricted to administrators."""
    # Check model version update
    if payload.model_version is not None:
        try:
            prediction_service.set_model_version(payload.model_version)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to switch model version: {e}",
            )

    # Check threshold updates
    if payload.low_medium_threshold is not None and payload.medium_high_threshold is not None:
        if not (0.0 < payload.low_medium_threshold < payload.medium_high_threshold < 1.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Thresholds must satisfy 0 < low_medium < medium_high < 1.",
            )
        settings.DEFAULT_LOW_MEDIUM_THRESHOLD = payload.low_medium_threshold
        settings.DEFAULT_MEDIUM_HIGH_THRESHOLD = payload.medium_high_threshold
    elif payload.low_medium_threshold is not None:
        if not (0.0 < payload.low_medium_threshold < settings.DEFAULT_MEDIUM_HIGH_THRESHOLD):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"low_medium_threshold must be strictly less than medium_high_threshold ({settings.DEFAULT_MEDIUM_HIGH_THRESHOLD}).",
            )
        settings.DEFAULT_LOW_MEDIUM_THRESHOLD = payload.low_medium_threshold
    elif payload.medium_high_threshold is not None:
        if not (settings.DEFAULT_LOW_MEDIUM_THRESHOLD < payload.medium_high_threshold < 1.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"medium_high_threshold must be strictly greater than low_medium_threshold ({settings.DEFAULT_LOW_MEDIUM_THRESHOLD}).",
            )
        settings.DEFAULT_MEDIUM_HIGH_THRESHOLD = payload.medium_high_threshold

    return get_settings(db=db, current_user=current_admin)


@router.post("/model-version", response_model=SettingsResponse)
def switch_model_version(
    payload: ModelVersionSwitchRequest,
    db: Session = Depends(get_db),
    current_admin: User = Depends(get_current_admin),
):
    """Switch active ML model version (v1 or v2). Restricted to administrators."""
    try:
        prediction_service.set_model_version(payload.version)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to switch model version: {e}",
        )
    return get_settings(db=db, current_user=current_admin)
