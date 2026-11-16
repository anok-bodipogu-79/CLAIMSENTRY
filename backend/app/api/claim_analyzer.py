"""
HealthGuard AI - Single Claim Analyzer Endpoint
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.core.dependencies import get_current_user
from backend.app.db.database import get_db
from backend.app.db.models import User
from backend.app.db.schemas import SingleClaimRequest, SingleClaimResponse
from backend.app.services.claim_analysis_pipeline import (
    claim_analysis_pipeline,
    AnalysisValidationError,
)

router = APIRouter(prefix="/analyzer", tags=["Claim Analyzer"])


@router.post("/single", response_model=SingleClaimResponse)
def analyze_single_claim(
    payload: SingleClaimRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Analyze a single healthcare claim against provider history and training peer benchmarks.
    Executes the genuine multi-modal ML ensemble, runs rule-based anomaly detection,
    validates response invariants, and returns explanations and investigation guidance
    via the shared universal analysis pipeline.
    """
    try:
        return claim_analysis_pipeline.analyze_single_claim(
            payload=payload,
            db=db,
            current_user=current_user,
        )
    except AnalysisValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Claim analysis failed: {e}",
        )
