"""
HealthGuard AI - Batch CSV Upload & Analysis Endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, status
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.app.core.dependencies import get_current_user
from backend.app.db.database import get_db
from backend.app.db.models import User, CSVUpload, CSVUploadRow, Claim, ClaimAnalysis
from backend.app.db.schemas import (
    CSVUploadResponse,
    CSVUploadDetail,
    PaginatedCSVRowsResponse,
    CSVUploadRowResponse,
    PaginatedClaimsResponse,
    ClaimListItem,
)
from backend.app.services.csv_service import csv_service

router = APIRouter(prefix="/analyzer/csv", tags=["CSV Analysis"])


@router.post("", response_model=CSVUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_csv_claims(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Upload a batch CSV file of claims for asynchronous processing and fraud risk assessment.
    Validates file format, required headers, and queues background prediction worker.
    """
    upload_record = await csv_service.validate_and_queue(
        file=file,
        current_user=current_user,
        db=db,
        background_tasks=background_tasks,
    )

    return CSVUploadResponse(
        upload_id=upload_record.id,
        filename=upload_record.filename,
        status=upload_record.status,
        total_rows=upload_record.total_rows,
        successful_rows=upload_record.successful_rows,
        rejected_rows=upload_record.rejected_rows,
        message="CSV upload received and queued for background evaluation.",
    )


@router.get("/{upload_id}", response_model=CSVUploadDetail)
def get_upload_status(
    upload_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Check processing status and progress counts for a CSV batch upload."""
    upload = db.query(CSVUpload).filter(CSVUpload.id == upload_id).first()
    if not upload:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"CSV upload ID {upload_id} not found.",
        )

    # Ownership check: only owner or administrator may access upload status
    if current_user.role.lower() != "admin" and upload.user_id is not None and upload.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: You do not have permission to view this upload.",
        )

    # Verify reconciliation invariant if completed
    reconciled = (
        (upload.successful_rows + upload.rejected_rows == upload.total_rows)
        if upload.status == "COMPLETED"
        else True
    )

    return CSVUploadDetail(
        upload_id=upload.id,
        filename=upload.filename,
        status=upload.status,
        total_rows=upload.total_rows,
        successful_rows=upload.successful_rows,
        rejected_rows=upload.rejected_rows,
        reconciliation_verified=reconciled,
        created_at=upload.created_at,
        completed_at=upload.completed_at,
        error_message=upload.error_message,
    )


@router.get("/{upload_id}/results", response_model=PaginatedClaimsResponse)
def get_upload_results(
    upload_id: int,
    page: int = 1,
    page_size: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve successfully analysed claims from this CSV upload."""
    upload = db.query(CSVUpload).filter(CSVUpload.id == upload_id).first()
    if not upload:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"CSV upload ID {upload_id} not found.",
        )

    # Ownership check: only owner or administrator may access upload status
    if current_user.role.lower() != "admin" and upload.user_id is not None and upload.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: You do not have permission to view this upload.",
        )

    # Get claim IDs successfully processed in this upload
    success_rows = (
        db.query(CSVUploadRow.claim_id)
        .filter(
            CSVUploadRow.csv_upload_id == upload_id,
            CSVUploadRow.processing_status == "SUCCESS",
            CSVUploadRow.claim_id.isnot(None),
        )
        .all()
    )
    claim_ids = [r[0] for r in success_rows if r[0]]

    query = db.query(Claim).filter(Claim.claim_id.in_(claim_ids))
    total = query.count()
    offset = (page - 1) * page_size
    claims = query.offset(offset).limit(page_size).all()

    items = []
    for c in claims:
        analysis = c.analyses[0] if c.analyses else None
        flags = analysis.anomaly_flags if analysis else []
        items.append(
            ClaimListItem(
                claim_id=c.claim_id,
                provider_id=c.provider_id,
                beneficiary_id=c.beneficiary_id,
                claim_type=c.claim_type,
                claim_date=c.claim_date,
                reimbursement_amount=c.reimbursement_amount,
                anomaly_flags=flags,
                has_anomalies=bool(flags),
            )
        )

    total_pages = (total + page_size - 1) // page_size if total > 0 else 1
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "claims": items,
    }


@router.get("/{upload_id}/rejected-rows", response_model=PaginatedCSVRowsResponse)
def get_upload_rejected_rows(
    upload_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve all rejected rows with explicit validation failure reasons."""
    upload = db.query(CSVUpload).filter(CSVUpload.id == upload_id).first()
    if not upload:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"CSV upload ID {upload_id} not found.",
        )

    # Ownership check: only owner or administrator may access upload status
    if current_user.role.lower() != "admin" and upload.user_id is not None and upload.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: You do not have permission to view this upload.",
        )

    rejected_rows = (
        db.query(CSVUploadRow)
        .filter(
            CSVUploadRow.csv_upload_id == upload_id,
            CSVUploadRow.processing_status == "REJECTED",
        )
        .order_by(CSVUploadRow.row_number)
        .all()
    )

    items = [
        CSVUploadRowResponse(
            row_number=r.row_number,
            claim_id=r.claim_id,
            processing_status=r.processing_status,
            rejection_reason=r.rejection_reason,
        )
        for r in rejected_rows
    ]

    return PaginatedCSVRowsResponse(
        upload_id=upload_id,
        total_rejected=len(items),
        rejected_rows=items,
    )
