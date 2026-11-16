"""
HealthGuard AI - CSV Upload Validation & Orchestration Service
"""

import io
from typing import Tuple, List, Optional
import pandas as pd
from fastapi import UploadFile, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session

from backend.app.db.models import CSVUpload, User
from backend.app.workers.csv_worker import process_csv_in_background


class CSVService:
    """Validates CSV files, creates upload tracking records, and queues background processing."""

    MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB
    REQUIRED_COLUMNS = ["ClaimID", "ProviderID", "InscClaimAmtReimbursed", "ClaimStartDt"]

    async def validate_and_queue(
        self,
        file: UploadFile,
        current_user: Optional[User],
        db: Session,
        background_tasks: BackgroundTasks,
    ) -> CSVUpload:
        """Validate uploaded CSV file and schedule background inference."""
        filename = file.filename or "unknown.csv"
        if not filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file format. Uploaded file must have a .csv extension."
            )

        content = await file.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded CSV file is empty."
            )
        if len(content) > self.MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of {self.MAX_FILE_SIZE_BYTES // (1024*1024)}MB."
            )

        # Preliminary header check
        try:
            sample_df = pd.read_csv(io.BytesIO(content), nrows=5)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Malformed CSV content: {e}"
            )

        col_map = {c.lower(): c for c in sample_df.columns}
        missing_cols = []
        for req in ["ClaimID", "InscClaimAmtReimbursed", "ClaimStartDt"]:
            if req.lower() not in col_map:
                missing_cols.append(req)
        if "providerid" not in col_map and "provider" not in col_map:
            missing_cols.append("ProviderID")
        if missing_cols:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Missing required CSV columns: {missing_cols}. Required: {self.REQUIRED_COLUMNS}"
            )

        # Create upload record
        upload_record = CSVUpload(
            user_id=current_user.id if current_user else None,
            filename=filename,
            total_rows=0,
            successful_rows=0,
            rejected_rows=0,
            status="PENDING",
        )
        db.add(upload_record)
        db.commit()
        db.refresh(upload_record)

        # Queue background worker
        background_tasks.add_task(process_csv_in_background, upload_record.id, content)

        return upload_record


csv_service = CSVService()
