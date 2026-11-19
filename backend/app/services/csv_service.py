"""
HealthGuard AI - CSV Upload Validation & Orchestration Service
"""

import io
import math
from typing import Tuple, List, Optional, Dict, Any, Set
from datetime import datetime, date
import pandas as pd
from fastapi import UploadFile, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session

from backend.app.db.models import CSVUpload, Claim, User
from backend.app.db.schemas import CSVPreviewResponse, CSVPreviewRow
from backend.app.workers.csv_worker import process_csv_in_background


class CSVService:
    """Validates CSV files, creates upload tracking records, and queues background processing."""

    MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB
    REQUIRED_COLUMNS = ["ClaimID", "ProviderID", "InscClaimAmtReimbursed", "ClaimStartDt"]

    # Canonical mapping: lowercase alias -> canonical field name
    CANONICAL_MAPPING: Dict[str, str] = {
        "claimid": "ClaimID",
        "claim_id": "ClaimID",
        "claim id": "ClaimID",
        "clm_id": "ClaimID",
        "clmid": "ClaimID",

        "providerid": "ProviderID",
        "provider_id": "ProviderID",
        "provider": "ProviderID",
        "prv_id": "ProviderID",
        "prvid": "ProviderID",

        "beneid": "BeneID",
        "bene_id": "BeneID",
        "beneficiaryid": "BeneID",
        "beneficiary_id": "BeneID",
        "beneficiary": "BeneID",
        "patientid": "BeneID",
        "patient_id": "BeneID",

        "inscclaimamtreimbursed": "InscClaimAmtReimbursed",
        "reimbursement": "InscClaimAmtReimbursed",
        "reimbursement_amount": "InscClaimAmtReimbursed",
        "reimbursementamount": "InscClaimAmtReimbursed",
        "claim_amount": "InscClaimAmtReimbursed",
        "claimamount": "InscClaimAmtReimbursed",
        "amount": "InscClaimAmtReimbursed",
        "insc_claim_amt_reimbursed": "InscClaimAmtReimbursed",

        "claimstartdt": "ClaimStartDt",
        "claim_start_date": "ClaimStartDt",
        "claimstartdate": "ClaimStartDt",
        "claim_date": "ClaimStartDt",
        "claimdate": "ClaimStartDt",
        "service_date": "ClaimStartDt",
        "servicedt": "ClaimStartDt",

        "claimenddt": "ClaimEndDt",
        "claim_end_date": "ClaimEndDt",
        "claimenddate": "ClaimEndDt",

        "admissiondt": "AdmissionDt",
        "admission_date": "AdmissionDt",
        "admissiondate": "AdmissionDt",
        "admit_date": "AdmissionDt",

        "dischargedt": "DischargeDt",
        "discharge_date": "DischargeDt",
        "dischargedate": "DischargeDt",

        "claimtype": "ClaimType",
        "claim_type": "ClaimType",
        "type": "ClaimType",

        "deductibleamtpaid": "DeductibleAmtPaid",
        "deductible": "DeductibleAmtPaid",
        "deductible_amount": "DeductibleAmtPaid",
        "deductibleamount": "DeductibleAmtPaid",
        "ded_paid": "DeductibleAmtPaid",

        "currency": "Currency",
        "curr": "Currency",

        "attendingphysician": "AttendingPhysician",
        "attending_physician": "AttendingPhysician",
        "attending": "AttendingPhysician",

        "operatingphysician": "OperatingPhysician",
        "operating_physician": "OperatingPhysician",
        "operating": "OperatingPhysician",

        "otherphysician": "OtherPhysician",
        "other_physician": "OtherPhysician",
        "other": "OtherPhysician",

        "clmadmitdiagnosiscode": "ClmAdmitDiagnosisCode",
        "admit_diagnosis_code": "ClmAdmitDiagnosisCode",
        "admitdiagnosiscode": "ClmAdmitDiagnosisCode",

        "dod": "DOD",
        "date_of_death": "DOD",
        "dateofdeath": "DOD",
    }

    # Add diagnosis and procedure codes
    for i in range(1, 11):
        CANONICAL_MAPPING[f"clmdiagnosiscode_{i}"] = f"ClmDiagnosisCode_{i}"
        CANONICAL_MAPPING[f"clmdiagnosiscode{i}"] = f"ClmDiagnosisCode_{i}"
        CANONICAL_MAPPING[f"diagnosis_code_{i}"] = f"ClmDiagnosisCode_{i}"
        CANONICAL_MAPPING[f"diag_{i}"] = f"ClmDiagnosisCode_{i}"
        CANONICAL_MAPPING[f"diag{i}"] = f"ClmDiagnosisCode_{i}"

    for i in range(1, 7):
        CANONICAL_MAPPING[f"clmprocedurecode_{i}"] = f"ClmProcedureCode_{i}"
        CANONICAL_MAPPING[f"clmprocedurecode{i}"] = f"ClmProcedureCode_{i}"
        CANONICAL_MAPPING[f"procedure_code_{i}"] = f"ClmProcedureCode_{i}"
        CANONICAL_MAPPING[f"proc_{i}"] = f"ClmProcedureCode_{i}"
        CANONICAL_MAPPING[f"proc{i}"] = f"ClmProcedureCode_{i}"

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renames DataFrame columns using canonical alias mapping."""
        rename_map = {}
        for col in df.columns:
            cleaned = str(col).strip().lower().replace(" ", "_")
            if cleaned in self.CANONICAL_MAPPING:
                rename_map[col] = self.CANONICAL_MAPPING[cleaned]
            elif cleaned.replace("_", "") in self.CANONICAL_MAPPING:
                rename_map[col] = self.CANONICAL_MAPPING[cleaned.replace("_", "")]
        return df.rename(columns=rename_map)

    def generate_csv_template(self, currency: str = "INR") -> str:
        """Generate schema-conforming CSV template with 2 representative sample rows."""
        cur = (currency or "INR").strip().upper()
        if cur not in ("INR", "USD"):
            cur = "INR"

        headers = [
            "ClaimID",
            "ProviderID",
            "BeneID",
            "ClaimType",
            "InscClaimAmtReimbursed",
            "DeductibleAmtPaid",
            "Currency",
            "ClaimStartDt",
            "ClaimEndDt",
            "AdmissionDt",
            "DischargeDt",
            "AttendingPhysician",
            "OperatingPhysician",
            "OtherPhysician",
            "ClmDiagnosisCode_1",
            "ClmDiagnosisCode_2",
            "ClmAdmitDiagnosisCode",
            "ClmProcedureCode_1",
        ]

        if cur == "INR":
            sample_1 = [
                "CLM_INP_1001",
                "PRV51001",
                "BENE10001",
                "Inpatient",
                "350000.00",
                "8000.00",
                "INR",
                "2009-03-01",
                "2009-03-05",
                "2009-03-01",
                "2009-03-05",
                "PHY31201",
                "PHY42109",
                "PHY53110",
                "4019",
                "25000",
                "4019",
                "3722",
            ]
            sample_2 = [
                "CLM_OUT_1002",
                "PRV51001",
                "BENE10002",
                "Outpatient",
                "12500.00",
                "1000.00",
                "INR",
                "2009-03-10",
                "2009-03-10",
                "",
                "",
                "PHY10001",
                "",
                "",
                "25000",
                "",
                "",
                "",
            ]
        else:
            sample_1 = [
                "CLM_INP_1001",
                "PRV51001",
                "BENE10001",
                "Inpatient",
                "4200.00",
                "100.00",
                "USD",
                "2009-03-01",
                "2009-03-05",
                "2009-03-01",
                "2009-03-05",
                "PHY31201",
                "PHY42109",
                "PHY53110",
                "4019",
                "25000",
                "4019",
                "3722",
            ]
            sample_2 = [
                "CLM_OUT_1002",
                "PRV51001",
                "BENE10002",
                "Outpatient",
                "150.00",
                "12.00",
                "USD",
                "2009-03-10",
                "2009-03-10",
                "",
                "",
                "PHY10001",
                "",
                "",
                "25000",
                "",
                "",
                "",
            ]

        rows = [",".join(headers), ",".join(sample_1), ",".join(sample_2)]
        return "\n".join(rows) + "\n"

    async def preview_csv(self, file: UploadFile, db: Session) -> CSVPreviewResponse:
        """Inspect and validate CSV file for pre-upload confirmation."""
        filename = file.filename or "unknown.csv"
        if not filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file format. Uploaded file must have a .csv extension."
            )

        content = await file.read()
        await file.seek(0)

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

        try:
            df_raw = pd.read_csv(io.BytesIO(content), low_memory=False)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Malformed CSV content: {e}"
            )

        df = self.normalize_columns(df_raw)
        detected_columns = list(df.columns)

        missing_required = [req for req in self.REQUIRED_COLUMNS if req not in df.columns]

        # Query existing claim IDs from DB
        existing_claim_rows = db.query(Claim.claim_id).all()
        existing_db_cids: Set[str] = {str(r[0]) for r in existing_claim_rows if r[0]}

        total_rows = len(df)
        valid_count = 0
        invalid_count = 0
        duplicate_count = 0
        warnings: List[str] = []

        if missing_required:
            warnings.append(f"Missing required columns: {', '.join(missing_required)}")

        seen_batch_cids: Set[str] = set()
        preview_rows: List[CSVPreviewRow] = []

        today = date.today()

        for idx, row in df.iterrows():
            row_num = idx + 1
            row_issues: List[str] = []
            row_status = "VALID"

            # Check ClaimID
            cid_val = row.get("ClaimID")
            cid_str = str(cid_val).strip() if pd.notna(cid_val) and str(cid_val).strip() else None

            if not cid_str:
                row_issues.append("Missing ClaimID")
                row_status = "INVALID"
            elif cid_str in existing_db_cids:
                row_issues.append(f"ClaimID '{cid_str}' already exists in database")
                row_status = "DUPLICATE"
                duplicate_count += 1
            elif cid_str in seen_batch_cids:
                row_issues.append(f"Duplicate ClaimID '{cid_str}' in batch")
                row_status = "DUPLICATE"
                duplicate_count += 1
            else:
                seen_batch_cids.add(cid_str)

            # Check ProviderID
            pid_val = row.get("ProviderID")
            pid_str = str(pid_val).strip() if pd.notna(pid_val) and str(pid_val).strip() else None
            if not pid_str:
                row_issues.append("Missing ProviderID")
                if row_status != "DUPLICATE":
                    row_status = "INVALID"

            # Check Reimbursement
            reimb_val = row.get("InscClaimAmtReimbursed")
            reimb_num = None
            if pd.isna(reimb_val) or reimb_val is None or str(reimb_val).strip() == "":
                row_issues.append("Missing reimbursement amount")
                if row_status != "DUPLICATE":
                    row_status = "INVALID"
            else:
                try:
                    reimb_num = float(reimb_val)
                    if math.isnan(reimb_num) or not math.isfinite(reimb_num) or reimb_num < 0:
                        row_issues.append("Invalid reimbursement amount (must be >= 0)")
                        if row_status != "DUPLICATE":
                            row_status = "INVALID"
                except Exception:
                    row_issues.append(f"Non-numeric reimbursement amount: '{reimb_val}'")
                    if row_status != "DUPLICATE":
                        row_status = "INVALID"

            # Check ClaimStartDt
            sdt_val = row.get("ClaimStartDt")
            sdt_str = str(sdt_val).strip() if pd.notna(sdt_val) and str(sdt_val).strip() else None
            if not sdt_str:
                row_issues.append("Missing ClaimStartDt")
                if row_status != "DUPLICATE":
                    row_status = "INVALID"
            else:
                try:
                    parsed_sdt = pd.to_datetime(sdt_str).date()
                    if parsed_sdt > today:
                        row_issues.append(f"Future ClaimStartDt ({sdt_str}) not permitted")
                        if row_status != "DUPLICATE":
                            row_status = "INVALID"
                except Exception:
                    row_issues.append(f"Invalid date format for ClaimStartDt: '{sdt_str}'")
                    if row_status != "DUPLICATE":
                        row_status = "INVALID"

            if row_status == "VALID":
                valid_count += 1
            elif row_status == "INVALID":
                invalid_count += 1

            # Keep preview for up to 10 rows
            if idx < 10:
                bene_val = row.get("BeneID")
                bene_str = str(bene_val).strip() if pd.notna(bene_val) and str(bene_val).strip() else None
                ctype_val = row.get("ClaimType")
                ctype_str = str(ctype_val).strip() if pd.notna(ctype_val) and str(ctype_val).strip() else "Outpatient"
                curr_val = row.get("Currency")
                curr_str = str(curr_val).strip().upper() if pd.notna(curr_val) and str(curr_val).strip() else "USD"

                preview_rows.append(
                    CSVPreviewRow(
                        row_number=row_num,
                        claim_id=cid_str,
                        provider_id=pid_str,
                        beneficiary_id=bene_str,
                        claim_type=ctype_str,
                        reimbursement=reimb_num,
                        currency=curr_str,
                        claim_start_date=sdt_str,
                        status=row_status,
                        issues=row_issues,
                    )
                )

        return CSVPreviewResponse(
            filename=filename,
            total_rows=total_rows,
            valid_rows_count=valid_count,
            invalid_rows_count=invalid_count,
            duplicate_rows_count=duplicate_count,
            detected_columns=detected_columns,
            missing_required_columns=missing_required,
            preview_rows=preview_rows,
            warnings=warnings,
        )

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

        # Preliminary header check with normalization
        try:
            sample_df = pd.read_csv(io.BytesIO(content), nrows=5)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Malformed CSV content: {e}"
            )

        sample_norm = self.normalize_columns(sample_df)
        missing_cols = [req for req in self.REQUIRED_COLUMNS if req not in sample_norm.columns]

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
