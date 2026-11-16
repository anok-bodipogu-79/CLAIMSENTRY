"""
HealthGuard AI - CSV Processing Background Worker
"""

import io
import logging
from typing import Optional, Dict, Any, List, Set
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy.orm import Session

from backend.app.db.database import SessionLocal
from backend.app.db.models import CSVUpload, CSVUploadRow, Provider, Claim, Beneficiary
from backend.app.services.prediction_service import prediction_service
from backend.app.services.anomaly_service import anomaly_service
from backend.app.services.explanation_service import explanation_service
from backend.app.core.config import settings

logger = logging.getLogger(__name__)


def _parse_dt(val: Any) -> Optional[datetime]:
    """Safely parse date values to native datetime objects."""
    if pd.isna(val) or val is None or str(val).strip() == "":
        return None
    try:
        dt = pd.to_datetime(val)
        return dt.to_pydatetime()
    except Exception:
        return None


def _extract_codes(row: Any, prefix: str, max_count: int, fallback_key: str) -> List[str]:
    """Extract ordered diagnosis or procedure codes from row."""
    codes: List[str] = []
    for i in range(1, max_count + 1):
        v = row.get(f"{prefix}_{i}")
        if pd.notna(v) and str(v).strip():
            codes.append(str(v).strip())
    if not codes and fallback_key in row and pd.notna(row.get(fallback_key)):
        fb = row.get(fallback_key)
        if isinstance(fb, list):
            codes = [str(x).strip() for x in fb if str(x).strip()]
        elif isinstance(fb, str):
            codes = [str(x).strip() for x in fb.split(",") if str(x).strip()]
    return codes


def process_csv_in_background(upload_id: int, file_content: bytes, db: Optional[Session] = None):
    """
    Background worker task to parse, validate, predict, and reconcile uploaded CSV claims.
    Enforces strict invariant: successful_rows + rejected_rows == total_rows.
    Preserves actual claim types, claim dates, beneficiary IDs, reimbursement, and deductible amounts.
    """
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True

    try:
        upload = db.query(CSVUpload).filter(CSVUpload.id == upload_id).first()
        if not upload:
            return

        upload.status = "PROCESSING"
        db.commit()

        # Read CSV from memory buffer
        df_raw = pd.read_csv(io.BytesIO(file_content), low_memory=False)
        total_rows = len(df_raw)
        upload.total_rows = total_rows

        # Normalize Provider column if ProviderID is missing
        if "ProviderID" not in df_raw.columns and "Provider" in df_raw.columns:
            df_raw["ProviderID"] = df_raw["Provider"]

        if total_rows == 0:
            upload.status = "COMPLETED"
            upload.successful_rows = 0
            upload.rejected_rows = 0
            upload.completed_at = datetime.now(timezone.utc)
            db.commit()
            return

        # Check for pre-existing claims in database and in-batch duplicates
        existing_claim_rows = db.query(Claim.claim_id).all()
        existing_db_cids = {str(r[0]) for r in existing_claim_rows if r[0]}

        seen_batch_cids: Set[str] = set()
        pre_rejected_indices: Set[int] = set()
        pre_rejection_records: List[Dict[str, Any]] = []
        valid_row_indices: List[int] = []

        for idx, row in df_raw.iterrows():
            cid_val = row.get("ClaimID", row.get("claim_id"))
            cid_str = str(cid_val).strip() if pd.notna(cid_val) else None

            if not cid_str:
                pre_rejected_indices.add(idx)
                pre_rejection_records.append({
                    "row_index": idx,
                    "claim_id": None,
                    "rejection_reason": "Missing ClaimID in row",
                })
            elif cid_str in existing_db_cids:
                pre_rejected_indices.add(idx)
                pre_rejection_records.append({
                    "row_index": idx,
                    "claim_id": cid_str,
                    "rejection_reason": f"ClaimID '{cid_str}' already exists in database",
                })
            elif cid_str in seen_batch_cids:
                pre_rejected_indices.add(idx)
                pre_rejection_records.append({
                    "row_index": idx,
                    "claim_id": cid_str,
                    "rejection_reason": f"Duplicate ClaimID '{cid_str}' within uploaded batch",
                })
            else:
                seen_batch_cids.add(cid_str)
                valid_row_indices.append(idx)

        analysed_claims_cnt = 0
        pipeline_rejected_cnt = 0
        pipeline_rejection_records: List[Dict[str, Any]] = []
        provider_results: Dict[str, Any] = {}
        claims_result: List[Dict[str, Any]] = []

        if valid_row_indices:
            df_valid = df_raw.iloc[valid_row_indices].copy().reset_index(drop=True)
            pipeline_result = prediction_service.predict_claims_pipeline(df_valid)
            analysed_claims_cnt = pipeline_result.get("analysed_claims", 0)
            pipeline_rejected_cnt = pipeline_result.get("rejected_rows", 0)
            pipeline_rejection_records = pipeline_result.get("rejected_records", [])
            provider_results = pipeline_result.get("provider_results", {})
            claims_result = pipeline_result.get("claims_result", [])

        total_successful = analysed_claims_cnt
        total_rejected = len(pre_rejection_records) + pipeline_rejected_cnt

        # Strict reconciliation invariant check
        assert total_successful + total_rejected == total_rows, (
            f"Reconciliation invariant violated: {total_successful} + {total_rejected} != {total_rows}"
        )

        # 1. Record pre-rejection records
        for rej in pre_rejection_records:
            r_idx = int(rej.get("row_index", 0))
            cid = rej.get("claim_id", f"ROW_{r_idx}")
            reason = rej.get("rejection_reason", "Validation error")
            row_rec = CSVUploadRow(
                csv_upload_id=upload_id,
                row_number=r_idx + 1,
                claim_id=str(cid) if cid else None,
                processing_status="REJECTED",
                rejection_reason=reason,
            )
            db.add(row_rec)

        # 2. Record pipeline rejection records (mapped to original df_raw index)
        pipeline_rejected_orig_indices: Set[int] = set()
        for rej in pipeline_rejection_records:
            sub_idx = int(rej.get("row_index", 0))
            orig_idx = valid_row_indices[sub_idx] if sub_idx < len(valid_row_indices) else sub_idx
            pipeline_rejected_orig_indices.add(orig_idx)
            cid = rej.get("ClaimID", rej.get("claim_id", f"ROW_{orig_idx}"))
            reason = rej.get("rejection_reason", rej.get("reason", "Unknown validation error"))
            row_rec = CSVUploadRow(
                csv_upload_id=upload_id,
                row_number=orig_idx + 1,
                claim_id=str(cid) if cid else None,
                processing_status="REJECTED",
                rejection_reason=reason,
            )
            db.add(row_rec)

        # 3. Record itemized successful rows
        all_rejected_indices = pre_rejected_indices.union(pipeline_rejected_orig_indices)
        for idx in valid_row_indices:
            if idx not in all_rejected_indices:
                row = df_raw.iloc[idx]
                cid = str(row.get("ClaimID", row.get("claim_id", f"ROW_{idx}")))
                row_rec = CSVUploadRow(
                    csv_upload_id=upload_id,
                    row_number=idx + 1,
                    claim_id=cid,
                    processing_status="SUCCESS",
                    rejection_reason=None,
                )
                db.add(row_rec)

        # 4. Persist provider risk scores and explanations
        provider_score_map = {}
        for pid_str, p_eval in provider_results.items():
            prs = prediction_service.persist_provider_assessment(
                db=db,
                provider_id=pid_str,
                eval_result=p_eval,
                model_version_str=settings.MODEL_VERSION,
            )
            provider_score_map[pid_str] = prs.id

            expl_data = explanation_service.format_explanation(p_eval)
            explanation_service.persist_explanation(
                db=db,
                provider_risk_score_id=prs.id,
                claim_analysis_id=None,
                explanation_data=expl_data,
            )

        # Build fast lookup from ClaimID to raw row for rich metadata persistence
        cid_to_raw_row: Dict[str, Any] = {}
        for _, r in df_raw.iterrows():
            c_val = r.get("ClaimID", r.get("claim_id"))
            if pd.notna(c_val):
                cid_to_raw_row[str(c_val).strip()] = r

        # 5. Persist rich claim records and claim anomaly analyses
        for ca in claims_result:
            cid = str(ca.get("claim_id", "")).strip()
            pid = str(ca.get("provider_id", "")).strip()
            if not cid or not pid:
                continue

            row_data = cid_to_raw_row.get(cid)
            if row_data is not None:
                # Actual claim type preservation
                raw_ct = row_data.get("ClaimType", row_data.get("claim_type"))
                adm_dt = _parse_dt(row_data.get("AdmissionDt", row_data.get("admission_date")))
                dis_dt = _parse_dt(row_data.get("DischargeDt", row_data.get("discharge_date")))
                if pd.notna(raw_ct) and str(raw_ct).strip():
                    ct_str = str(raw_ct).strip()
                    claim_type = ct_str.capitalize() if ct_str.lower() in ("inpatient", "outpatient") else ct_str
                elif adm_dt is not None or dis_dt is not None:
                    claim_type = "Inpatient"
                else:
                    claim_type = "Outpatient"

                clm_dt = _parse_dt(row_data.get("ClaimStartDt", row_data.get("ClaimStartDate", row_data.get("claim_date"))))
                clm_end_dt = _parse_dt(row_data.get("ClaimEndDt", row_data.get("ClaimEndDate", row_data.get("claim_end_date"))))

                reimb_val = row_data.get("InscClaimAmtReimbursed", ca.get("reimbursement", 0.0))
                reimb_amt = float(reimb_val) if pd.notna(reimb_val) else 0.0
                deduct_val = row_data.get("DeductibleAmtPaid", row_data.get("deductible_amount", 0.0))
                deduct_amt = float(deduct_val) if pd.notna(deduct_val) else 0.0

                if adm_dt and dis_dt:
                    los = max(0, (dis_dt - adm_dt).days)
                else:
                    los_val = row_data.get("inpatient_los", 0)
                    try:
                        los = int(los_val) if pd.notna(los_val) else 0
                    except Exception:
                        los = 0

                bene_val = row_data.get("BeneID", row_data.get("beneficiary_id"))
                bene_id = str(bene_val).strip() if pd.notna(bene_val) and str(bene_val).strip() else None

                att_val = row_data.get("AttendingPhysician")
                attending = str(att_val).strip() if pd.notna(att_val) and str(att_val).strip() else None
                op_val = row_data.get("OperatingPhysician")
                operating = str(op_val).strip() if pd.notna(op_val) and str(op_val).strip() else None
                oth_val = row_data.get("OtherPhysician")
                other = str(oth_val).strip() if pd.notna(oth_val) and str(oth_val).strip() else None

                diag_codes = _extract_codes(row_data, "ClmDiagnosisCode", 10, "diagnosis_codes")
                proc_codes = _extract_codes(row_data, "ClmProcedureCode", 6, "procedure_codes")
            else:
                claim_type = "Outpatient"
                clm_dt = None
                clm_end_dt = None
                adm_dt = None
                dis_dt = None
                reimb_amt = float(ca.get("reimbursement", 0.0))
                deduct_amt = 0.0
                los = 0
                bene_id = None
                attending = None
                operating = None
                other = None
                diag_codes = []
                proc_codes = []

            # Ensure Provider exists
            provider = db.query(Provider).filter(Provider.provider_id == pid).first()
            if not provider:
                provider = Provider(provider_id=pid)
                db.add(provider)
                db.flush()

            # Ensure Beneficiary exists if present
            if bene_id:
                bene = db.query(Beneficiary).filter(Beneficiary.beneficiary_id == bene_id).first()
                if not bene:
                    bene = Beneficiary(beneficiary_id=bene_id)
                    db.add(bene)
                    db.flush()

            # Create or update claim record
            claim_rec = db.query(Claim).filter(Claim.claim_id == cid).first()
            if not claim_rec:
                claim_rec = Claim(
                    claim_id=cid,
                    provider_id=pid,
                    beneficiary_id=bene_id,
                    claim_type=claim_type,
                    claim_date=clm_dt,
                    claim_end_date=clm_end_dt,
                    admission_date=adm_dt,
                    discharge_date=dis_dt,
                    reimbursement_amount=reimb_amt,
                    deductible_amount=deduct_amt,
                    inpatient_los=los,
                    attending_physician=attending,
                    operating_physician=operating,
                    other_physician=other,
                    diagnosis_codes=diag_codes,
                    procedure_codes=proc_codes,
                )
                db.add(claim_rec)
            else:
                claim_rec.provider_id = pid
                if bene_id:
                    claim_rec.beneficiary_id = bene_id
                claim_rec.claim_type = claim_type
                claim_rec.reimbursement_amount = reimb_amt
                claim_rec.deductible_amount = deduct_amt
                if clm_dt:
                    claim_rec.claim_date = clm_dt
                if clm_end_dt:
                    claim_rec.claim_end_date = clm_end_dt
                if adm_dt:
                    claim_rec.admission_date = adm_dt
                if dis_dt:
                    claim_rec.discharge_date = dis_dt
                if attending:
                    claim_rec.attending_physician = attending
                if operating:
                    claim_rec.operating_physician = operating
                if other:
                    claim_rec.other_physician = other
                if diag_codes:
                    claim_rec.diagnosis_codes = diag_codes
                if proc_codes:
                    claim_rec.procedure_codes = proc_codes
            db.flush()

            prs_id = provider_score_map.get(pid)
            anomaly_service.persist_claim_analysis(
                db=db,
                claim_id=cid,
                provider_risk_score_id=prs_id,
                anomaly_record=ca,
            )

        # Update upload status
        upload.status = "COMPLETED"
        upload.successful_rows = total_successful
        upload.rejected_rows = total_rejected
        upload.completed_at = datetime.now(timezone.utc)
        db.commit()

    except Exception as e:
        logger.exception(f"Error processing CSV upload {upload_id}: {e}")
        db.rollback()
        upload = db.query(CSVUpload).filter(CSVUpload.id == upload_id).first()
        if upload:
            upload.status = "FAILED"
            upload.error_message = str(e)
            upload.completed_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        if close_db:
            db.close()
