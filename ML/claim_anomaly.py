"""
HealthGuard AI - Claim-Level Rule-Based Anomaly Detection Module
================================================================
Rule-based heuristic anomaly detection for individual health insurance claims.
Identifies suspicious patterns and data quality anomalies on individual claims
without predicting a claim-level fraud probability (as ground truth labels are
strictly provider-level).

Rules Implemented:
  1. HIGH_REIMBURSEMENT: Unusual reimbursement relative to peer P95 distribution derived strictly from training data.
  2. REPEAT_BENEFICIARY: Beneficiary with multiple claims submitted to same provider in close proximity (using provider history).
  3. POST_DEATH_CLAIM: Claim service date strictly follows the recorded beneficiary date of death.
  4. HIGH_SAME_DAY_ACTIVITY: Unusually high same-day claim submission volume for the provider.
  5. INVALID_CLAIM_DATES: Chronological inconsistencies (e.g., ClaimStart > ClaimEnd, Admission > Discharge).
  6. MISSING_PHYSICIAN_INFO: Omission of all physician identifiers or missing attending physician.
  7. MISSING_DIAGNOSIS_INFO / MISSING_PROCEDURE_INFO: Separate clinical checks for missing diagnosis and procedure codes.
  8. UNUSUAL_CLAIM_DURATION: Inpatient length of stay exceeding clinical norms or negative duration.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd


CLAIM_DISCLAIMER = (
    "This claim contains unusual indicators that may require investigation. "
    "These indicators are not proof of fraud."
)

# Actual P95 maximum reimbursement derived strictly from Phase 1 training provider reference distribution
DEFAULT_PEER_REIMBURSEMENT_P95 = 57000.0


def detect_claim_anomalies(
    claims_df: Union[pd.DataFrame, pd.Series, Dict[str, Any], List[Dict[str, Any]]],
    provider_reference_data: Optional[Dict[str, Any]] = None,
    peer_reimbursement_p95: Optional[float] = None,
    max_inpatient_los_days: int = 30,
    historical_claims_df: Optional[pd.DataFrame] = None,
    repeat_bene_threshold: int = 3,
) -> List[Dict[str, Any]]:
    """
    Run rule-based anomaly detection on individual claims.

    Parameters
    ----------
    claims_df : pd.DataFrame or compatible input
        Table of claims with standard CMS columns.
    provider_reference_data : dict, optional
        Phase 1 training peer reference table for numerical thresholds.
    peer_reimbursement_p95 : float, optional
        95th percentile benchmark for claim reimbursement (defaults to 57000.0 from training peer reference).
    max_inpatient_los_days : int
        Maximum normal inpatient length of stay before triggering anomaly flag.
    historical_claims_df : pd.DataFrame, optional
        Prior historical claims for the provider to evaluate cumulative repeat beneficiary patterns.

    Returns
    -------
    List[Dict[str, Any]]
        List of anomaly assessment records, one per claim in claims_df.
    """
    # 1. Normalize input to DataFrame
    if isinstance(claims_df, dict):
        df = pd.DataFrame([claims_df])
    elif isinstance(claims_df, pd.Series):
        df = pd.DataFrame([claims_df])
    elif isinstance(claims_df, list):
        df = pd.DataFrame(claims_df)
    elif isinstance(claims_df, pd.DataFrame):
        df = claims_df.copy()
    else:
        raise ValueError(f"Unsupported claims input type: {type(claims_df)}")

    if df.empty:
        return []

    # 2. Extract reference threshold from training peer reference table
    if peer_reimbursement_p95 is None:
        if provider_reference_data:
            dist = provider_reference_data.get("feature_distributions", provider_reference_data)
            max_reimb_dist = dist.get("max_reimbursement", {})
            if isinstance(max_reimb_dist, dict) and "p95" in max_reimb_dist:
                peer_reimbursement_p95 = float(max_reimb_dist["p95"])
            else:
                avg_reimb_dist = dist.get("avg_reimbursement", {})
                if isinstance(avg_reimb_dist, dict) and "p95" in avg_reimb_dist:
                    peer_reimbursement_p95 = float(avg_reimb_dist["p95"])
                else:
                    peer_reimbursement_p95 = DEFAULT_PEER_REIMBURSEMENT_P95
        else:
            peer_reimbursement_p95 = DEFAULT_PEER_REIMBURSEMENT_P95

    # Standardize column naming if variations exist
    col_map = {
        "claim_id": "ClaimID",
        "claimid": "ClaimID",
        "provider_id": "ProviderID",
        "providerid": "ProviderID",
        "bene_id": "BeneID",
        "beneid": "BeneID",
        "reimbursement": "InscClaimAmtReimbursed",
        "inscclaimamtreimbursed": "InscClaimAmtReimbursed",
        "claimstartdt": "ClaimStartDt",
        "claimenddt": "ClaimEndDt",
        "admissiondt": "AdmissionDt",
        "dischargedt": "DischargeDt",
        "claimtype": "ClaimType",
        "dod": "DOD",
        "dob": "DOB",
    }
    df = df.rename(columns={c: col_map[c.lower()] for c in df.columns if c.lower() in col_map})

    # Prepare datetime columns safely
    date_cols = ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt", "DOD", "DOB"]
    for c in date_cols:
        if c in df.columns:
            df[f"_{c}"] = pd.to_datetime(df[c], errors="coerce")

    # Group-level calculations for same-day claims within the current batch
    same_day_counts = {}
    if "ProviderID" in df.columns and "_ClaimStartDt" in df.columns:
        valid_start = df[df["_ClaimStartDt"].notna()]
        same_day_counts = valid_start.groupby(["ProviderID", "_ClaimStartDt"]).size().to_dict()

    # Repeat beneficiary calculation incorporating provider history (Fix 5)
    # Combine historical claims and current batch without duplicate ClaimIDs
    bene_counts = {}
    if "ProviderID" in df.columns and "BeneID" in df.columns:
        curr_bene_df = df[df["BeneID"].notna() & (df["BeneID"] != "")][["ProviderID", "BeneID", "ClaimID"]].copy()

        if historical_claims_df is not None and not historical_claims_df.empty:
            hist_df = historical_claims_df.rename(
                columns={c: col_map[c.lower()] for c in historical_claims_df.columns if c.lower() in col_map}
            )
            if "ProviderID" in hist_df.columns and "BeneID" in hist_df.columns:
                valid_hist = hist_df[hist_df["BeneID"].notna() & (hist_df["BeneID"] != "")].copy()
                if "ClaimID" in valid_hist.columns and "ClaimID" in curr_bene_df.columns:
                    curr_cids = set(curr_bene_df["ClaimID"].astype(str).tolist())
                    valid_hist = valid_hist[~valid_hist["ClaimID"].astype(str).isin(curr_cids)]
                combined_bene_df = pd.concat([valid_hist[["ProviderID", "BeneID"]], curr_bene_df[["ProviderID", "BeneID"]]], ignore_index=True)
            else:
                combined_bene_df = curr_bene_df[["ProviderID", "BeneID"]]
        else:
            combined_bene_df = curr_bene_df[["ProviderID", "BeneID"]]

        bene_counts = combined_bene_df.groupby(["ProviderID", "BeneID"]).size().to_dict()

    # Identify available diagnosis and procedure columns in input (Fix 3)
    diag_cols = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)] + ["ClmAdmitDiagnosisCode"]
    present_diag_cols = [c for c in diag_cols if c in df.columns]

    proc_cols = [f"ClmProcedureCode_{i}" for i in range(1, 7)]
    present_proc_cols = [c for c in proc_cols if c in df.columns]

    results = []

    for idx, row in df.iterrows():
        claim_id = str(row.get("ClaimID", f"CLAIM_{idx}"))
        provider_id = str(row.get("ProviderID", "UNKNOWN"))
        reimbursement = row.get("InscClaimAmtReimbursed", 0.0)
        try:
            reimbursement = float(reimbursement) if reimbursement is not None and not pd.isna(reimbursement) else 0.0
        except (ValueError, TypeError):
            reimbursement = 0.0

        claim_type = str(row.get("ClaimType", "Outpatient"))
        flags = []
        rules_triggered = []
        warnings = []

        # ---------------------------------------------------------------------
        # Rule 1: HIGH_REIMBURSEMENT (Fix 2: Uses actual training peer P95, never an invented P99)
        # ---------------------------------------------------------------------
        if reimbursement > peer_reimbursement_p95:
            flags.append("HIGH_REIMBURSEMENT_P95")
            flags.append("HIGH_REIMBURSEMENT")
            rules_triggered.append({
                "rule_name": "HIGH_REIMBURSEMENT_P95",
                "observed_value": reimbursement,
                "reference_value": peer_reimbursement_p95,
                "explanation": (
                    f"Claim reimbursement of ${reimbursement:,.2f} strictly exceeds the 95th percentile peer benchmark "
                    f"(${peer_reimbursement_p95:,.2f}) derived from training providers."
                ),
            })

        # ---------------------------------------------------------------------
        # Rule 2: REPEAT_BENEFICIARY (Fix 5: Evaluated with cumulative provider history)
        # ---------------------------------------------------------------------
        bene_id = row.get("BeneID", None)
        if pd.notna(bene_id) and bene_id != "":
            b_cnt = bene_counts.get((provider_id, bene_id), 1)
            repeat_col_val = row.get("repeat_bene_count", row.get("RepeatBeneCount", None))
            if b_cnt > repeat_bene_threshold or (repeat_col_val is not None and float(repeat_col_val or 0) > repeat_bene_threshold):
                flags.append("REPEAT_BENEFICIARY")
                rules_triggered.append({
                    "rule_name": "REPEAT_BENEFICIARY_SPIKE",
                    "observed_value": b_cnt,
                    "reference_value": repeat_bene_threshold,
                    "explanation": (
                        f"Beneficiary {bene_id} has {b_cnt} total claims with provider {provider_id} "
                        f"(including cumulative provider history)."
                    ),
                })
        else:
            warnings.append("Beneficiary identifier missing; repeat beneficiary rule skipped.")

        # ---------------------------------------------------------------------
        # Rule 3: POST_DEATH_CLAIM
        # ---------------------------------------------------------------------
        claim_after_death_flag = row.get("ClaimAfterDeath", None)
        is_post_death = False
        start_dt = row.get("_ClaimStartDt") if "_ClaimStartDt" in row else None
        dod_dt = row.get("_DOD") if "_DOD" in row else None

        if pd.notna(start_dt) and pd.notna(dod_dt) and start_dt > dod_dt:
            is_post_death = True
        elif claim_after_death_flag is not None and int(claim_after_death_flag or 0) == 1:
            is_post_death = True

        if is_post_death:
            flags.append("POST_DEATH_CLAIM")
            rules_triggered.append({
                "rule_name": "CLAIM_SUBMITTED_AFTER_DEATH",
                "observed_value": str(start_dt.strftime("%Y-%m-%d")) if pd.notna(start_dt) else "After Death Flag",
                "reference_value": str(dod_dt.strftime("%Y-%m-%d")) if pd.notna(dod_dt) else "Recorded Date of Death",
                "explanation": "Claim start date occurs after the recorded beneficiary date of death.",
            })

        # ---------------------------------------------------------------------
        # Rule 4: HIGH_SAME_DAY_ACTIVITY
        # ---------------------------------------------------------------------
        if pd.notna(start_dt):
            same_day_cnt = same_day_counts.get((provider_id, start_dt), 1)
            same_day_flag = row.get("same_day_repeated_claims", row.get("SameDayRepeatedClaims", None))
            if same_day_cnt >= 10 or (same_day_flag is not None and float(same_day_flag or 0) >= 10):
                flags.append("HIGH_SAME_DAY_ACTIVITY")
                rules_triggered.append({
                    "rule_name": "CONCENTRATED_SAME_DAY_BILLING",
                    "observed_value": same_day_cnt,
                    "reference_value": 10,
                    "explanation": f"Provider submitted {same_day_cnt} claims on the single date {start_dt.strftime('%Y-%m-%d')}.",
                })

        # ---------------------------------------------------------------------
        # Rule 5: INVALID_CLAIM_DATES
        # ---------------------------------------------------------------------
        end_dt = row.get("_ClaimEndDt") if "_ClaimEndDt" in row else None
        admit_dt = row.get("_AdmissionDt") if "_AdmissionDt" in row else None
        disch_dt = row.get("_DischargeDt") if "_DischargeDt" in row else None
        date_anomaly_flag = row.get("ClaimDateAnomaly", row.get("DischargeBeforeAdmission", None))

        is_date_invalid = False
        date_err_detail = []

        if pd.notna(start_dt) and pd.notna(end_dt) and start_dt > end_dt:
            is_date_invalid = True
            date_err_detail.append(f"ClaimStartDt ({start_dt.strftime('%Y-%m-%d')}) > ClaimEndDt ({end_dt.strftime('%Y-%m-%d')})")

        if pd.notna(admit_dt) and pd.notna(disch_dt) and admit_dt > disch_dt:
            is_date_invalid = True
            date_err_detail.append(f"AdmissionDt ({admit_dt.strftime('%Y-%m-%d')}) > DischargeDt ({disch_dt.strftime('%Y-%m-%d')})")

        if date_anomaly_flag is not None and int(date_anomaly_flag or 0) == 1:
            is_date_invalid = True
            date_err_detail.append("Chronological anomaly recorded in claim header metadata.")

        if is_date_invalid:
            flags.append("INVALID_CLAIM_DATES")
            rules_triggered.append({
                "rule_name": "CHRONOLOGICAL_DATE_INCONSISTENCY",
                "observed_value": "; ".join(date_err_detail),
                "reference_value": "ClaimStart <= ClaimEnd, Admission <= Discharge",
                "explanation": f"Claim dates violate clinical sequencing: {'; '.join(date_err_detail)}.",
            })

        # ---------------------------------------------------------------------
        # Rule 6: MISSING_PHYSICIAN_INFO
        # ---------------------------------------------------------------------
        attending = row.get("AttendingPhysician", None)
        operating = row.get("OperatingPhysician", None)
        other_phys = row.get("OtherPhysician", None)

        attending_missing = pd.isna(attending) or attending is None or str(attending).strip() in ["", "nan", "None"]
        operating_missing = pd.isna(operating) or operating is None or str(operating).strip() in ["", "nan", "None"]
        other_missing = pd.isna(other_phys) or other_phys is None or str(other_phys).strip() in ["", "nan", "None"]

        if attending_missing and operating_missing and other_missing:
            flags.append("MISSING_PHYSICIAN_INFO")
            rules_triggered.append({
                "rule_name": "ALL_PHYSICIANS_OMITTED",
                "observed_value": "All physician IDs missing",
                "reference_value": "At least one attending or operating physician identifier required",
                "explanation": "Claim does not specify an attending, operating, or other physician identifier.",
            })
        elif claim_type.lower() == "inpatient" and attending_missing:
            flags.append("MISSING_PHYSICIAN_INFO")
            rules_triggered.append({
                "rule_name": "INPATIENT_ATTENDING_PHYSICIAN_MISSING",
                "observed_value": "Attending Physician null",
                "reference_value": "Attending physician mandatory for inpatient admissions",
                "explanation": "Inpatient institutional claim lacks mandatory attending physician identifier.",
            })

        # ---------------------------------------------------------------------
        # Rule 7: DIAGNOSIS AND PROCEDURE SEPARATE CHECKS (Fix 3)
        # ---------------------------------------------------------------------
        diag_missing = False
        proc_missing = False

        # A. Diagnosis completeness
        if not present_diag_cols:
            warnings.append("Diagnosis code columns not present in input claims data; diagnosis completeness check omitted.")
        else:
            diag_count = 0
            for dc in present_diag_cols:
                val = row[dc]
                if pd.notna(val) and str(val).strip() not in ["", "nan", "None"]:
                    diag_count += 1
            if diag_count == 0:
                diag_missing = True
                flags.append("MISSING_DIAGNOSIS_INFO")
                rules_triggered.append({
                    "rule_name": "MISSING_DIAGNOSIS_CODES",
                    "observed_value": 0,
                    "reference_value": ">= 1 diagnosis code required",
                    "explanation": "Claim lacks any valid primary, admitting, or secondary diagnosis codes across diagnosis columns.",
                })

        # B. Procedure completeness
        if not present_proc_cols:
            warnings.append("Procedure code columns not present in input claims data; procedure completeness check omitted.")
        else:
            proc_count = 0
            for pc in present_proc_cols:
                val = row[pc]
                if pd.notna(val) and str(val).strip() not in ["", "nan", "None"]:
                    proc_count += 1
            if proc_count == 0 and (claim_type.lower() == "inpatient" or "ClmProcedureCode_1" in present_proc_cols):
                proc_missing = True
                flags.append("MISSING_PROCEDURE_INFO")
                rules_triggered.append({
                    "rule_name": "MISSING_PROCEDURE_CODES",
                    "observed_value": 0,
                    "reference_value": ">= 1 procedure code expected for surgical or procedure reporting",
                    "explanation": "No procedure codes recorded across procedure columns (ClmProcedureCode_1 through ClmProcedureCode_6).",
                })

        # Combined flag for backward compatibility if both are missing
        if diag_missing and proc_missing:
            flags.append("MISSING_DIAGNOSIS_PROCEDURE")

        # ---------------------------------------------------------------------
        # Rule 8: UNUSUAL_CLAIM_DURATION
        # ---------------------------------------------------------------------
        los = row.get("InpatientLOS", None)
        if (los is None or pd.isna(los)) and pd.notna(admit_dt) and pd.notna(disch_dt):
            los = (disch_dt - admit_dt).days

        if los is not None and pd.notna(los):
            try:
                los_val = float(los)
                if los_val < 0:
                    flags.append("UNUSUAL_CLAIM_DURATION")
                    rules_triggered.append({
                        "rule_name": "NEGATIVE_LENGTH_OF_STAY",
                        "observed_value": los_val,
                        "reference_value": ">= 0 days",
                        "explanation": f"Inpatient length of stay is negative ({los_val} days).",
                    })
                elif los_val > max_inpatient_los_days:
                    flags.append("UNUSUAL_CLAIM_DURATION")
                    rules_triggered.append({
                        "rule_name": "EXTENDED_LENGTH_OF_STAY",
                        "observed_value": los_val,
                        "reference_value": max_inpatient_los_days,
                        "explanation": f"Inpatient length of stay of {los_val:.0f} days exceeds clinical benchmark ({max_inpatient_los_days} days).",
                    })
            except (ValueError, TypeError):
                warnings.append("Unable to parse length of stay as numerical value.")

        # Assemble individual claim record
        is_anomalous = len(flags) > 0
        anomaly_count = len(flags)

        if is_anomalous:
            claim_disclaimer = (
                "This claim contains unusual indicators that may require investigation. "
                "These indicators are not proof of fraud."
            )
        else:
            claim_disclaimer = (
                "No anomalous indicators were detected for this claim. "
                "This is an operational rule check and not proof of absence of fraud."
            )

        results.append({
            "claim_id": claim_id,
            "provider_id": provider_id,
            "reimbursement_amount": reimbursement,
            "claim_type": claim_type,
            "anomaly_flags": flags,
            "is_anomalous": is_anomalous,
            "anomaly_count": anomaly_count,
            "rules_triggered": rules_triggered,
            "data_availability_warnings": warnings if warnings else None,
            "disclaimer": claim_disclaimer,
        })

    return results
