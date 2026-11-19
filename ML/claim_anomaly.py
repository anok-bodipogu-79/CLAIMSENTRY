"""
HealthGuard AI - Advanced Claim-Level Anomaly & Fraud Detection Engine (Phase 2.2)
==================================================================================
Dual-Assessment Architecture: Evaluates individual claim-level suspicious activity,
billing integrity, temporal consistency, duplicate submissions, and clinical coding
syntax strictly independently of provider-level machine learning fraud models.

Core Design Invariants:
1. Provider ML Risk (systemic probability) and Claim Anomaly Detection (encounter evidence)
   are strictly decoupled.
2. A LOW provider-level ML score never suppresses individual claim anomalies.
3. Anomaly scores are weighted evidence severity indices, NOT statistical fraud probabilities.
4. Preserves all 8 legacy anomaly rules and extends with advanced cross-provider,
   duplicate billing, financial deviation, and clinical coding integrity checks.
5. Authoritative dependency boundaries: Clinical rules that require unavailable reference
   data (e.g., NCCI edits requiring CPT/HCPCS modifiers) are explicitly logged as
   UNAVAILABLE_MISSING_DEPENDENCY rather than fabricating unsupported assertions.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone, date
import re
import math
import numpy as np
import pandas as pd

CLAIM_DISCLAIMER = (
    "This claim contains unusual indicators that may require investigation. "
    "These indicators are not proof of fraud."
)

CLEAN_CLAIM_DISCLAIMER = (
    "No anomalous indicators were detected for this claim. "
    "This is an operational rule check and not proof of absence of fraud."
)

DEFAULT_PEER_REIMBURSEMENT_P95 = 57000.0
RULE_VERSION = "2.2.0"

# Mathematical weights for evidence scoring (Documented severity index)
SEVERITY_WEIGHTS = {
    "HIGH": 3.0,
    "MEDIUM": 1.5,
    "LOW": 0.5,
    "INFORMATIONAL": 0.0,
}

# Standard ICD-9 regex: 3-5 chars, optional leading V or E
ICD9_DIAG_PATTERN = re.compile(r"^(V\d{2}(\d{1,2})?|E\d{3}(\d)?|\d{3}(\d{1,2})?)$", re.IGNORECASE)
ICD9_PROC_PATTERN = re.compile(r"^\d{2,4}$")
ICD10_PATTERN = re.compile(r"^[A-TV-Z][0-9][0-9A-Z](\.?[0-9A-Z]{1,4})?$", re.IGNORECASE)
ICD10_CUTOFF_DATE = date(2015, 10, 1)


def _safe_str(val: Any) -> Optional[str]:
    """Return non-empty stripped string or None."""
    if val is None or pd.isna(val):
        return None
    s = str(val).strip()
    return s if s and s.lower() not in ("nan", "none", "null") else None


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely parse finite float."""
    if val is None or pd.isna(val):
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (ValueError, TypeError):
        return default


def _safe_date(val: Any) -> Optional[date]:
    """Safely parse to datetime.date object."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (datetime, pd.Timestamp)):
        return val.date()
    if isinstance(val, date):
        return val
    try:
        dt = pd.to_datetime(val)
        return dt.date() if pd.notna(dt) else None
    except Exception:
        return None


def detect_claim_anomalies(
    claims_df: Union[pd.DataFrame, pd.Series, Dict[str, Any], List[Dict[str, Any]]],
    provider_reference_data: Optional[Dict[str, Any]] = None,
    peer_reimbursement_p95: Optional[float] = None,
    max_inpatient_los_days: int = 30,
    historical_claims_df: Optional[pd.DataFrame] = None,
    repeat_bene_threshold: int = 3,
    patient_historical_claims: Optional[Union[pd.DataFrame, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """
    Run comprehensive rule-based anomaly detection on individual claims.

    Parameters
    ----------
    claims_df : pd.DataFrame or compatible input
        Table of claims with standard CMS columns.
    provider_reference_data : dict, optional
        Training peer reference table for numerical thresholds.
    peer_reimbursement_p95 : float, optional
        95th percentile benchmark for claim reimbursement (defaults to 57000.0).
    max_inpatient_los_days : int
        Maximum normal inpatient length of stay before triggering anomaly flag.
    historical_claims_df : pd.DataFrame, optional
        Prior historical claims for the provider strictly excluding the current claim.
    repeat_bene_threshold : int
        Threshold count for repeat beneficiary billing at same provider.
    patient_historical_claims : pd.DataFrame or list, optional
        Beneficiary historical claims across ALL providers strictly excluding current claim.

    Returns
    -------
    List[Dict[str, Any]]
        List of anomaly assessment records with structured evidence, evaluated checks,
        and documented anomaly scores.
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

    # Standardize column naming
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
        "deductibleamtpaid": "DeductibleAmtPaid",
        "attendingphysician": "AttendingPhysician",
        "operatingphysician": "OperatingPhysician",
        "otherphysician": "OtherPhysician",
    }
    df = df.rename(columns={c: col_map[c.lower()] for c in df.columns if c.lower() in col_map})

    # Prepare datetime columns safely
    date_cols = ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt", "DOD", "DOB"]
    for c in date_cols:
        if c in df.columns:
            df[f"_{c}"] = pd.to_datetime(df[c], errors="coerce")

    # Normalize historical claims dataframes
    hist_df_clean = None
    if historical_claims_df is not None and not historical_claims_df.empty:
        hist_df_clean = historical_claims_df.rename(
            columns={c: col_map[c.lower()] for c in historical_claims_df.columns if c.lower() in col_map}
        ).copy()
        for c in date_cols:
            if c in hist_df_clean.columns:
                hist_df_clean[f"_{c}"] = pd.to_datetime(hist_df_clean[c], errors="coerce")

    patient_hist_df = None
    if patient_historical_claims is not None:
        if isinstance(patient_historical_claims, list):
            patient_hist_df = pd.DataFrame(patient_historical_claims)
        elif isinstance(patient_historical_claims, pd.DataFrame):
            patient_hist_df = patient_historical_claims.copy()

        if patient_hist_df is not None and not patient_hist_df.empty:
            patient_hist_df = patient_hist_df.rename(
                columns={c: col_map[c.lower()] for c in patient_hist_df.columns if c.lower() in col_map}
            )
            for c in date_cols:
                if c in patient_hist_df.columns:
                    patient_hist_df[f"_{c}"] = pd.to_datetime(patient_hist_df[c], errors="coerce")

    # Group-level calculations for same-day claims within the current batch
    same_day_counts = {}
    if "ProviderID" in df.columns and "_ClaimStartDt" in df.columns:
        valid_start = df[df["_ClaimStartDt"].notna()]
        same_day_counts = valid_start.groupby(["ProviderID", "_ClaimStartDt"]).size().to_dict()

    # Repeat beneficiary calculation incorporating provider history
    bene_counts = {}
    if "ProviderID" in df.columns and "BeneID" in df.columns:
        curr_bene_df = df[df["BeneID"].notna() & (df["BeneID"] != "")][["ProviderID", "BeneID", "ClaimID"]].copy()

        if hist_df_clean is not None and not hist_df_clean.empty and "ProviderID" in hist_df_clean.columns and "BeneID" in hist_df_clean.columns:
            valid_hist = hist_df_clean[hist_df_clean["BeneID"].notna() & (hist_df_clean["BeneID"] != "")].copy()
            if "ClaimID" in valid_hist.columns and "ClaimID" in curr_bene_df.columns:
                curr_cids = set(curr_bene_df["ClaimID"].astype(str).tolist())
                valid_hist = valid_hist[~valid_hist["ClaimID"].astype(str).isin(curr_cids)]
            combined_bene_df = pd.concat([valid_hist[["ProviderID", "BeneID"]], curr_bene_df[["ProviderID", "BeneID"]]], ignore_index=True)
        else:
            combined_bene_df = curr_bene_df[["ProviderID", "BeneID"]]

        bene_counts = combined_bene_df.groupby(["ProviderID", "BeneID"]).size().to_dict()

    # Identify available diagnosis and procedure columns in input
    diag_cols = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)] + ["ClmAdmitDiagnosisCode"]
    present_diag_cols = [c for c in diag_cols if c in df.columns]

    proc_cols = [f"ClmProcedureCode_{i}" for i in range(1, 7)]
    present_proc_cols = [c for c in proc_cols if c in df.columns]

    now_utc_str = datetime.now(timezone.utc).isoformat()
    results = []

    for idx, row in df.iterrows():
        claim_id = str(row.get("ClaimID", f"CLAIM_{idx}"))
        provider_id = str(row.get("ProviderID", "UNKNOWN"))
        bene_id = _safe_str(row.get("BeneID"))
        reimbursement = _safe_float(row.get("InscClaimAmtReimbursed", 0.0))
        claim_type = str(row.get("ClaimType", "Outpatient")).capitalize()

        flags: List[str] = []
        rules_triggered: List[Dict[str, Any]] = []
        evaluated_checks: List[Dict[str, Any]] = []
        warnings: List[str] = []

        start_dt = row.get("_ClaimStartDt") if "_ClaimStartDt" in row else None
        end_dt = row.get("_ClaimEndDt") if "_ClaimEndDt" in row else None
        admit_dt = row.get("_AdmissionDt") if "_AdmissionDt" in row else None
        disch_dt = row.get("_DischargeDt") if "_DischargeDt" in row else None
        dod_dt = row.get("_DOD") if "_DOD" in row else None

        start_d = _safe_date(start_dt)
        end_d = _safe_date(end_dt) or start_d

        # Collect clinical diagnosis and procedure codes for current claim
        claim_diag_codes = []
        for dc in present_diag_cols:
            val_s = _safe_str(row.get(dc))
            if val_s:
                claim_diag_codes.append(val_s)

        claim_proc_codes = []
        for pc in present_proc_cols:
            val_s = _safe_str(row.get(pc))
            if val_s:
                claim_proc_codes.append(val_s)

        # =====================================================================
        # CATEGORY 1: FINANCIAL ANOMALIES
        # =====================================================================

        # Rule 1: HIGH_REIMBURSEMENT (Preserved Legacy Rule)
        is_high_reimb = reimbursement > peer_reimbursement_p95
        if is_high_reimb:
            flags.extend(["HIGH_REIMBURSEMENT_P95", "HIGH_REIMBURSEMENT"])
            rules_triggered.append({
                "rule_id": "HIGH_REIMBURSEMENT_P95",
                "rule_name": "High Reimbursement Relative to Peer Benchmark",
                "rule_category": "FINANCIAL_ANOMALY",
                "severity": "HIGH",
                "observed_value": f"${reimbursement:,.2f}",
                "expected_value": f"<= ${peer_reimbursement_p95:,.2f} (Peer 95th Percentile)",
                "evidence_description": (
                    f"Claim reimbursement of ${reimbursement:,.2f} strictly exceeds the national 95th percentile "
                    f"peer benchmark (${peer_reimbursement_p95:,.2f}) derived from training providers."
                ),
                "historical_records": [],
                "detection_timestamp": now_utc_str,
                "rule_version": RULE_VERSION,
            })
        evaluated_checks.append({
            "rule_id": "HIGH_REIMBURSEMENT_P95",
            "rule_name": "Peer Reimbursement Benchmark Check",
            "rule_category": "FINANCIAL_ANOMALY",
            "status": "FLAGGED" if is_high_reimb else "PASSED",
            "description": "Compares claim reimbursement against peer P95 distribution benchmark.",
            "details": f"Reimbursement ${reimbursement:,.2f} vs P95 benchmark ${peer_reimbursement_p95:,.2f}.",
        })

        # Rule 1B: Provider-Specific Historical Reimbursement Deviation (New Phase 2.2)
        prov_reimb_flag = False
        prov_reimb_details = "Insufficient provider history (< 5 prior claims) for baseline comparison."
        if hist_df_clean is not None and not hist_df_clean.empty and "InscClaimAmtReimbursed" in hist_df_clean.columns:
            valid_hist_reimb = hist_df_clean[hist_df_clean["ClaimID"].astype(str) != claim_id]["InscClaimAmtReimbursed"].dropna()
            if len(valid_hist_reimb) >= 5:
                hist_mean = float(valid_hist_reimb.mean())
                hist_std = float(valid_hist_reimb.std()) if len(valid_hist_reimb) > 1 else 0.0
                hist_median = float(valid_hist_reimb.median())
                threshold_3x = max(hist_median * 3.0, hist_mean + 3.0 * hist_std)

                if reimbursement > threshold_3x and reimbursement > 5000.0:
                    prov_reimb_flag = True
                    flags.append("FIN_PROVIDER_HIST_DEVIATION")
                    rules_triggered.append({
                        "rule_id": "FIN_PROVIDER_HIST_DEVIATION",
                        "rule_name": "Substantial Provider Historical Reimbursement Deviation",
                        "rule_category": "FINANCIAL_ANOMALY",
                        "severity": "MEDIUM",
                        "observed_value": f"${reimbursement:,.2f}",
                        "expected_value": f"Provider historical median: ${hist_median:,.2f} (mean: ${hist_mean:,.2f})",
                        "evidence_description": (
                            f"Claim reimbursement of ${reimbursement:,.2f} is substantially higher than the provider's "
                            f"historical baseline (median: ${hist_median:,.2f}, mean: ${hist_mean:,.2f} across {len(valid_hist_reimb)} prior claims)."
                        ),
                        "historical_records": [
                            {"prior_claim_count": len(valid_hist_reimb), "provider_median": hist_median, "provider_mean": hist_mean}
                        ],
                        "detection_timestamp": now_utc_str,
                        "rule_version": RULE_VERSION,
                    })
                    prov_reimb_details = f"Reimbursement ${reimbursement:,.2f} exceeds 3x provider historical median (${hist_median:,.2f})."
                else:
                    prov_reimb_details = f"Within expected historical variance (provider median: ${hist_median:,.2f}, n={len(valid_hist_reimb)})."
            else:
                prov_reimb_details = f"Only {len(valid_hist_reimb)} prior claims recorded; minimum 5 required for baseline."
        evaluated_checks.append({
            "rule_id": "FIN_PROVIDER_HIST_DEVIATION",
            "rule_name": "Provider Baseline Reimbursement Comparison",
            "rule_category": "FINANCIAL_ANOMALY",
            "status": "FLAGGED" if prov_reimb_flag else ("PASSED" if "Within" in prov_reimb_details else "NOT_APPLICABLE"),
            "description": "Flags claims deviating > 3x from provider's own historical median reimbursement.",
            "details": prov_reimb_details,
        })

        # Rule 1C: Extreme Outpatient Reimbursement Outlier (New Phase 2.2)
        outpatient_extreme_flag = False
        if claim_type.lower() == "outpatient" and reimbursement > 15000.0:
            outpatient_extreme_flag = True
            flags.append("FIN_OUTPATIENT_EXTREME_REIMBURSEMENT")
            rules_triggered.append({
                "rule_id": "FIN_OUTPATIENT_EXTREME_REIMBURSEMENT",
                "rule_name": "Extreme Outpatient Reimbursement Outlier",
                "rule_category": "FINANCIAL_ANOMALY",
                "severity": "MEDIUM",
                "observed_value": f"${reimbursement:,.2f}",
                "expected_value": "<= $15,000.00 for non-inpatient ambulatory encounter",
                "evidence_description": (
                    f"Outpatient claim reimbursement of ${reimbursement:,.2f} is exceptionally high for an ambulatory encounter "
                    f"without inpatient facility DRG grouping."
                ),
                "historical_records": [],
                "detection_timestamp": now_utc_str,
                "rule_version": RULE_VERSION,
            })
        evaluated_checks.append({
            "rule_id": "FIN_OUTPATIENT_EXTREME_REIMBURSEMENT",
            "rule_name": "Outpatient Ambulatory Cap Check",
            "rule_category": "FINANCIAL_ANOMALY",
            "status": "FLAGGED" if outpatient_extreme_flag else "PASSED",
            "description": "Checks if outpatient reimbursement exceeds $15,000.00 standard ambulatory threshold.",
            "details": f"Claim type: {claim_type}, Reimbursement: ${reimbursement:,.2f}.",
        })

        # =====================================================================
        # CATEGORY 2: DUPLICATE BILLING DETECTION
        # =====================================================================

        # Rule 2A: Exact Duplicate Claim Screening (New Phase 2.2)
        exact_dup_flag = False
        exact_dup_records = []
        if bene_id and start_d:
            # Check against provider history
            if hist_df_clean is not None and not hist_df_clean.empty:
                for _, hrow in hist_df_clean.iterrows():
                    h_cid = str(hrow.get("ClaimID", "")).strip()
                    if h_cid == claim_id:
                        continue
                    h_bene = _safe_str(hrow.get("BeneID"))
                    h_sdt = _safe_date(hrow.get("_ClaimStartDt") if "_ClaimStartDt" in hrow else hrow.get("ClaimStartDt"))
                    h_reimb = _safe_float(hrow.get("InscClaimAmtReimbursed"))
                    if h_bene == bene_id and h_sdt == start_d and abs(h_reimb - reimbursement) < 0.01:
                        exact_dup_flag = True
                        exact_dup_records.append({
                            "conflicting_claim_id": h_cid,
                            "service_date": str(h_sdt),
                            "reimbursement": h_reimb,
                            "source": "provider_history",
                        })
            # Check against patient history across all providers
            if patient_hist_df is not None and not patient_hist_df.empty:
                for _, prow in patient_hist_df.iterrows():
                    p_cid = str(prow.get("ClaimID", "")).strip()
                    if p_cid == claim_id or any(r["conflicting_claim_id"] == p_cid for r in exact_dup_records):
                        continue
                    p_prov = _safe_str(prow.get("ProviderID"))
                    p_sdt = _safe_date(prow.get("_ClaimStartDt") if "_ClaimStartDt" in prow else prow.get("ClaimStartDt"))
                    p_reimb = _safe_float(prow.get("InscClaimAmtReimbursed"))
                    if p_prov == provider_id and p_sdt == start_d and abs(p_reimb - reimbursement) < 0.01:
                        exact_dup_flag = True
                        exact_dup_records.append({
                            "conflicting_claim_id": p_cid,
                            "service_date": str(p_sdt),
                            "reimbursement": p_reimb,
                            "source": "patient_history",
                        })

        if exact_dup_flag:
            flags.append("DUP_EXACT_CLAIM")
            rules_triggered.append({
                "rule_id": "DUP_EXACT_CLAIM",
                "rule_name": "Exact Duplicate Claim Submission",
                "rule_category": "DUPLICATE_BILLING",
                "severity": "HIGH",
                "observed_value": f"Exact match with prior claim(s): {', '.join(r['conflicting_claim_id'] for r in exact_dup_records)}",
                "expected_value": "Zero duplicate submissions with identical beneficiary, provider, service date, and amount",
                "evidence_description": (
                    f"Claim matches prior claim(s) ({', '.join(r['conflicting_claim_id'] for r in exact_dup_records)}) "
                    f"for beneficiary {bene_id} at provider {provider_id} on {start_d} with identical reimbursement (${reimbursement:,.2f})."
                ),
                "historical_records": exact_dup_records,
                "detection_timestamp": now_utc_str,
                "rule_version": RULE_VERSION,
            })
        evaluated_checks.append({
            "rule_id": "DUP_EXACT_CLAIM",
            "rule_name": "Exact Duplicate Billing Verification",
            "rule_category": "DUPLICATE_BILLING",
            "status": "FLAGGED" if exact_dup_flag else "PASSED",
            "description": "Detects identical claim submissions matching beneficiary, provider, service date, and reimbursement.",
            "details": f"{len(exact_dup_records)} duplicate match(es) detected." if exact_dup_flag else "No exact duplicate submissions found.",
        })

        # Rule 2B: Intra-Claim Duplicate Procedure Codes (New Phase 2.2)
        intra_proc_dup_flag = False
        dup_proc_codes = []
        if len(claim_proc_codes) > 1:
            seen_procs = set()
            for pc in claim_proc_codes:
                if pc in seen_procs:
                    dup_proc_codes.append(pc)
                seen_procs.add(pc)
            if dup_proc_codes:
                intra_proc_dup_flag = True
                flags.append("CLIN_INTRA_CLAIM_DUPLICATE_PROCEDURE")
                rules_triggered.append({
                    "rule_id": "CLIN_INTRA_CLAIM_DUPLICATE_PROCEDURE",
                    "rule_name": "Intra-Claim Duplicate Procedure Code Billing",
                    "rule_category": "CLINICAL_INTEGRITY",
                    "severity": "MEDIUM",
                    "observed_value": f"Duplicate procedure code(s): {', '.join(dup_proc_codes)}",
                    "expected_value": "Distinct procedure codes per encounter (or appropriate multi-unit reporting)",
                    "evidence_description": (
                        f"Procedure code(s) {', '.join(dup_proc_codes)} billed multiple times on the same claim encounter "
                        f"without distinct procedural modifiers."
                    ),
                    "historical_records": [],
                    "detection_timestamp": now_utc_str,
                    "rule_version": RULE_VERSION,
                })
        evaluated_checks.append({
            "rule_id": "CLIN_INTRA_CLAIM_DUPLICATE_PROCEDURE",
            "rule_name": "Intra-Claim Duplicate Procedure Verification",
            "rule_category": "CLINICAL_INTEGRITY",
            "status": "FLAGGED" if intra_proc_dup_flag else "PASSED",
            "description": "Verifies that the same procedure code is not repeated across procedure columns on the same claim.",
            "details": f"Duplicate codes: {dup_proc_codes}." if intra_proc_dup_flag else f"{len(claim_proc_codes)} unique procedure code(s) recorded.",
        })

        # Rule 2C: Same-Day Cross-Claim Duplicate Procedure Billing (New Phase 2.2)
        same_day_proc_dup_flag = False
        same_day_proc_records = []
        if bene_id and start_d and claim_proc_codes and patient_hist_df is not None and not patient_hist_df.empty:
            for _, prow in patient_hist_df.iterrows():
                p_cid = str(prow.get("ClaimID", "")).strip()
                if p_cid == claim_id:
                    continue
                p_sdt = _safe_date(prow.get("_ClaimStartDt") if "_ClaimStartDt" in prow else prow.get("ClaimStartDt"))
                if p_sdt == start_d:
                    p_procs = []
                    for pc in proc_cols:
                        v = _safe_str(prow.get(pc))
                        if v:
                            p_procs.append(v)
                    # Check procedure intersection
                    overlap_procs = set(claim_proc_codes).intersection(set(p_procs))
                    if overlap_procs:
                        same_day_proc_dup_flag = True
                        same_day_proc_records.append({
                            "conflicting_claim_id": p_cid,
                            "provider_id": str(prow.get("ProviderID", "UNKNOWN")),
                            "service_date": str(p_sdt),
                            "repeated_procedure_codes": list(overlap_procs),
                        })

        if same_day_proc_dup_flag:
            flags.append("DUP_SAME_DAY_PROCEDURE")
            rules_triggered.append({
                "rule_id": "DUP_SAME_DAY_PROCEDURE",
                "rule_name": "Repeated Procedure Billing on Same Service Date",
                "rule_category": "DUPLICATE_BILLING",
                "severity": "HIGH",
                "observed_value": f"Repeated procedure(s) on same date across claims: {', '.join(str(r['repeated_procedure_codes']) for r in same_day_proc_records)}",
                "expected_value": "Single procedure execution per clinical indication on same service date",
                "evidence_description": (
                    f"Beneficiary {bene_id} has the same procedure code(s) billed on the same service date ({start_d}) "
                    f"across multiple distinct claims ({', '.join(r['conflicting_claim_id'] for r in same_day_proc_records)})."
                ),
                "historical_records": same_day_proc_records,
                "detection_timestamp": now_utc_str,
                "rule_version": RULE_VERSION,
            })
        evaluated_checks.append({
            "rule_id": "DUP_SAME_DAY_PROCEDURE",
            "rule_name": "Same-Day Procedure Repetition Across Claims",
            "rule_category": "DUPLICATE_BILLING",
            "status": "FLAGGED" if same_day_proc_dup_flag else "PASSED",
            "description": "Detects identical surgical/clinical procedures billed for the beneficiary on the same service date.",
            "details": f"{len(same_day_proc_records)} conflicting claim(s) with identical procedures on {start_d}." if same_day_proc_dup_flag else "No repeated procedures on same service date.",
        })

        # =====================================================================
        # CATEGORY 3: TEMPORAL ANOMALY DETECTION
        # =====================================================================

        # Rule 3A: POST_DEATH_CLAIM (Preserved Legacy Rule)
        claim_after_death_flag = row.get("ClaimAfterDeath", None)
        is_post_death = False
        if pd.notna(start_dt) and pd.notna(dod_dt) and start_dt > dod_dt:
            is_post_death = True
        elif claim_after_death_flag is not None and int(claim_after_death_flag or 0) == 1:
            is_post_death = True

        if is_post_death:
            flags.append("POST_DEATH_CLAIM")
            rules_triggered.append({
                "rule_id": "POST_DEATH_CLAIM",
                "rule_name": "Claim Service Date Follows Beneficiary Date of Death",
                "rule_category": "TEMPORAL_ANOMALY",
                "severity": "HIGH",
                "observed_value": str(start_d) if start_d else "Claim After Death Flag",
                "expected_value": f"Service date on or before date of death ({str(_safe_date(dod_dt)) if dod_dt is not None else 'DOD'})",
                "evidence_description": (
                    f"Claim start date ({start_d}) occurs strictly after the recorded beneficiary date of death "
                    f"({_safe_date(dod_dt)}). Billing for services rendered post-mortem is a critical integrity defect."
                ),
                "historical_records": [{"date_of_death": str(_safe_date(dod_dt)), "claim_start_date": str(start_d)}],
                "detection_timestamp": now_utc_str,
                "rule_version": RULE_VERSION,
            })
        evaluated_checks.append({
            "rule_id": "POST_DEATH_CLAIM",
            "rule_name": "Post-Mortem Billing Verification",
            "rule_category": "TEMPORAL_ANOMALY",
            "status": "FLAGGED" if is_post_death else ("PASSED" if dod_dt is not None else "NOT_APPLICABLE"),
            "description": "Verifies that claim service dates do not exceed the recorded beneficiary date of death.",
            "details": f"Service date: {start_d}, Beneficiary DOD: {_safe_date(dod_dt) or 'Living/Unrecorded'}.",
        })

        # Rule 3B: INVALID_CLAIM_DATES (Preserved Legacy Rule)
        date_anomaly_flag = row.get("ClaimDateAnomaly", row.get("DischargeBeforeAdmission", None))
        is_date_invalid = False
        date_err_detail = []

        if pd.notna(start_dt) and pd.notna(end_dt) and start_dt > end_dt:
            is_date_invalid = True
            date_err_detail.append(f"ClaimStartDt ({start_d}) > ClaimEndDt ({end_d})")

        if pd.notna(admit_dt) and pd.notna(disch_dt) and admit_dt > disch_dt:
            is_date_invalid = True
            date_err_detail.append(f"AdmissionDt ({_safe_date(admit_dt)}) > DischargeDt ({_safe_date(disch_dt)})")

        if date_anomaly_flag is not None and int(date_anomaly_flag or 0) == 1:
            is_date_invalid = True
            date_err_detail.append("Chronological anomaly recorded in claim header metadata.")

        if is_date_invalid:
            flags.append("INVALID_CLAIM_DATES")
            rules_triggered.append({
                "rule_id": "INVALID_CLAIM_DATES",
                "rule_name": "Chronological Date Inconsistency",
                "rule_category": "DATA_INTEGRITY",
                "severity": "HIGH",
                "observed_value": "; ".join(date_err_detail),
                "expected_value": "ClaimStart <= ClaimEnd, Admission <= Discharge",
                "evidence_description": f"Claim dates violate clinical sequencing: {'; '.join(date_err_detail)}.",
                "historical_records": [],
                "detection_timestamp": now_utc_str,
                "rule_version": RULE_VERSION,
            })
        evaluated_checks.append({
            "rule_id": "INVALID_CLAIM_DATES",
            "rule_name": "Chronological Date Sequencing Check",
            "rule_category": "DATA_INTEGRITY",
            "status": "FLAGGED" if is_date_invalid else "PASSED",
            "description": "Validates ClaimStart <= ClaimEnd and Admission <= Discharge.",
            "details": "Chronological date sequence verified." if not is_date_invalid else "; ".join(date_err_detail),
        })

        # Rule 3C: UNUSUAL_CLAIM_DURATION (Preserved Legacy Rule)
        los = row.get("InpatientLOS", None)
        if (los is None or pd.isna(los)) and pd.notna(admit_dt) and pd.notna(disch_dt):
            los = (disch_dt - admit_dt).days

        los_flag = False
        los_details = "Length of stay within expected norms."
        if los is not None and pd.notna(los):
            try:
                los_val = float(los)
                if los_val < 0:
                    los_flag = True
                    flags.append("UNUSUAL_CLAIM_DURATION")
                    rules_triggered.append({
                        "rule_id": "UNUSUAL_CLAIM_DURATION",
                        "rule_name": "Negative Inpatient Length of Stay",
                        "rule_category": "TEMPORAL_ANOMALY",
                        "severity": "HIGH",
                        "observed_value": f"{los_val:.0f} days",
                        "expected_value": ">= 0 days",
                        "evidence_description": f"Inpatient length of stay is negative ({los_val:.0f} days).",
                        "historical_records": [],
                        "detection_timestamp": now_utc_str,
                        "rule_version": RULE_VERSION,
                    })
                    los_details = f"Negative length of stay ({los_val:.0f} days)."
                elif los_val > max_inpatient_los_days:
                    los_flag = True
                    flags.append("UNUSUAL_CLAIM_DURATION")
                    rules_triggered.append({
                        "rule_id": "UNUSUAL_CLAIM_DURATION",
                        "rule_name": "Extended Inpatient Length of Stay Outlier",
                        "rule_category": "TEMPORAL_ANOMALY",
                        "severity": "MEDIUM",
                        "observed_value": f"{los_val:.0f} days",
                        "expected_value": f"<= {max_inpatient_los_days} days (Clinical Benchmark)",
                        "evidence_description": (
                            f"Inpatient length of stay of {los_val:.0f} days strictly exceeds the normal "
                            f"acute care benchmark ({max_inpatient_los_days} days)."
                        ),
                        "historical_records": [],
                        "detection_timestamp": now_utc_str,
                        "rule_version": RULE_VERSION,
                    })
                    los_details = f"Length of stay {los_val:.0f} days exceeds {max_inpatient_los_days} day clinical threshold."
                else:
                    los_details = f"Length of stay: {los_val:.0f} days (within {max_inpatient_los_days} day threshold)."
            except (ValueError, TypeError):
                warnings.append("Unable to parse length of stay as numerical value.")
                los_details = "Unable to evaluate length of stay."
        evaluated_checks.append({
            "rule_id": "UNUSUAL_CLAIM_DURATION",
            "rule_name": "Inpatient Length of Stay Clinical Norm Check",
            "rule_category": "TEMPORAL_ANOMALY",
            "status": "FLAGGED" if los_flag else ("PASSED" if los is not None else "NOT_APPLICABLE"),
            "description": "Checks for negative or excessive (> 30 days) inpatient stay durations.",
            "details": los_details,
        })

        # Rule 3D: Cross-Provider Active Service Date Overlap (New Phase 2.2)
        cross_prov_overlap_flag = False
        concurrent_inpatient_flag = False
        overlap_records = []
        if start_d and end_d and patient_hist_df is not None and not patient_hist_df.empty:
            for _, prow in patient_hist_df.iterrows():
                p_cid = str(prow.get("ClaimID", "")).strip()
                if p_cid == claim_id:
                    continue
                p_prov = str(prow.get("ProviderID", "")).strip()
                # Must be a DIFFERENT provider for cross-provider conflict
                if p_prov == provider_id:
                    continue
                p_sdt = _safe_date(prow.get("_ClaimStartDt") if "_ClaimStartDt" in prow else prow.get("ClaimStartDt"))
                p_edt = _safe_date(prow.get("_ClaimEndDt") if "_ClaimEndDt" in prow else prow.get("ClaimEndDt")) or p_sdt
                if p_sdt and p_edt:
                    # Check interval overlap: max(start1, start2) <= min(end1, end2)
                    if max(start_d, p_sdt) <= min(end_d, p_edt):
                        cross_prov_overlap_flag = True
                        p_type = str(prow.get("ClaimType", "Outpatient")).capitalize()
                        rec_item = {
                            "conflicting_claim_id": p_cid,
                            "conflicting_provider": p_prov,
                            "conflicting_period": f"{p_sdt} to {p_edt}",
                            "conflicting_claim_type": p_type,
                        }
                        overlap_records.append(rec_item)
                        # Check concurrent inpatient stays
                        if claim_type.lower() == "inpatient" and p_type.lower() == "inpatient":
                            concurrent_inpatient_flag = True

        if cross_prov_overlap_flag:
            flags.append("TEMP_CROSS_PROVIDER_OVERLAP")
            overlap_summary_parts = [
                f"Claim {r.get('conflicting_claim_id')} at {r.get('conflicting_provider')} ({r.get('conflicting_period')})"
                for r in overlap_records[:3]
            ]
            overlap_summary_str = "; ".join(overlap_summary_parts)
            rules_triggered.append({
                "rule_id": "TEMP_CROSS_PROVIDER_OVERLAP",
                "rule_name": "Cross-Provider Active Service Date Overlap",
                "rule_category": "TEMPORAL_ANOMALY",
                "severity": "HIGH",
                "observed_value": f"Overlaps with {len(overlap_records)} claim(s) at different provider(s)",
                "expected_value": "Zero active service overlaps between different healthcare facilities",
                "evidence_description": (
                    f"The beneficiary has another active claim with an overlapping service period at a different provider: "
                    f"{overlap_summary_str}."
                ),
                "historical_records": overlap_records,
                "detection_timestamp": now_utc_str,
                "rule_version": RULE_VERSION,
            })

        if concurrent_inpatient_flag:
            flags.append("TEMP_CONCURRENT_INPATIENT_STAY")
            rules_triggered.append({
                "rule_id": "TEMP_CONCURRENT_INPATIENT_STAY",
                "rule_name": "Concurrent Inpatient Admissions Across Multiple Facilities",
                "rule_category": "TEMPORAL_ANOMALY",
                "severity": "HIGH",
                "observed_value": "Multiple active inpatient hospitalizations",
                "expected_value": "Single concurrent inpatient admission per beneficiary",
                "evidence_description": (
                    f"Beneficiary is recorded as concurrently admitted to multiple inpatient hospital facilities "
                    f"during the period {start_d} to {end_d}. A patient cannot be physically hospitalized in multiple facilities simultaneously."
                ),
                "historical_records": [r for r in overlap_records if r.get("conflicting_claim_type", "").lower() == "inpatient"],
                "detection_timestamp": now_utc_str,
                "rule_version": RULE_VERSION,
            })

        evaluated_checks.append({
            "rule_id": "TEMP_CROSS_PROVIDER_OVERLAP",
            "rule_name": "Cross-Provider Service Date Overlap Check",
            "rule_category": "TEMPORAL_ANOMALY",
            "status": "FLAGGED" if cross_prov_overlap_flag else "PASSED",
            "description": "Verifies that beneficiary does not have active concurrent claims at competing providers.",
            "details": f"{len(overlap_records)} overlapping encounter(s) detected across providers." if cross_prov_overlap_flag else "No cross-provider date conflicts.",
        })

        # =====================================================================
        # CATEGORY 4: PROVIDER BEHAVIOR PATTERNS
        # =====================================================================

        # Rule 4A: HIGH_SAME_DAY_ACTIVITY (Preserved Legacy Rule)
        same_day_flag = False
        same_day_cnt = 1
        if pd.notna(start_dt):
            same_day_cnt = same_day_counts.get((provider_id, start_dt), 1)
            same_day_col = row.get("same_day_repeated_claims", row.get("SameDayRepeatedClaims", None))
            if same_day_cnt >= 10 or (same_day_col is not None and float(same_day_col or 0) >= 10):
                same_day_flag = True
                flags.append("HIGH_SAME_DAY_ACTIVITY")
                rules_triggered.append({
                    "rule_id": "HIGH_SAME_DAY_ACTIVITY",
                    "rule_name": "Concentrated Same-Day Claim Submissions",
                    "rule_category": "PROVIDER_BEHAVIOR",
                    "severity": "MEDIUM",
                    "observed_value": f"{same_day_cnt} claims",
                    "expected_value": "< 10 claims submitted on single service date",
                    "evidence_description": (
                        f"Provider submitted {same_day_cnt} claims on the single date {start_d}, "
                        f"reflecting a concentrated claim submission spike."
                    ),
                    "historical_records": [{"service_date": str(start_d), "daily_claims_count": same_day_cnt}],
                    "detection_timestamp": now_utc_str,
                    "rule_version": RULE_VERSION,
                })
        evaluated_checks.append({
            "rule_id": "HIGH_SAME_DAY_ACTIVITY",
            "rule_name": "Concentrated Same-Day Claim Volume Check",
            "rule_category": "PROVIDER_BEHAVIOR",
            "status": "FLAGGED" if same_day_flag else "PASSED",
            "description": "Detects high-volume billing bursts (>= 10 claims on a single date) by a provider.",
            "details": f"{same_day_cnt} claim(s) submitted for {start_d}.",
        })

        # Rule 4B: REPEAT_BENEFICIARY (Preserved Legacy Rule)
        repeat_bene_flag = False
        b_cnt = 1
        if bene_id:
            b_cnt = bene_counts.get((provider_id, bene_id), 1)
            repeat_col_val = row.get("repeat_bene_count", row.get("RepeatBeneCount", None))
            if b_cnt > repeat_bene_threshold or (repeat_col_val is not None and float(repeat_col_val or 0) > repeat_bene_threshold):
                repeat_bene_flag = True
                flags.append("REPEAT_BENEFICIARY")
                rules_triggered.append({
                    "rule_id": "REPEAT_BENEFICIARY",
                    "rule_name": "Unusual Repeat Beneficiary Billing Concentration",
                    "rule_category": "PROVIDER_BEHAVIOR",
                    "severity": "MEDIUM",
                    "observed_value": f"{b_cnt} claims",
                    "expected_value": f"<= {repeat_bene_threshold} repeat encounters",
                    "evidence_description": (
                        f"Beneficiary {bene_id} has {b_cnt} total claims recorded with provider {provider_id} "
                        f"(including cumulative provider history), exceeding the normal repeat threshold of {repeat_bene_threshold}."
                    ),
                    "historical_records": [{"beneficiary_id": bene_id, "provider_claim_count": b_cnt}],
                    "detection_timestamp": now_utc_str,
                    "rule_version": RULE_VERSION,
                })
        else:
            warnings.append("Beneficiary identifier missing; repeat beneficiary rule skipped.")

        evaluated_checks.append({
            "rule_id": "REPEAT_BENEFICIARY",
            "rule_name": "Repeat Beneficiary Frequency Check",
            "rule_category": "PROVIDER_BEHAVIOR",
            "status": "FLAGGED" if repeat_bene_flag else ("PASSED" if bene_id else "UNAVAILABLE"),
            "description": "Identifies excessive repeat billing for the same patient at a single provider.",
            "details": f"{b_cnt} claim(s) for beneficiary {bene_id or 'UNKNOWN'} with provider {provider_id}.",
        })

        # Rule 4C: Sudden Provider Volume Surge (New Phase 2.2)
        prov_surge_flag = False
        prov_surge_details = "Insufficient provider history (< 10 claims) for volume baseline."
        if hist_df_clean is not None and not hist_df_clean.empty and "_ClaimStartDt" in hist_df_clean.columns:
            valid_hist_dates = hist_df_clean["_ClaimStartDt"].dropna()
            if len(valid_hist_dates) >= 10:
                daily_counts = valid_hist_dates.dt.date.value_counts()
                hist_daily_mean = float(daily_counts.mean())
                curr_day_cnt = float(same_day_cnt)
                if curr_day_cnt >= 5 and curr_day_cnt > hist_daily_mean * 3.0:
                    prov_surge_flag = True
                    flags.append("PROV_SUDDEN_VOLUME_SURGE")
                    rules_triggered.append({
                        "rule_id": "PROV_SUDDEN_VOLUME_SURGE",
                        "rule_name": "Abnormal Provider Daily Submission Volume Surge",
                        "rule_category": "PROVIDER_BEHAVIOR",
                        "severity": "MEDIUM",
                        "observed_value": f"{curr_day_cnt:.0f} claims on {start_d}",
                        "expected_value": f"Provider historical average: {hist_daily_mean:.1f} claims/day",
                        "evidence_description": (
                            f"Provider submitted {curr_day_cnt:.0f} claims on {start_d}, strictly exceeding 3x their "
                            f"historical baseline daily average ({hist_daily_mean:.1f} claims/day across {len(valid_hist_dates)} prior claims)."
                        ),
                        "historical_records": [{"prior_total_claims": len(valid_hist_dates), "daily_average": hist_daily_mean}],
                        "detection_timestamp": now_utc_str,
                        "rule_version": RULE_VERSION,
                    })
                    prov_surge_details = f"Daily count {curr_day_cnt:.0f} exceeds 3x baseline ({hist_daily_mean:.1f}/day)."
                else:
                    prov_surge_details = f"Within normal volume bounds (baseline avg: {hist_daily_mean:.1f}/day)."
        evaluated_checks.append({
            "rule_id": "PROV_SUDDEN_VOLUME_SURGE",
            "rule_name": "Provider Volume Surge Anomaly Check",
            "rule_category": "PROVIDER_BEHAVIOR",
            "status": "FLAGGED" if prov_surge_flag else ("PASSED" if "Within" in prov_surge_details else "NOT_APPLICABLE"),
            "description": "Compares daily claim volume against the provider's historical daily average.",
            "details": prov_surge_details,
        })

        # =====================================================================
        # CATEGORY 5: BENEFICIARY BEHAVIOR ANOMALIES
        # =====================================================================

        # Rule 5A: Multi-Provider Hopping within 30 Days (New Phase 2.2)
        bene_hopping_flag = False
        bene_hopping_records = []
        if bene_id and start_d and patient_hist_df is not None and not patient_hist_df.empty:
            recent_provs = {provider_id}
            for _, prow in patient_hist_df.iterrows():
                p_sdt = _safe_date(prow.get("_ClaimStartDt") if "_ClaimStartDt" in prow else prow.get("ClaimStartDt"))
                p_prov = _safe_str(prow.get("ProviderID"))
                if p_sdt and p_prov:
                    delta_days = abs((start_d - p_sdt).days)
                    if delta_days <= 30:
                        recent_provs.add(p_prov)
                        bene_hopping_records.append({
                            "provider_id": p_prov,
                            "service_date": str(p_sdt),
                            "days_from_claim": delta_days,
                        })
            if len(recent_provs) >= 3:
                bene_hopping_flag = True
                flags.append("BENE_MULTI_PROVIDER_HOPPING")
                rules_triggered.append({
                    "rule_id": "BENE_MULTI_PROVIDER_HOPPING",
                    "rule_name": "High-Frequency Multi-Provider Hopping (30-Day Window)",
                    "rule_category": "BENEFICIARY_BEHAVIOR",
                    "severity": "MEDIUM",
                    "observed_value": f"{len(recent_provs)} distinct providers in 30 days",
                    "expected_value": "< 3 distinct providers in a rolling 30-day window",
                    "evidence_description": (
                        f"Beneficiary {bene_id} received services across {len(recent_provs)} distinct provider organizations "
                        f"({', '.join(sorted(list(recent_provs)))}) within a 30-day window, suggesting doctor shopping or fragmented care."
                    ),
                    "historical_records": bene_hopping_records,
                    "detection_timestamp": now_utc_str,
                    "rule_version": RULE_VERSION,
                })
        evaluated_checks.append({
            "rule_id": "BENE_MULTI_PROVIDER_HOPPING",
            "rule_name": "Beneficiary Multi-Provider Hopping Check",
            "rule_category": "BENEFICIARY_BEHAVIOR",
            "status": "FLAGGED" if bene_hopping_flag else "PASSED",
            "description": "Flags beneficiaries visiting >= 3 distinct healthcare providers within 30 days.",
            "details": f"{len(bene_hopping_records)} recent encounter(s) across providers in 30-day window." if bene_id else "Beneficiary ID missing.",
        })

        # Rule 5B: Beneficiary High Claim Velocity (New Phase 2.2)
        bene_velocity_flag = False
        bene_recent_claims_count = 1
        if bene_id and start_d and patient_hist_df is not None and not patient_hist_df.empty:
            for _, prow in patient_hist_df.iterrows():
                p_sdt = _safe_date(prow.get("_ClaimStartDt") if "_ClaimStartDt" in prow else prow.get("ClaimStartDt"))
                if p_sdt and abs((start_d - p_sdt).days) <= 30:
                    bene_recent_claims_count += 1
            if bene_recent_claims_count >= 6:
                bene_velocity_flag = True
                flags.append("BENE_HIGH_CLAIM_VELOCITY")
                rules_triggered.append({
                    "rule_id": "BENE_HIGH_CLAIM_VELOCITY",
                    "rule_name": "Abnormal Beneficiary 30-Day Claim Velocity",
                    "rule_category": "BENEFICIARY_BEHAVIOR",
                    "severity": "LOW",
                    "observed_value": f"{bene_recent_claims_count} claims in 30 days",
                    "expected_value": "< 6 claims in a 30-day window",
                    "evidence_description": (
                        f"Beneficiary {bene_id} generated {bene_recent_claims_count} total claims across all providers "
                        f"within a 30-day rolling window."
                    ),
                    "historical_records": [{"claims_in_30_days": bene_recent_claims_count}],
                    "detection_timestamp": now_utc_str,
                    "rule_version": RULE_VERSION,
                })
        evaluated_checks.append({
            "rule_id": "BENE_HIGH_CLAIM_VELOCITY",
            "rule_name": "Beneficiary 30-Day Encounter Velocity Check",
            "rule_category": "BENEFICIARY_BEHAVIOR",
            "status": "FLAGGED" if bene_velocity_flag else "PASSED",
            "description": "Checks if beneficiary generated >= 6 claims within a 30-day period.",
            "details": f"{bene_recent_claims_count} claim(s) in 30-day window.",
        })

        # =====================================================================
        # CATEGORY 6: CLINICAL CODING & BILLING INTEGRITY
        # =====================================================================

        # Rule 6A: MISSING_PHYSICIAN_INFO (Preserved Legacy Rule)
        attending = row.get("AttendingPhysician", None)
        operating = row.get("OperatingPhysician", None)
        other_phys = row.get("OtherPhysician", None)

        attending_missing = _safe_str(attending) is None
        operating_missing = _safe_str(operating) is None
        other_missing = _safe_str(other_phys) is None

        phys_flag = False
        phys_details = "Physician identifiers recorded."
        if attending_missing and operating_missing and other_missing:
            phys_flag = True
            flags.append("MISSING_PHYSICIAN_INFO")
            rules_triggered.append({
                "rule_id": "MISSING_PHYSICIAN_INFO",
                "rule_name": "All Physician Identifiers Omitted",
                "rule_category": "DATA_INTEGRITY",
                "severity": "MEDIUM",
                "observed_value": "All physician IDs null",
                "expected_value": "At least one attending or operating physician identifier required",
                "evidence_description": "Claim does not specify an attending, operating, or other physician identifier.",
                "historical_records": [],
                "detection_timestamp": now_utc_str,
                "rule_version": RULE_VERSION,
            })
            phys_details = "All physician IDs null."
        elif claim_type.lower() == "inpatient" and attending_missing:
            phys_flag = True
            flags.append("MISSING_PHYSICIAN_INFO")
            rules_triggered.append({
                "rule_id": "MISSING_PHYSICIAN_INFO",
                "rule_name": "Inpatient Attending Physician Missing",
                "rule_category": "DATA_INTEGRITY",
                "severity": "MEDIUM",
                "observed_value": "Attending physician null",
                "expected_value": "Attending physician mandatory for inpatient admissions",
                "evidence_description": "Inpatient institutional claim lacks mandatory attending physician identifier.",
                "historical_records": [],
                "detection_timestamp": now_utc_str,
                "rule_version": RULE_VERSION,
            })
            phys_details = "Inpatient attending physician missing."

        evaluated_checks.append({
            "rule_id": "MISSING_PHYSICIAN_INFO",
            "rule_name": "Physician Identification Completeness Check",
            "rule_category": "DATA_INTEGRITY",
            "status": "FLAGGED" if phys_flag else "PASSED",
            "description": "Verifies that attending or operating physician identifiers are present.",
            "details": phys_details,
        })

        # Rule 6B: MISSING_DIAGNOSIS_INFO (Preserved Legacy Rule)
        diag_missing = False
        if not present_diag_cols:
            warnings.append("Diagnosis code columns not present in input claims data; diagnosis completeness check omitted.")
            evaluated_checks.append({
                "rule_id": "MISSING_DIAGNOSIS_INFO",
                "rule_name": "Diagnosis Code Completeness Check",
                "rule_category": "DATA_INTEGRITY",
                "status": "UNAVAILABLE",
                "description": "Verifies that at least one diagnosis code is provided.",
                "details": "Diagnosis columns not provided in claim input.",
            })
        else:
            if len(claim_diag_codes) == 0:
                diag_missing = True
                flags.append("MISSING_DIAGNOSIS_INFO")
                rules_triggered.append({
                    "rule_id": "MISSING_DIAGNOSIS_INFO",
                    "rule_name": "Missing Diagnosis Codes",
                    "rule_category": "DATA_INTEGRITY",
                    "severity": "MEDIUM",
                    "observed_value": 0,
                    "expected_value": ">= 1 diagnosis code required",
                    "evidence_description": "Claim lacks any valid primary, admitting, or secondary diagnosis codes across diagnosis columns.",
                    "historical_records": [],
                    "detection_timestamp": now_utc_str,
                    "rule_version": RULE_VERSION,
                })
            evaluated_checks.append({
                "rule_id": "MISSING_DIAGNOSIS_INFO",
                "rule_name": "Diagnosis Code Completeness Check",
                "rule_category": "DATA_INTEGRITY",
                "status": "FLAGGED" if diag_missing else "PASSED",
                "description": "Verifies that at least one valid diagnosis code is supplied.",
                "details": f"{len(claim_diag_codes)} diagnosis code(s) recorded." if not diag_missing else "Zero diagnosis codes present.",
            })

        # Rule 6C: MISSING_PROCEDURE_INFO (Preserved Legacy Rule)
        proc_missing = False
        if not present_proc_cols:
            warnings.append("Procedure code columns not present in input claims data; procedure completeness check omitted.")
            evaluated_checks.append({
                "rule_id": "MISSING_PROCEDURE_INFO",
                "rule_name": "Procedure Code Completeness Check",
                "rule_category": "DATA_INTEGRITY",
                "status": "UNAVAILABLE",
                "description": "Verifies procedure codes on surgical/inpatient claims.",
                "details": "Procedure columns not provided in claim input.",
            })
        else:
            if len(claim_proc_codes) == 0 and (claim_type.lower() == "inpatient" or "ClmProcedureCode_1" in present_proc_cols):
                proc_missing = True
                flags.append("MISSING_PROCEDURE_INFO")
                rules_triggered.append({
                    "rule_id": "MISSING_PROCEDURE_INFO",
                    "rule_name": "Missing Procedure Codes",
                    "rule_category": "DATA_INTEGRITY",
                    "severity": "LOW",
                    "observed_value": 0,
                    "expected_value": ">= 1 procedure code expected for surgical or procedure reporting",
                    "evidence_description": "No procedure codes recorded across procedure columns (ClmProcedureCode_1 through ClmProcedureCode_6).",
                    "historical_records": [],
                    "detection_timestamp": now_utc_str,
                    "rule_version": RULE_VERSION,
                })
            evaluated_checks.append({
                "rule_id": "MISSING_PROCEDURE_INFO",
                "rule_name": "Procedure Code Completeness Check",
                "rule_category": "DATA_INTEGRITY",
                "status": "FLAGGED" if proc_missing else "PASSED",
                "description": "Verifies procedure codes on inpatient or surgical claims.",
                "details": f"{len(claim_proc_codes)} procedure code(s) recorded." if not proc_missing else "Zero procedure codes present.",
            })

        if diag_missing and proc_missing:
            flags.append("MISSING_DIAGNOSIS_PROCEDURE")

        # Rule 6D: ICD Code Syntax & Coding Version Boundary Check (New Phase 2.2)
        clin_version_flag = False
        clin_syntax_flag = False
        invalid_syntax_codes = []
        version_discrepancy_codes = []

        if claim_diag_codes:
            for dc_val in claim_diag_codes:
                clean_code = dc_val.replace(".", "").strip().upper()
                # 1. Version compatibility check: ICD-10 code billed prior to Oct 1, 2015 effective date
                if start_d and start_d < ICD10_CUTOFF_DATE:
                    # Historical CMS Part-B era: ICD-10 was not yet effective
                    if ICD10_PATTERN.match(dc_val) and not ICD9_DIAG_PATTERN.match(clean_code):
                        version_discrepancy_codes.append(f"{dc_val} (ICD-10 code billed on service date {start_d} prior to official CMS Oct 1, 2015 adoption)")

                # 2. Syntax validity check (ICD-9 syntax for 2008-2009 data)
                if not ICD9_DIAG_PATTERN.match(clean_code) and not ICD10_PATTERN.match(dc_val):
                    invalid_syntax_codes.append(dc_val)

        if version_discrepancy_codes:
            clin_version_flag = True
            flags.append("CLIN_CODE_VERSION_DISCREPANCY")
            rules_triggered.append({
                "rule_id": "CLIN_CODE_VERSION_DISCREPANCY",
                "rule_name": "ICD Coding Version and Effective Date Discrepancy",
                "rule_category": "CLINICAL_INTEGRITY",
                "severity": "LOW",
                "observed_value": "; ".join(version_discrepancy_codes),
                "expected_value": "ICD-9 effective before Oct 1, 2015; ICD-10 effective on/after Oct 1, 2015",
                "evidence_description": (
                    f"Clinical diagnosis code(s) violate CMS coding version effective boundaries: "
                    f"{'; '.join(version_discrepancy_codes)}."
                ),
                "historical_records": [],
                "detection_timestamp": now_utc_str,
                "rule_version": RULE_VERSION,
            })

        if invalid_syntax_codes:
            clin_syntax_flag = True
            flags.append("CLIN_INVALID_ICD9_SYNTAX")
            rules_triggered.append({
                "rule_id": "CLIN_INVALID_ICD9_SYNTAX",
                "rule_name": "Invalid Diagnosis Code Syntax",
                "rule_category": "CLINICAL_INTEGRITY",
                "severity": "LOW",
                "observed_value": f"Malformed code(s): {', '.join(invalid_syntax_codes)}",
                "expected_value": "Standard ICD-9-CM syntax (3-5 alphanumeric characters) or valid ICD-10",
                "evidence_description": (
                    f"Diagnosis code(s) '{', '.join(invalid_syntax_codes)}' do not conform to authoritative "
                    f"ICD coding structural specifications."
                ),
                "historical_records": [],
                "detection_timestamp": now_utc_str,
                "rule_version": RULE_VERSION,
            })

        evaluated_checks.append({
            "rule_id": "CLIN_CODE_VERSION_DISCREPANCY",
            "rule_name": "ICD Coding Version and Effective Date Validation",
            "rule_category": "CLINICAL_INTEGRITY",
            "status": "FLAGGED" if clin_version_flag else ("PASSED" if claim_diag_codes else "NOT_APPLICABLE"),
            "description": "Verifies that diagnosis codes conform to effective CMS date boundaries (ICD-9 vs ICD-10).",
            "details": f"Version discrepancies: {version_discrepancy_codes}." if clin_version_flag else "ICD version consistent with service date.",
        })

        evaluated_checks.append({
            "rule_id": "CLIN_INVALID_ICD9_SYNTAX",
            "rule_name": "Diagnosis Code Syntax Verification",
            "rule_category": "CLINICAL_INTEGRITY",
            "status": "FLAGGED" if clin_syntax_flag else ("PASSED" if claim_diag_codes else "NOT_APPLICABLE"),
            "description": "Verifies diagnosis code syntax against authoritative ICD structure.",
            "details": f"Malformed codes: {invalid_syntax_codes}." if clin_syntax_flag else "All diagnosis codes conform to standard syntax.",
        })

        # Rule 6E & 6F: Explicit Dependency Boundaries for NCCI and MUE Edits
        evaluated_checks.append({
            "rule_id": "CLIN_NCCI_PROCEDURE_EDITS",
            "rule_name": "National Correct Coding Initiative (NCCI) Procedure Edits",
            "rule_category": "CLINICAL_INTEGRITY",
            "status": "UNAVAILABLE_MISSING_DEPENDENCY",
            "description": "Evaluates mutually exclusive CPT/HCPCS procedure pairs and unbundling edits.",
            "details": "Authoritative dependency unavailable: NCCI edit files require 5-character CPT/HCPCS modifier pairs absent in this CMS dataset.",
        })
        evaluated_checks.append({
            "rule_id": "CLIN_MEDICALLY_UNLIKELY_UNITS",
            "rule_name": "Medically Unlikely Units (MUE) Validation",
            "rule_category": "CLINICAL_INTEGRITY",
            "status": "UNAVAILABLE_MISSING_DEPENDENCY",
            "description": "Checks maximum allowable units of service per procedure code on a single day of service.",
            "details": "Authoritative dependency unavailable: MUE reference tables require CPT/HCPCS unit quantities absent in this CMS dataset.",
        })

        # =====================================================================
        # ANOMALY SCORE & SUMMARY COMPILATION
        # =====================================================================
        # Calculate mathematically documented evidence score
        anomaly_score = sum(SEVERITY_WEIGHTS.get(r.get("severity", "LOW"), 0.5) for r in rules_triggered)
        anomaly_score = round(anomaly_score, 2)

        is_anomalous = len(flags) > 0
        anomaly_count = len(flags)

        disclaimer = CLAIM_DISCLAIMER if is_anomalous else CLEAN_CLAIM_DISCLAIMER

        results.append({
            "claim_id": claim_id,
            "provider_id": provider_id,
            "beneficiary_id": bene_id,
            "reimbursement_amount": reimbursement,
            "claim_type": claim_type,
            "anomaly_flags": flags,
            "is_anomalous": is_anomalous,
            "anomaly_count": anomaly_count,
            "anomaly_score": anomaly_score,
            "rules_triggered": rules_triggered,
            "evaluated_checks": evaluated_checks,
            "data_availability_warnings": warnings if warnings else None,
            "disclaimer": disclaimer,
            "engine_version": RULE_VERSION,
        })

    return results
