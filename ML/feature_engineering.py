"""
HealthGuard AI - Phase 1: Provider-Level Feature Engineering
=============================================================
Builds provider-level tabular features, peer percentile reference tables
(fitted strictly on training providers), and monthly LSTM sequences
with explicit valid-month masks.

Usage:
    python ML/feature_engineering.py
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
REP  = ROOT / "reports"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("feature_engineering")

RANDOM_SEED = 42

DIAG_COLS = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
PROC_COLS  = [f"ClmProcedureCode_{i}" for i in range(1, 7)]

CHRONIC_COLS = [
    "ChronicCond_Alzheimer", "ChronicCond_Heartfailure",
    "ChronicCond_KidneyDisease", "ChronicCond_Cancer",
    "ChronicCond_ObstrPulmonary", "ChronicCond_Depression",
    "ChronicCond_Diabetes", "ChronicCond_IschemicHeart",
    "ChronicCond_Osteoporasis", "ChronicCond_rheumatoidarthritis",
    "ChronicCond_stroke",
]


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _cv(series: pd.Series) -> float:
    """Coefficient of variation (std / mean), 0 if mean is 0 or series has <= 1 observations."""
    if len(series) <= 1:
        return 0.0
    m = series.mean()
    s = series.std()
    if pd.isna(s) or pd.isna(m) or m == 0:
        return 0.0
    val = float(s / m)
    return 0.0 if np.isnan(val) else val


def _shannon_entropy(series: pd.Series) -> float:
    """Compute Shannon entropy from value counts."""
    counts = series.value_counts()
    n = counts.sum()
    if n <= 1 or len(counts) <= 1:
        return 0.0
    p = counts / n
    return float(-np.sum(p * np.log2(p)))


# ══════════════════════════════════════════════════════════════════════════════
# PEER REFERENCE TABLE (Fitted on Training Providers Only)
# ══════════════════════════════════════════════════════════════════════════════

def compute_peer_reference_table(train_features: pd.DataFrame) -> dict:
    """
    Compute comprehensive peer percentile reference distributions
    strictly using training providers.
    Never calculates statistics using validation or test providers.
    Includes mean, std, min, P10, P25, P50 (median), P75, P90, P95, and max
    for all numerical provider features so explanations and scoring can
    retrieve actual training peer comparisons.
    """
    exclude = {"ProviderID", "PotentialFraud"}
    numeric_cols = [
        c for c in train_features.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    reimb = train_features["total_reimbursement"]
    avg_reimb = train_features["avg_reimbursement"]
    claims = train_features["total_claim_count"]

    ref_table = {
        "fit_split": "train",
        "n_training_providers": int(len(train_features)),
        "median_total_reimbursement": float(reimb.median()),
        "mean_total_reimbursement": float(reimb.mean()),
        "std_total_reimbursement": float(reimb.std()),
        "p10_total_reimbursement": float(reimb.quantile(0.10)),
        "p25_total_reimbursement": float(reimb.quantile(0.25)),
        "p75_total_reimbursement": float(reimb.quantile(0.75)),
        "p90_total_reimbursement": float(reimb.quantile(0.90)),
        "p95_total_reimbursement": float(reimb.quantile(0.95)),
        "median_avg_reimbursement": float(avg_reimb.median()),
        "median_claim_count": float(claims.median()),
        "feature_distributions": {},
    }

    percentiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    for col in numeric_cols:
        series = train_features[col].dropna()
        if len(series) == 0:
            continue
        q = series.quantile(percentiles)
        ref_table["feature_distributions"][col] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "p10": float(q[0.10]),
            "p25": float(q[0.25]),
            "p50": float(q[0.50]),
            "median": float(series.median()),
            "p75": float(q[0.75]),
            "p90": float(q[0.90]),
            "p95": float(q[0.95]),
            "max": float(series.max()),
        }

    log.info(f"  Fitted comprehensive peer reference table on {len(train_features)} training providers "
             f"across {len(ref_table['feature_distributions'])} numerical features.")
    return ref_table


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FEATURE BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_provider_features(claims_df: pd.DataFrame, peer_ref: dict = None) -> pd.DataFrame:
    """
    Aggregate claims to one row per ProviderID with feature groups:
      - Financial (including peer-relative reimbursement)
      - Behavioural (including Shannon entropies, LOS, anomalies)
      - Beneficiary (including per-claim age, coverage months)
      - Temporal

    Parameters
    ----------
    claims_df : pd.DataFrame
        Cleaned combined claims with beneficiary columns merged in.
    peer_ref : dict, optional
        Peer reference table computed strictly from training providers.
        If None, computed from the current dataframe (train split only).

    Returns
    -------
    pd.DataFrame
        One row per ProviderID, provider-level features only.
        Sensitive demographic columns (Race, Gender, State) are excluded.
    """
    if "ProviderID" not in claims_df.columns:
        raise ValueError("claims_df must contain 'ProviderID'")

    df = claims_df.copy()

    # Ensure datetime
    for col in ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt", "DOD", "DOB"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Inpatient flag
    is_ip = (df["ClaimType"] == "Inpatient")
    df["_is_inpatient"] = is_ip.astype(int)

    log.info(f"  Building provider features for {df['ProviderID'].nunique():,} providers…")
    grouped = df.groupby("ProviderID")

    # ══════════════════════════════════════════════════════════════════════════
    # 1. FINANCIAL FEATURES
    # ══════════════════════════════════════════════════════════════════════════
    fin = grouped["InscClaimAmtReimbursed"].agg(
        total_reimbursement="sum",
        avg_reimbursement="mean",
        median_reimbursement="median",
        max_reimbursement="max",
        std_reimbursement="std",
    ).fillna(0)
    fin["cv_reimbursement"] = grouped["InscClaimAmtReimbursed"].apply(_cv)

    # Inpatient vs outpatient reimbursement
    fin["ip_total_reimbursement"] = (
        df[is_ip].groupby("ProviderID")["InscClaimAmtReimbursed"].sum()
    )
    fin["op_total_reimbursement"] = (
        df[~is_ip].groupby("ProviderID")["InscClaimAmtReimbursed"].sum()
    )
    fin[["ip_total_reimbursement", "op_total_reimbursement"]] = (
        fin[["ip_total_reimbursement", "op_total_reimbursement"]].fillna(0)
    )

    # Inpatient claim percentage
    claim_counts = grouped["ClaimID"].count().rename("total_claim_count")
    ip_counts = df[is_ip].groupby("ProviderID")["ClaimID"].count().rename("ip_claim_count")
    fin["ip_claim_pct"] = (ip_counts / claim_counts).fillna(0)

    # Average deductible
    if "DeductibleAmtPaid" in df.columns:
        fin["avg_deductible"] = grouped["DeductibleAmtPaid"].mean().fillna(0)
    else:
        fin["avg_deductible"] = 0.0

    # Peer relative reimbursement (relative to training providers median)
    if peer_ref is not None and "median_total_reimbursement" in peer_ref:
        peer_med = peer_ref["median_total_reimbursement"]
    else:
        peer_med = fin["total_reimbursement"].median()
    fin["reimbursement_to_peer_median"] = fin["total_reimbursement"] / (peer_med + 1e-6)

    # ══════════════════════════════════════════════════════════════════════════
    # 2. BEHAVIOURAL FEATURES
    # ══════════════════════════════════════════════════════════════════════════
    beh = pd.DataFrame(index=fin.index)
    beh["total_claim_count"] = claim_counts

    # Unique beneficiaries & claims per beneficiary
    beh["unique_beneficiaries"] = grouped["BeneID"].nunique()
    beh["claims_per_beneficiary"] = (
        beh["total_claim_count"] / beh["unique_beneficiaries"]
    ).fillna(0)

    # Repeat beneficiary ratio
    bene_claim_counts = df.groupby(["ProviderID", "BeneID"]).size().reset_index(name="cnt")
    repeat_benes = bene_claim_counts[bene_claim_counts["cnt"] > 1].groupby("ProviderID")["BeneID"].count()
    beh["repeat_bene_count"] = repeat_benes.reindex(beh.index, fill_value=0)
    beh["repeat_bene_ratio"] = (
        beh["repeat_bene_count"] / beh["unique_beneficiaries"]
    ).fillna(0)

    # Same-day repeated claims (same provider + same bene + same start date)
    if "ClaimStartDt" in df.columns:
        same_day = df.groupby(["ProviderID", "BeneID", "ClaimStartDt"]).size()
        same_day_multi = (same_day[same_day > 1]
                          .reset_index()
                          .groupby("ProviderID")
                          .size()
                          .rename("same_day_repeated_claims"))
        beh["same_day_repeated_claims"] = same_day_multi.reindex(beh.index, fill_value=0)
    else:
        beh["same_day_repeated_claims"] = 0

    # Missing physician ratio & distinct attending physicians
    if "AttendingPhysician_Missing" in df.columns:
        beh["missing_physician_ratio"] = grouped["AttendingPhysician_Missing"].mean().fillna(0)
    else:
        beh["missing_physician_ratio"] = 0.0

    if "AttendingPhysician" in df.columns:
        beh["distinct_physicians"] = grouped["AttendingPhysician"].nunique()
    else:
        beh["distinct_physicians"] = 0

    # Diagnosis code counts, diversity & entropy
    diag_present = [c for c in DIAG_COLS if c in df.columns]
    if diag_present:
        df["_diag_code_count"] = df[diag_present].notna().sum(axis=1)
        beh["avg_diagnosis_codes"] = grouped["_diag_code_count"].mean().fillna(0)
        all_diag = (
            df[["ProviderID"] + diag_present]
            .melt(id_vars="ProviderID", value_vars=diag_present, value_name="code")
            .dropna(subset=["code"])
        )
        if not all_diag.empty:
            beh["unique_diagnosis_codes"] = (
                all_diag.groupby("ProviderID")["code"].nunique()
                .reindex(beh.index, fill_value=0)
            )
            diag_counts = all_diag.groupby(["ProviderID", "code"]).size().reset_index(name="cnt")
            diag_totals = diag_counts.groupby("ProviderID")["cnt"].transform("sum")
            p_diag = diag_counts["cnt"] / diag_totals
            diag_counts["term"] = - p_diag * np.log2(p_diag)
            beh["diagnosis_code_entropy"] = (
                diag_counts.groupby("ProviderID")["term"].sum()
                .reindex(beh.index, fill_value=0.0)
            )
        else:
            beh["unique_diagnosis_codes"] = 0
            beh["diagnosis_code_entropy"] = 0.0
    else:
        beh["avg_diagnosis_codes"] = 0.0
        beh["unique_diagnosis_codes"] = 0
        beh["diagnosis_code_entropy"] = 0.0

    # Procedure code counts, diversity & entropy
    proc_present = [c for c in PROC_COLS if c in df.columns]
    if proc_present:
        df["_proc_code_count"] = df[proc_present].notna().sum(axis=1)
        beh["avg_procedure_codes"] = grouped["_proc_code_count"].mean().fillna(0)
        all_proc = (
            df[["ProviderID"] + proc_present]
            .melt(id_vars="ProviderID", value_vars=proc_present, value_name="code")
            .dropna(subset=["code"])
        )
        if not all_proc.empty:
            beh["unique_procedure_codes"] = (
                all_proc.groupby("ProviderID")["code"].nunique()
                .reindex(beh.index, fill_value=0)
            )
            proc_counts = all_proc.groupby(["ProviderID", "code"]).size().reset_index(name="cnt")
            proc_totals = proc_counts.groupby("ProviderID")["cnt"].transform("sum")
            p_proc = proc_counts["cnt"] / proc_totals
            proc_counts["term"] = - p_proc * np.log2(p_proc)
            beh["procedure_code_entropy"] = (
                proc_counts.groupby("ProviderID")["term"].sum()
                .reindex(beh.index, fill_value=0.0)
            )
        else:
            beh["unique_procedure_codes"] = 0
            beh["procedure_code_entropy"] = 0.0
    else:
        beh["avg_procedure_codes"] = 0.0
        beh["unique_procedure_codes"] = 0
        beh["procedure_code_entropy"] = 0.0

    # Inpatient Average Length of Stay (days)
    if "InpatientLOS" in df.columns:
        beh["avg_inpatient_length_of_stay"] = grouped["InpatientLOS"].mean().fillna(0.0)
    elif "AdmissionDt" in df.columns and "DischargeDt" in df.columns:
        valid_ip = (
            is_ip &
            df["AdmissionDt"].notna() &
            df["DischargeDt"].notna() &
            (df["DischargeDt"] >= df["AdmissionDt"])
        )
        los = (df.loc[valid_ip, "DischargeDt"] - df.loc[valid_ip, "AdmissionDt"]).dt.days
        beh["avg_inpatient_length_of_stay"] = (
            df.loc[valid_ip].assign(_los=los).groupby("ProviderID")["_los"].mean()
            .reindex(beh.index, fill_value=0.0)
        )
    else:
        beh["avg_inpatient_length_of_stay"] = 0.0

    # Date Anomaly Features (impossible claim dates & discharge before admission)
    if "ClaimDateAnomaly" in df.columns:
        beh["claim_date_anomaly_count"] = grouped["ClaimDateAnomaly"].sum().fillna(0)
        beh["claim_date_anomaly_ratio"] = grouped["ClaimDateAnomaly"].mean().fillna(0.0)
    else:
        beh["claim_date_anomaly_count"] = 0
        beh["claim_date_anomaly_ratio"] = 0.0

    if "DischargeBeforeAdmission" in df.columns:
        beh["discharge_before_admission_count"] = grouped["DischargeBeforeAdmission"].sum().fillna(0)
        beh["discharge_before_admission_ratio"] = grouped["DischargeBeforeAdmission"].mean().fillna(0.0)
    else:
        beh["discharge_before_admission_count"] = 0
        beh["discharge_before_admission_ratio"] = 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # 3. BENEFICIARY FEATURES
    # ══════════════════════════════════════════════════════════════════════════
    ben_feat = pd.DataFrame(index=fin.index)

    # Beneficiary age at claim date (calculated per claim, NOT fixed 2009 reference year)
    if "BeneAgeAtClaim" in df.columns:
        ben_feat["avg_beneficiary_age"] = grouped["BeneAgeAtClaim"].mean().fillna(0.0)
    elif "ClaimStartDt" in df.columns and "DOB" in df.columns:
        valid_dob = (
            df["ClaimStartDt"].notna() &
            df["DOB"].notna() &
            (df["ClaimStartDt"] >= df["DOB"])
        )
        claim_age = (df.loc[valid_dob, "ClaimStartDt"] - df.loc[valid_dob, "DOB"]).dt.days / 365.25
        ben_feat["avg_beneficiary_age"] = (
            df.loc[valid_dob].assign(_cage=claim_age).groupby("ProviderID")["_cage"].mean()
            .reindex(ben_feat.index, fill_value=0.0)
        )
    else:
        ben_feat["avg_beneficiary_age"] = 0.0

    if "ChronicCondCount" in df.columns:
        ben_feat["avg_chronic_cond_count"] = grouped["ChronicCondCount"].mean().fillna(0.0)
    else:
        ben_feat["avg_chronic_cond_count"] = 0.0

    if "RenalDiseaseIndicator" in df.columns:
        ben_feat["renal_disease_ratio"] = grouped["RenalDiseaseIndicator"].mean().fillna(0.0)
    else:
        ben_feat["renal_disease_ratio"] = 0.0

    if "IsDeceased" in df.columns:
        ben_feat["deceased_bene_ratio"] = grouped["IsDeceased"].mean().fillna(0.0)
    else:
        ben_feat["deceased_bene_ratio"] = 0.0

    if "ClaimAfterDeath" in df.columns:
        ben_feat["claims_after_death_ratio"] = grouped["ClaimAfterDeath"].mean().fillna(0.0)
        ben_feat["claims_after_death_count"] = grouped["ClaimAfterDeath"].sum().fillna(0)
    else:
        ben_feat["claims_after_death_ratio"] = 0.0
        ben_feat["claims_after_death_count"] = 0

    # Beneficiary coverage-month features
    if "NoOfMonths_PartACov" in df.columns:
        ben_feat["avg_part_a_coverage_months"] = grouped["NoOfMonths_PartACov"].mean().fillna(0.0)
        ben_feat["full_part_a_coverage_ratio"] = (
            (df["NoOfMonths_PartACov"] == 12).astype(int).groupby(df["ProviderID"]).mean()
            .reindex(ben_feat.index, fill_value=0.0)
        )
    else:
        ben_feat["avg_part_a_coverage_months"] = 0.0
        ben_feat["full_part_a_coverage_ratio"] = 0.0

    if "NoOfMonths_PartBCov" in df.columns:
        ben_feat["avg_part_b_coverage_months"] = grouped["NoOfMonths_PartBCov"].mean().fillna(0.0)
        ben_feat["full_part_b_coverage_ratio"] = (
            (df["NoOfMonths_PartBCov"] == 12).astype(int).groupby(df["ProviderID"]).mean()
            .reindex(ben_feat.index, fill_value=0.0)
        )
    else:
        ben_feat["avg_part_b_coverage_months"] = 0.0
        ben_feat["full_part_b_coverage_ratio"] = 0.0

    # Annual reimbursement & deductible info
    for col in ["IPAnnualReimbursementAmt", "IPAnnualDeductibleAmt",
                "OPAnnualReimbursementAmt", "OPAnnualDeductibleAmt"]:
        if col in df.columns:
            ben_feat[f"avg_{col.lower()}"] = grouped[col].mean().fillna(0.0)
        else:
            ben_feat[f"avg_{col.lower()}"] = 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # 4. TEMPORAL FEATURES
    # ══════════════════════════════════════════════════════════════════════════
    temp = pd.DataFrame(index=fin.index)

    if "ClaimStartDt" in df.columns:
        df["_ym"] = df["ClaimStartDt"].dt.to_period("M")
        temp["active_months"] = grouped["_ym"].nunique()

        monthly = (
            df.groupby(["ProviderID", "_ym"])["ClaimID"]
            .count()
            .reset_index(name="monthly_claim_count")
        )
        temp["avg_monthly_claims"] = (
            monthly.groupby("ProviderID")["monthly_claim_count"].mean()
            .reindex(temp.index, fill_value=0.0)
        )
        temp["max_monthly_claims"] = (
            monthly.groupby("ProviderID")["monthly_claim_count"].max()
            .reindex(temp.index, fill_value=0.0)
        )

        monthly_reimb = (
            df.groupby(["ProviderID", "_ym"])["InscClaimAmtReimbursed"]
            .sum()
            .reset_index(name="monthly_reimb")
        )
        temp["avg_monthly_reimbursement"] = (
            monthly_reimb.groupby("ProviderID")["monthly_reimb"].mean()
            .reindex(temp.index, fill_value=0.0)
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            temp["monthly_spike_ratio"] = np.where(
                temp["avg_monthly_claims"] > 0,
                temp["max_monthly_claims"] / temp["avg_monthly_claims"],
                0.0,
            )

        def _mean_abs_change(group):
            counts = group.set_index("_ym")["monthly_claim_count"].sort_index()
            return float(counts.diff().abs().mean()) if len(counts) > 1 else 0.0

        mtm_change = (
            monthly.groupby("ProviderID")
            .apply(_mean_abs_change, include_groups=False)
            .rename("avg_monthly_claim_change")
        )
        temp["avg_monthly_claim_change"] = mtm_change.reindex(temp.index, fill_value=0.0)

        def _avg_gap(group):
            dates = group["ClaimStartDt"].dropna().sort_values()
            if len(dates) < 2:
                return 0.0
            return float(dates.diff().dt.days.dropna().abs().mean())

        avg_gap = (
            df.groupby("ProviderID")
            .apply(_avg_gap, include_groups=False)
            .rename("avg_days_between_claims")
        )
        temp["avg_days_between_claims"] = avg_gap.reindex(temp.index, fill_value=0.0)
    else:
        for col in ["active_months", "avg_monthly_claims", "max_monthly_claims",
                    "avg_monthly_reimbursement", "monthly_spike_ratio",
                    "avg_monthly_claim_change", "avg_days_between_claims"]:
            temp[col] = 0.0

    # ── Combine all feature groups ────────────────────────────────────────────
    features = pd.concat([fin, beh, ben_feat, temp], axis=1)
    features.index.name = "ProviderID"
    features = features.reset_index()

    log.info(f"  Provider features shape: {features.shape}")
    return features


# ══════════════════════════════════════════════════════════════════════════════
# MONTHLY SEQUENCE & MASK BUILDER (for LSTM)
# ══════════════════════════════════════════════════════════════════════════════

SEQUENCE_FEATURES = [
    "monthly_claim_count",
    "log_total_reimbursement",
    "unique_beneficiaries",
    "ip_claim_pct",
    "avg_reimbursement",
    "repeat_bene_count",
]


def build_monthly_sequences(
    claims_df: pd.DataFrame,
    provider_ids: pd.Series,
    seq_len: int = 12,
) -> tuple:
    """
    Build fixed-length monthly provider sequences and matching valid-month masks.

    Masking logic:
      - For each provider, the observation window spans from first claim month (p_min)
        to last claim month (p_max).
      - Zero-claim months within this window are PRESERVED as actual zero-claim observation
        months with mask = 1.0 (valid). They are NOT marked as padding.
      - If window length n < seq_len, the sequence is prepadded with zeros;
        indices 0..(seq_len - n - 1) receive mask = 0.0 (padding).
      - If window length n >= seq_len, the most recent seq_len months are retained,
        and all months receive mask = 1.0 (valid).

    Parameters
    ----------
    claims_df   : Cleaned combined claims DataFrame.
    provider_ids: Provider IDs to include.
    seq_len     : Sequence length (default 12).

    Returns
    -------
    sequences : np.ndarray, shape (n_providers, seq_len, n_features), float32
    masks     : np.ndarray, shape (n_providers, seq_len), float32 (1.0 = valid, 0.0 = pad)
    provider_list : list of str, exact ProviderID row ordering
    metadata  : dict, metadata including feature names and mask definition
    """
    if "ClaimStartDt" not in claims_df.columns:
        raise ValueError("claims_df must contain 'ClaimStartDt'")

    df = claims_df[claims_df["ProviderID"].isin(provider_ids)].copy()
    df["ClaimStartDt"] = pd.to_datetime(df["ClaimStartDt"], errors="coerce")
    df["_ym"] = df["ClaimStartDt"].dt.to_period("M")
    is_ip = (df["ClaimType"] == "Inpatient")
    df["_is_inpatient"] = is_ip.astype(int)

    # Bene repeat count per month
    bene_month = df.groupby(["ProviderID", "_ym", "BeneID"]).size().reset_index(name="_bcnt")
    repeat_month = (
        bene_month[bene_month["_bcnt"] > 1]
        .groupby(["ProviderID", "_ym"])["BeneID"].count()
        .reset_index(name="repeat_bene_count")
    )

    # Monthly aggregation
    monthly = (
        df.groupby(["ProviderID", "_ym"])
        .agg(
            monthly_claim_count=("ClaimID", "count"),
            total_reimbursement=("InscClaimAmtReimbursed", "sum"),
            unique_beneficiaries=("BeneID", "nunique"),
            avg_reimbursement=("InscClaimAmtReimbursed", "mean"),
            ip_claim_count=("_is_inpatient", "sum"),
        )
        .reset_index()
    )

    monthly["ip_claim_pct"] = (
        monthly["ip_claim_count"] / monthly["monthly_claim_count"]
    ).fillna(0.0)
    monthly["log_total_reimbursement"] = np.log1p(monthly["total_reimbursement"])
    monthly = monthly.merge(repeat_month, on=["ProviderID", "_ym"], how="left")
    monthly["repeat_bene_count"] = monthly["repeat_bene_count"].fillna(0.0)

    n_features = len(SEQUENCE_FEATURES)
    provider_list = sorted(provider_ids.tolist())
    n_providers = len(provider_list)

    sequences = np.zeros((n_providers, seq_len, n_features), dtype=np.float32)
    masks = np.zeros((n_providers, seq_len), dtype=np.float32)

    global_min_period = df["_ym"].min() if not df.empty else None
    global_max_period = df["_ym"].max() if not df.empty else None

    # Group monthly data by ProviderID for fast lookup
    grouped_monthly = dict(tuple(monthly.groupby("ProviderID")))

    for i, pid in enumerate(provider_list):
        pdata = grouped_monthly.get(pid)
        if pdata is None or len(pdata) == 0:
            continue
        pdata = pdata.sort_values("_ym")

        p_min = pdata["_ym"].min()
        p_max = pdata["_ym"].max()
        full_range = pd.period_range(start=p_min, end=p_max, freq="M")
        full_df = pd.DataFrame({"_ym": full_range}).merge(
            pdata[["_ym"] + SEQUENCE_FEATURES], on="_ym", how="left"
        ).fillna(0.0)

        vals = full_df[SEQUENCE_FEATURES].values.astype(np.float32)
        n = len(vals)

        if n >= seq_len:
            sequences[i] = vals[-seq_len:]
            masks[i, :] = 1.0
        else:
            sequences[i, seq_len - n:] = vals
            masks[i, :seq_len - n] = 0.0   # Leading padding
            masks[i, seq_len - n:] = 1.0   # Active observed window (including zero-claim months)

    metadata = {
        "sequence_length": seq_len,
        "n_features": n_features,
        "feature_names": SEQUENCE_FEATURES,
        "feature_descriptions": {
            "monthly_claim_count":      "Number of claims submitted in the month",
            "log_total_reimbursement":  "log(1 + total reimbursement) for the month",
            "unique_beneficiaries":     "Distinct beneficiaries served in the month",
            "ip_claim_pct":             "Fraction of claims that are inpatient",
            "avg_reimbursement":        "Average per-claim reimbursement in the month",
            "repeat_bene_count":        "Beneficiaries with more than one claim in the month",
        },
        "mask_definition": {
            "shape": [n_providers, seq_len],
            "values": {
                "0.0": "Leading padding month (before provider's first observed claim)",
                "1.0": "Valid observation month (within provider's active window; includes actual zero-claim months)",
            },
        },
        "ordering": "Chronological (most recent seq_len months; leading zero-pad if shorter)",
        "global_period_range": [str(global_min_period), str(global_max_period)],
    }

    log.info(f"  Sequences shape: {sequences.shape}, Masks shape: {masks.shape}")
    return sequences, masks, provider_list, metadata


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE SPEC JSON
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_spec(features_df: pd.DataFrame) -> dict:
    """Create a JSON-serialisable feature specification."""
    spec = {}
    exclude = {"ProviderID", "PotentialFraud"}
    for col in features_df.columns:
        if col in exclude:
            continue
        dtype = str(features_df[col].dtype)

        if any(col.startswith(p) for p in [
            "total_reimb", "avg_reimb", "median_reimb", "max_reimb",
            "std_reimb", "cv_reimb", "ip_total", "op_total", "ip_claim_pct",
            "avg_deductible", "reimbursement_to_peer",
        ]):
            group = "financial"
        elif any(col.startswith(p) for p in [
            "total_claim", "unique_bene", "claims_per", "repeat_bene",
            "same_day", "missing_phys", "distinct_phys", "avg_diag",
            "unique_diag", "avg_proc", "unique_proc", "diagnosis_code_entropy",
            "procedure_code_entropy", "avg_inpatient_length", "claim_date_anomaly",
            "discharge_before_admission",
        ]):
            group = "behavioural"
        elif any(col.startswith(p) for p in [
            "avg_bene", "avg_chronic", "renal", "deceased", "claims_after",
            "avg_ip", "avg_op", "avg_part", "full_part",
        ]):
            group = "beneficiary"
        elif any(col.startswith(p) for p in [
            "active_month", "avg_monthly", "max_monthly", "monthly_spike",
            "avg_days",
        ]):
            group = "temporal"
        else:
            group = "other"

        spec[col] = {
            "dtype": dtype,
            "group": group,
            "meaning": _feature_description(col),
        }
    return spec


def _feature_description(col: str) -> str:
    desc_map = {
        "total_reimbursement":              "Total insurance reimbursement across all claims ($)",
        "avg_reimbursement":                "Mean reimbursement per claim ($)",
        "median_reimbursement":             "Median reimbursement per claim ($)",
        "max_reimbursement":                "Maximum single-claim reimbursement ($)",
        "std_reimbursement":                "Standard deviation of claim reimbursements ($)",
        "cv_reimbursement":                 "Coefficient of variation of reimbursements (std/mean)",
        "ip_total_reimbursement":           "Total reimbursement for inpatient claims ($)",
        "op_total_reimbursement":           "Total reimbursement for outpatient claims ($)",
        "ip_claim_pct":                     "Fraction of claims that are inpatient",
        "avg_deductible":                   "Average deductible amount paid per claim ($)",
        "reimbursement_to_peer_median":      "Provider total reimbursement relative to training peer median",
        "total_claim_count":                "Total number of claims submitted",
        "unique_beneficiaries":             "Number of distinct patients served",
        "claims_per_beneficiary":           "Average claims per patient",
        "repeat_bene_count":                "Patients with more than one claim at this provider",
        "repeat_bene_ratio":                "Fraction of patients with repeated claims",
        "same_day_repeated_claims":         "Count of same-day same-patient repeated claims",
        "missing_physician_ratio":          "Fraction of claims with missing attending physician",
        "distinct_physicians":              "Number of distinct attending physicians billed",
        "avg_diagnosis_codes":              "Average diagnosis codes per claim",
        "unique_diagnosis_codes":           "Distinct diagnosis codes used across all claims",
        "diagnosis_code_entropy":           "Shannon entropy of diagnosis code distribution",
        "avg_procedure_codes":              "Average procedure codes per claim",
        "unique_procedure_codes":           "Distinct procedure codes used across all claims",
        "procedure_code_entropy":           "Shannon entropy of procedure code distribution",
        "avg_inpatient_length_of_stay":      "Average length of stay in days for inpatient claims",
        "claim_date_anomaly_count":         "Count of claims with ClaimEndDt < ClaimStartDt",
        "claim_date_anomaly_ratio":         "Fraction of claims with ClaimEndDt < ClaimStartDt",
        "discharge_before_admission_count": "Count of inpatient claims with DischargeDt < AdmissionDt",
        "discharge_before_admission_ratio": "Fraction of inpatient claims with DischargeDt < AdmissionDt",
        "avg_beneficiary_age":              "Average patient age at actual claim date (years)",
        "avg_chronic_cond_count":           "Average number of chronic conditions per patient",
        "renal_disease_ratio":              "Fraction of patient-claims with renal disease",
        "deceased_bene_ratio":              "Fraction of patient-claims where patient is deceased",
        "claims_after_death_ratio":         "Fraction of claims submitted after patient death",
        "claims_after_death_count":         "Total claims submitted after patient death",
        "avg_part_a_coverage_months":       "Average Part A Medicare coverage months (0-12)",
        "full_part_a_coverage_ratio":       "Fraction of claims where patient had full 12-month Part A coverage",
        "avg_part_b_coverage_months":       "Average Part B Medicare coverage months (0-12)",
        "full_part_b_coverage_ratio":       "Fraction of claims where patient had full 12-month Part B coverage",
        "avg_ipannualreimbursementamt":     "Average beneficiary annual inpatient reimbursement ($)",
        "avg_ipannualdeductibleamt":        "Average beneficiary annual inpatient deductible ($)",
        "avg_opannualreimbursementamt":     "Average beneficiary annual outpatient reimbursement ($)",
        "avg_opannualdeductibleamt":        "Average beneficiary annual outpatient deductible ($)",
        "active_months":                    "Number of distinct calendar months with at least one claim",
        "avg_monthly_claims":               "Average claims per active calendar month",
        "max_monthly_claims":               "Peak claims in any single calendar month",
        "avg_monthly_reimbursement":        "Average total reimbursement per active month ($)",
        "monthly_spike_ratio":              "Peak month to average month claim count ratio",
        "avg_monthly_claim_change":         "Mean absolute month-to-month claim count change",
        "avg_days_between_claims":          "Average calendar days between consecutive claims",
    }
    return desc_map.get(col, f"Provider-level aggregation of {col}")


# ══════════════════════════════════════════════════════════════════════════════
# AUTOMATED CHECKS
# ══════════════════════════════════════════════════════════════════════════════

def run_feature_checks(
    train_feat, val_feat, test_feat,
    train_ids, val_ids, test_ids,
    seq_data: dict,
) -> list:
    """
    Run automated consistency checks on the feature and sequence datasets.
    Returns a list of check dicts.
    """
    checks = []

    def chk(name, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        checks.append({"check": name, "status": status, "detail": detail})
        log.info(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    # 1 row per provider
    chk("train_one_row_per_provider",
        train_feat["ProviderID"].nunique() == len(train_feat),
        f"{train_feat['ProviderID'].nunique()} unique / {len(train_feat)} rows")
    chk("val_one_row_per_provider",
        val_feat["ProviderID"].nunique() == len(val_feat),
        f"{val_feat['ProviderID'].nunique()} unique / {len(val_feat)} rows")
    chk("test_one_row_per_provider",
        test_feat["ProviderID"].nunique() == len(test_feat),
        f"{test_feat['ProviderID'].nunique()} unique / {len(test_feat)} rows")

    # Split overlap checks
    tv = set(train_ids) & set(val_ids)
    tt = set(train_ids) & set(test_ids)
    vt = set(val_ids) & set(test_ids)
    chk("no_train_val_overlap", len(tv) == 0, f"{len(tv)} overlapping")
    chk("no_train_test_overlap", len(tt) == 0, f"{len(tt)} overlapping")
    chk("no_val_test_overlap",   len(vt) == 0, f"{len(vt)} overlapping")

    # Column consistency
    same_cols = (set(train_feat.columns) == set(val_feat.columns) == set(test_feat.columns))
    chk("feature_column_consistency", same_cols,
        f"train:{len(train_feat.columns)} val:{len(val_feat.columns)} test:{len(test_feat.columns)}")

    # No all-NaN columns in train
    nan_cols = [c for c in train_feat.columns if train_feat[c].isna().all()]
    chk("no_all_nan_columns_train", len(nan_cols) == 0, f"all-NaN: {nan_cols}")

    # Protected attributes excluded
    demographic_cols = {"Race", "Gender", "State", "County"}
    found_demos = set(train_feat.columns) & demographic_cols
    chk("no_protected_demographic_attributes", len(found_demos) == 0, f"found: {found_demos}")

    # Labels present
    chk("fraud_label_in_train", "PotentialFraud" in train_feat.columns)
    chk("fraud_label_in_val",   "PotentialFraud" in val_feat.columns)
    chk("fraud_label_in_test",  "PotentialFraud" in test_feat.columns)

    # Sequence & mask checks
    for split in ["train", "val", "test"]:
        if split in seq_data:
            s_arr = seq_data[split]["sequences"]
            m_arr = seq_data[split]["masks"]
            p_ids = seq_data[split]["provider_ids"]
            feat_df = {"train": train_feat, "val": val_feat, "test": test_feat}[split]

            # Shape match
            match_shape = (s_arr.shape[0] == m_arr.shape[0] == len(p_ids) == len(feat_df))
            chk(f"{split}_seq_mask_provider_count_match", match_shape,
                f"seq:{s_arr.shape[0]}, mask:{m_arr.shape[0]}, pids:{len(p_ids)}, feat:{len(feat_df)}")

            # Mask values check (only 0.0 and 1.0)
            valid_mask_vals = np.isin(m_arr, [0.0, 1.0]).all()
            chk(f"{split}_mask_binary_values", bool(valid_mask_vals),
                f"min:{m_arr.min()}, max:{m_arr.max()}")

            # Provider ID exact match
            sorted_feat_pids = sorted(feat_df["ProviderID"].tolist())
            chk(f"{split}_sequence_pids_match_feature_pids", p_ids == sorted_feat_pids)

    return checks


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_feature_engineering():
    log.info("=" * 70)
    log.info("HEALTHGUARD AI — Phase 1: Feature Engineering (Updated)")
    log.info("=" * 70)

    claims_path = PROC / "cleaned_training_claims.csv"
    if not claims_path.exists():
        raise FileNotFoundError(f"Run preprocessing.py first. Missing: {claims_path}")

    log.info("\n[1] Loading cleaned claims…")
    claims = pd.read_csv(claims_path, low_memory=False)
    for col in ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt", "DOD", "DOB"]:
        if col in claims.columns:
            claims[col] = pd.to_datetime(claims[col], errors="coerce")

    if "_is_inpatient" not in claims.columns:
        claims["_is_inpatient"] = (claims["ClaimType"] == "Inpatient").astype(int)

    train_ids_df = pd.read_csv(PROC / "train_provider_ids.csv")
    val_ids_df   = pd.read_csv(PROC / "validation_provider_ids.csv")
    test_ids_df  = pd.read_csv(PROC / "test_provider_ids.csv")

    train_ids = train_ids_df["ProviderID"]
    val_ids   = val_ids_df["ProviderID"]
    test_ids  = test_ids_df["ProviderID"]

    log.info(f"  Claims: {len(claims):,} rows")
    log.info(f"  Providers — train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")

    # ── Step 2: Build Training Features & Fit Peer Reference Table ───────────
    log.info("\n[2] Building training provider features…")
    train_claims = claims[claims["ProviderID"].isin(train_ids)]
    train_feat = build_provider_features(train_claims, peer_ref=None)

    log.info("\n[3] Fitting peer percentile reference table on training providers only…")
    peer_ref = compute_peer_reference_table(train_feat)
    with open(PROC / "peer_reference_table.json", "w") as f:
        json.dump(peer_ref, f, indent=2)
    with open(REP / "evaluation" / "peer_reference_table.json", "w") as f:
        json.dump(peer_ref, f, indent=2)
    log.info("  Saved peer_reference_table.json")

    # Update train feature reimbursement_to_peer_median
    train_feat["reimbursement_to_peer_median"] = (
        train_feat["total_reimbursement"] / (peer_ref["median_total_reimbursement"] + 1e-6)
    )

    # ── Step 3: Build Validation & Test Features Using Training Peer Stats ───
    log.info("\n[4] Building validation and test features using training peer reference…")
    val_claims  = claims[claims["ProviderID"].isin(val_ids)]
    test_claims = claims[claims["ProviderID"].isin(test_ids)]

    val_feat  = build_provider_features(val_claims, peer_ref=peer_ref)
    test_feat = build_provider_features(test_claims, peer_ref=peer_ref)

    # Attach fraud labels
    ids_map = {"train": (train_feat, train_ids_df),
               "validation": (val_feat, val_ids_df),
               "test": (test_feat, test_ids_df)}
    split_features = {}
    for split_name, (feat, id_df) in ids_map.items():
        feat = feat.merge(id_df[["ProviderID", "PotentialFraud"]], on="ProviderID", how="left")
        split_features[split_name] = feat
        out_path = PROC / f"{split_name}_provider_features.csv"
        feat.to_csv(out_path, index=False)
        log.info(f"  Saved {out_path.name}: {feat.shape}")

    # ── Step 4: Save Feature Spec JSON ────────────────────────────────────────
    log.info("\n[5] Saving feature specification JSON…")
    spec = build_feature_spec(split_features["train"])
    with open(PROC / "feature_spec.json", "w") as f:
        json.dump(spec, f, indent=2)
    log.info(f"  Saved feature_spec.json: {len(spec)} features")

    # ── Step 5: Build Monthly LSTM Sequences & Masks ─────────────────────────
    log.info("\n[6] Building monthly LSTM sequences and masks…")
    seq_len = 12
    seq_data = {}
    all_metadata = None

    split_triples = [
        ("train", train_claims, train_ids),
        ("validation", val_claims, val_ids),
        ("test", test_claims, test_ids),
    ]

    for split_name, s_claims, s_ids in split_triples:
        seqs, masks, ordered_pids, meta = build_monthly_sequences(
            s_claims, s_ids, seq_len=seq_len
        )
        all_metadata = meta
        key = "val" if split_name == "validation" else split_name
        seq_data[key] = {
            "sequences": seqs,
            "masks": masks,
            "provider_ids": ordered_pids,
        }

        # Save numpy arrays
        np.save(PROC / f"{split_name}_provider_sequences.npy", seqs)
        np.save(PROC / f"{split_name}_provider_masks.npy", masks)
        log.info(f"  Saved {split_name}_provider_sequences.npy: {seqs.shape}")
        log.info(f"  Saved {split_name}_provider_masks.npy: {masks.shape}")

        # Save provider ID ordering
        pd.DataFrame({"ProviderID": ordered_pids}).to_csv(
            PROC / f"{split_name}_sequence_provider_ids.csv", index=False
        )

    if all_metadata:
        with open(PROC / "sequence_metadata.json", "w") as f:
            json.dump(all_metadata, f, indent=2)
        log.info("  Saved sequence_metadata.json")

    # ── Step 6: Automated Checks ──────────────────────────────────────────────
    log.info("\n[7] Running automated feature & sequence checks…")
    checks = run_feature_checks(
        split_features["train"],
        split_features["validation"],
        split_features["test"],
        train_ids.tolist(),
        val_ids.tolist(),
        test_ids.tolist(),
        seq_data,
    )
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(REP / "evaluation" / "feature_checks.csv", index=False)
    n_pass = (checks_df["status"] == "PASS").sum()
    n_fail = (checks_df["status"] == "FAIL").sum()
    log.info(f"  Checks: {n_pass} passed, {n_fail} failed")

    log.info("\n[✓] Feature engineering complete.")
    return split_features, checks


if __name__ == "__main__":
    run_feature_engineering()
    print("\nDone. Feature and sequence files saved in data/processed/")
