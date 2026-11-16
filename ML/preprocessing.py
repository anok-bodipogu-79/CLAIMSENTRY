"""
HealthGuard AI - Phase 1: Data Loading, Cleaning and Preprocessing
===================================================================
Loads the 8 Kaggle CSV files, integrates them, cleans data quality
issues (flagging anomalies rather than dropping records), calculates
beneficiary age per claim at claim date, splits providers (stratified 70/15/15),
and saves processed outputs to data/processed/.

Usage:
    python ML/preprocessing.py
"""

import os
import sys
import json
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
REP  = ROOT / "reports"
PROC.mkdir(parents=True, exist_ok=True)
(REP / "charts").mkdir(parents=True, exist_ok=True)
(REP / "evaluation").mkdir(parents=True, exist_ok=True)
(REP / "screenshots").mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("preprocessing")

RANDOM_SEED = 42

# ── File Manifest ─────────────────────────────────────────────────────────────
TRAIN_LABELS   = "Train-1542865627584.csv"
TRAIN_BENE     = "Train_Beneficiarydata-1542865627584.csv"
TRAIN_INPAT    = "Train_Inpatientdata-1542865627584.csv"
TRAIN_OUTPAT   = "Train_Outpatientdata-1542865627584.csv"
TEST_LABELS    = "Test-1542969243754.csv"
TEST_BENE      = "Test_Beneficiarydata-1542969243754.csv"
TEST_INPAT     = "Test_Inpatientdata-1542969243754.csv"
TEST_OUTPAT    = "Test_Outpatientdata-1542969243754.csv"


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_csv(filename: str) -> pd.DataFrame:
    """Load a CSV from data/raw with error handling."""
    path = RAW / filename
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"  Loaded {filename}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df


def inspect_file(filename: str) -> dict:
    """Return record count, columns, dtypes and missing-value counts."""
    df = load_csv(filename)
    missing = df.isnull().sum()
    for col in df.select_dtypes(include="object"):
        na_str = (df[col].astype(str).str.strip().str.upper() == "NA").sum()
        if na_str > 0:
            missing[col] = missing.get(col, 0) + na_str
    return {
        "file": filename,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(v) for c, v in missing.items() if v > 0},
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATE CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

DATE_COLS_BENE  = ["DOB", "DOD"]
DATE_COLS_CLAIM = ["ClaimStartDt", "ClaimEndDt"]
DATE_COLS_INPAT = ["AdmissionDt", "DischargeDt"]


def to_datetime(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Convert listed columns to datetime; leave NaT where conversion fails."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. CHRONIC CONDITION ENCODING
# ══════════════════════════════════════════════════════════════════════════════

CHRONIC_COLS = [
    "ChronicCond_Alzheimer", "ChronicCond_Heartfailure",
    "ChronicCond_KidneyDisease", "ChronicCond_Cancer",
    "ChronicCond_ObstrPulmonary", "ChronicCond_Depression",
    "ChronicCond_Diabetes", "ChronicCond_IschemicHeart",
    "ChronicCond_Osteoporasis", "ChronicCond_rheumatoidarthritis",
    "ChronicCond_stroke",
]


def encode_chronic_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kaggle encoding: 1 = condition present, 2 = condition absent.
    Convert to binary: 1 → 1 (present), 2 → 0 (absent).
    """
    for col in CHRONIC_COLS:
        if col in df.columns:
            raw = df[col]
            df[col] = raw.map({1: 1, 2: 0})
    return df


def encode_renal(df: pd.DataFrame) -> pd.DataFrame:
    """RenalDiseaseIndicator: 'Y'→1, '0'→0, '1'→1, else NaN."""
    if "RenalDiseaseIndicator" in df.columns:
        df["RenalDiseaseIndicator"] = (
            df["RenalDiseaseIndicator"]
            .astype(str).str.strip().str.upper()
            .map({"Y": 1, "0": 0, "1": 1})
        )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. LOAD AND CLEAN BENEFICIARIES
# ══════════════════════════════════════════════════════════════════════════════

def load_clean_beneficiaries(filename: str) -> pd.DataFrame:
    """
    Load beneficiary data, parse dates, encode chronic conditions,
    and compute deceased flag. Fixed reference-year age is NOT computed
    here because age is computed per claim based on claim date.
    """
    df = load_csv(filename)
    df = to_datetime(df, DATE_COLS_BENE)

    # Deceased indicator
    df["IsDeceased"] = df["DOD"].notna().astype(int)

    # Chronic condition binary encoding
    df = encode_chronic_conditions(df)

    # Count chronic conditions per beneficiary
    present_cols = [c for c in CHRONIC_COLS if c in df.columns]
    df["ChronicCondCount"] = df[present_cols].sum(axis=1)

    # Renal disease
    df = encode_renal(df)

    # Deduplication
    before = len(df)
    df = df.drop_duplicates(subset=["BeneID"])
    after = len(df)
    if before > after:
        log.warning(f"  Beneficiaries: dropped {before - after} duplicate BeneID rows")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 5. LOAD AND CLEAN CLAIMS
# ══════════════════════════════════════════════════════════════════════════════

def _standardize_provider(df: pd.DataFrame) -> pd.DataFrame:
    """Rename 'Provider' → 'ProviderID' if needed."""
    if "Provider" in df.columns and "ProviderID" not in df.columns:
        df = df.rename(columns={"Provider": "ProviderID"})
    return df


def load_clean_inpatient(filename: str) -> pd.DataFrame:
    df = load_csv(filename)
    df = _standardize_provider(df)
    df["ClaimType"] = "Inpatient"
    df = to_datetime(df, DATE_COLS_CLAIM + DATE_COLS_INPAT)
    return df


def load_clean_outpatient(filename: str) -> pd.DataFrame:
    df = load_csv(filename)
    df = _standardize_provider(df)
    df["ClaimType"] = "Outpatient"
    df = to_datetime(df, DATE_COLS_CLAIM)
    # Outpatient has no AdmissionDt / DischargeDt / DiagnosisGroupCode
    for col in ["AdmissionDt", "DischargeDt", "DiagnosisGroupCode"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def combine_claims(inpat: pd.DataFrame, outpat: pd.DataFrame) -> pd.DataFrame:
    """Stack inpatient and outpatient into one dataframe."""
    combined = pd.concat([inpat, outpat], ignore_index=True, sort=False)
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# 6. DATA QUALITY CHECKS AND CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def clean_claims(claims: pd.DataFrame, bene: pd.DataFrame,
                 provider_labels: pd.DataFrame) -> tuple:
    """
    Apply quality checks and cleaning rules.
    Date anomalies are FLAGGED rather than deleted.
    Returns (cleaned_df, action_log).
    """
    actions = []
    df = claims.copy()
    original_count = len(df)
    log.info(f"  Starting claims cleaning: {original_count:,} rows")

    # ── 6.1 Duplicate ClaimIDs ────────────────────────────────────────────────
    dup_mask = df.duplicated(subset=["ClaimID"], keep=False)
    n_dup_claims = int(dup_mask.sum())
    if n_dup_claims > 0:
        conflict_mask = df.duplicated(subset=["ClaimID"], keep="first") & dup_mask
        n_conflict = int(conflict_mask.sum())
        df = df.drop_duplicates(subset=["ClaimID"], keep="first")
        actions.append({
            "action": "remove_duplicate_claimids",
            "total_duplicate_rows": n_dup_claims,
            "conflicting_rows_removed": n_conflict,
            "rows_after": len(df),
        })
        log.info(f"    Duplicate ClaimIDs: {n_dup_claims:,} rows ({n_conflict} conflicting) → kept first occurrence")
    else:
        actions.append({
            "action": "check_duplicate_claimids",
            "total_duplicate_rows": 0,
            "action_taken": "none_found",
        })

    # ── 6.2 Date Anomaly Handling (Flagging, not deleting) ────────────────────
    # Claim end before claim start
    df["ClaimDateAnomaly"] = 0
    if "ClaimStartDt" in df.columns and "ClaimEndDt" in df.columns:
        bad_claim_dates = (
            df["ClaimStartDt"].notna() &
            df["ClaimEndDt"].notna() &
            (df["ClaimEndDt"] < df["ClaimStartDt"])
        )
        df["ClaimDateAnomaly"] = bad_claim_dates.astype(int)
        n_bad = int(bad_claim_dates.sum())
        actions.append({
            "action": "flag_claim_end_before_start",
            "count_flagged": n_bad,
            "action_taken": "kept_with_flag",
        })
        if n_bad > 0:
            log.warning(f"    Flagged {n_bad} claims with ClaimEndDt < ClaimStartDt (records kept)")
        else:
            log.info("    ClaimEndDt < ClaimStartDt check: 0 anomalies found")

    # Inpatient discharge before admission
    df["DischargeBeforeAdmission"] = 0
    ip_mask = (df["ClaimType"] == "Inpatient")
    if "AdmissionDt" in df.columns and "DischargeDt" in df.columns:
        bad_ip = (
            ip_mask &
            df["AdmissionDt"].notna() &
            df["DischargeDt"].notna() &
            (df["DischargeDt"] < df["AdmissionDt"])
        )
        df["DischargeBeforeAdmission"] = bad_ip.astype(int)
        n_bad_ip = int(bad_ip.sum())
        actions.append({
            "action": "flag_discharge_before_admission",
            "count_flagged": n_bad_ip,
            "action_taken": "kept_with_flag",
        })
        if n_bad_ip > 0:
            log.warning(f"    Flagged {n_bad_ip} inpatient claims with DischargeDt < AdmissionDt (records kept)")
        else:
            log.info("    DischargeDt < AdmissionDt check: 0 anomalies found")

    # Inpatient Length of Stay (days) per claim
    df["InpatientLOS"] = np.nan
    if "AdmissionDt" in df.columns and "DischargeDt" in df.columns:
        valid_ip_stay = (
            ip_mask &
            df["AdmissionDt"].notna() &
            df["DischargeDt"].notna() &
            (df["DischargeDt"] >= df["AdmissionDt"])
        )
        df.loc[valid_ip_stay, "InpatientLOS"] = (
            df.loc[valid_ip_stay, "DischargeDt"] - df.loc[valid_ip_stay, "AdmissionDt"]
        ).dt.days

    # ── 6.3 Negative / zero reimbursement ────────────────────────────────────
    if "InscClaimAmtReimbursed" in df.columns:
        neg_mask = df["InscClaimAmtReimbursed"] < 0
        n_neg = int(neg_mask.sum())
        zero_mask = df["InscClaimAmtReimbursed"] == 0
        n_zero = int(zero_mask.sum())
        if n_neg > 0:
            df = df[~neg_mask]
            actions.append({
                "action": "remove_negative_reimbursement",
                "rows_removed": n_neg,
                "rows_after": len(df),
            })
            log.warning(f"    Removed {n_neg} rows with negative reimbursement")
        actions.append({
            "action": "note_zero_reimbursement",
            "count": n_zero,
            "action_taken": "kept",
        })
        log.info(f"    Zero reimbursement rows: {n_zero} (kept — valid copay/deductible scenarios)")

    # ── 6.4 Missing-value indicators ─────────────────────────────────────────
    for phys_col in ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]:
        if phys_col in df.columns:
            df[phys_col] = df[phys_col].replace("nan", np.nan)
    df["AttendingPhysician_Missing"] = df["AttendingPhysician"].isna().astype(int) \
        if "AttendingPhysician" in df.columns else 0
    df["AnyPhysician_Missing"] = (
        df[["AttendingPhysician", "OperatingPhysician", "OtherPhysician"]]
        .isnull().all(axis=1).astype(int)
        if all(c in df.columns for c in ["AttendingPhysician", "OperatingPhysician", "OtherPhysician"])
        else 0
    )

    # Diagnosis codes count & indicator
    diag_cols = [c for c in df.columns if c.startswith("ClmDiagnosisCode_")]
    if diag_cols:
        df["DiagnosisCode_Count"] = df[diag_cols].notna().sum(axis=1)
        df["DiagnosisCode_Missing"] = (df["DiagnosisCode_Count"] == 0).astype(int)

    # Procedure codes count & indicator
    proc_cols = [c for c in df.columns if c.startswith("ClmProcedureCode_")]
    if proc_cols:
        df["ProcedureCode_Count"] = df[proc_cols].notna().sum(axis=1)
        df["ProcedureCode_Missing"] = (df["ProcedureCode_Count"] == 0).astype(int)

    actions.append({
        "action": "add_missing_value_indicators",
        "indicators": [
            "AttendingPhysician_Missing", "AnyPhysician_Missing",
            "DiagnosisCode_Count", "DiagnosisCode_Missing",
            "ProcedureCode_Count", "ProcedureCode_Missing",
        ],
    })

    # ── 6.5 Join beneficiary information ─────────────────────────────────────
    bene_cols = [
        "BeneID", "DOB", "DOD", "IsDeceased", "ChronicCondCount",
        "RenalDiseaseIndicator", "NoOfMonths_PartACov", "NoOfMonths_PartBCov",
        "IPAnnualReimbursementAmt", "IPAnnualDeductibleAmt",
        "OPAnnualReimbursementAmt", "OPAnnualDeductibleAmt",
    ] + CHRONIC_COLS
    bene_cols = [c for c in bene_cols if c in bene.columns]
    before_join = len(df)
    df = df.merge(bene[bene_cols], on="BeneID", how="left")

    # ── 6.6 Calculate beneficiary age per claim at actual claim date ─────────
    df["BeneAgeAtClaim"] = np.nan
    if "ClaimStartDt" in df.columns and "DOB" in df.columns:
        valid_dob = (
            df["ClaimStartDt"].notna() &
            df["DOB"].notna() &
            (df["ClaimStartDt"] >= df["DOB"])
        )
        age_in_days = (df.loc[valid_dob, "ClaimStartDt"] - df.loc[valid_dob, "DOB"]).dt.days
        df.loc[valid_dob, "BeneAgeAtClaim"] = age_in_days / 365.25

    missing_bene = int(df["DOB"].isna().sum())
    if missing_bene > 0:
        log.warning(f"    {missing_bene} claims missing beneficiary info after join")
    actions.append({
        "action": "join_beneficiary_info_and_compute_claim_age",
        "claims_before": int(before_join),
        "claims_after": int(len(df)),
        "missing_bene_info": missing_bene,
        "mean_claim_age": float(df["BeneAgeAtClaim"].mean()) if df["BeneAgeAtClaim"].notna().any() else 0.0,
    })

    # ── 6.7 Claims after recorded date of death ───────────────────────────────
    if "DOD" in df.columns and "ClaimStartDt" in df.columns:
        after_death = df["IsDeceased"].eq(1) & (df["ClaimStartDt"] > df["DOD"])
        df["ClaimAfterDeath"] = after_death.astype(int)
        n_after_death = int(after_death.sum())
        actions.append({
            "action": "flag_claims_after_death",
            "count_flagged": n_after_death,
            "action_taken": "kept_with_flag",
        })
        log.info(f"    Claims after beneficiary death: {n_after_death} (flagged, kept)")

    # ── 6.8 Connect provider labels ───────────────────────────────────────────
    before_label = len(df)
    df = df.merge(provider_labels[["ProviderID", "PotentialFraud"]],
                  on="ProviderID", how="left")
    unlabeled = int(df["PotentialFraud"].isna().sum())
    if unlabeled > 0:
        log.warning(f"    {unlabeled} claims from providers without fraud labels — dropping")
        df = df[df["PotentialFraud"].notna()]
        actions.append({
            "action": "remove_unlabeled_provider_claims",
            "rows_removed": unlabeled,
            "rows_after": len(df),
        })

    # ── 6.9 Providers without claims ─────────────────────────────────────────
    provider_in_claims = set(df["ProviderID"].unique())
    provider_in_labels = set(provider_labels["ProviderID"].unique())
    no_claims = provider_in_labels - provider_in_claims
    actions.append({
        "action": "note_providers_without_claims",
        "count": len(no_claims),
        "provider_ids": sorted(list(no_claims))[:20],
    })
    if no_claims:
        log.warning(f"    {len(no_claims)} providers in label file have no claims")

    # ── 6.10 Final summary ───────────────────────────────────────────────────
    total_removed = original_count - len(df)
    actions.append({
        "action": "total_rows_removed",
        "original": int(original_count),
        "final": int(len(df)),
        "removed": int(total_removed),
    })
    log.info(f"  Cleaning complete: {original_count:,} → {len(df):,} rows ({total_removed:,} removed)")

    return df, actions


# ══════════════════════════════════════════════════════════════════════════════
# 7. PROVIDER LABELS
# ══════════════════════════════════════════════════════════════════════════════

def load_provider_labels(filename: str) -> pd.DataFrame:
    df = load_csv(filename)
    df = _standardize_provider(df)
    if "PotentialFraud" in df.columns:
        df["PotentialFraud"] = df["PotentialFraud"].map({"Yes": 1, "No": 0})
        log.info(f"  Fraud label distribution:\n{df['PotentialFraud'].value_counts().to_string()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 8. PROVIDER SPLIT (STRATIFIED 70 / 15 / 15)
# ══════════════════════════════════════════════════════════════════════════════

def split_providers(provider_labels: pd.DataFrame) -> tuple:
    """
    Stratified split at the PROVIDER level.
    Returns (train_ids, val_ids, test_ids) as Series.
    No claim from one provider will appear in multiple splits.
    """
    providers = provider_labels[["ProviderID", "PotentialFraud"]].drop_duplicates()
    X = providers["ProviderID"]
    y = providers["PotentialFraud"]

    # 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED
    )
    # 50% of 30% = 15% val, 15% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_SEED
    )

    log.info(f"  Provider split — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # Validate no overlap
    sets = [set(X_train), set(X_val), set(X_test)]
    assert sets[0].isdisjoint(sets[1]), "OVERLAP: train ∩ val"
    assert sets[0].isdisjoint(sets[2]), "OVERLAP: train ∩ test"
    assert sets[1].isdisjoint(sets[2]), "OVERLAP: val ∩ test"
    log.info("  ✓ Provider split overlap check passed")

    for name, ids, labels in [("Train", X_train, y_train),
                               ("Val",   X_val,   y_val),
                               ("Test",  X_test,  y_test)]:
        fraud_pct = labels.mean() * 100
        log.info(f"    {name}: {len(ids)} providers, {fraud_pct:.1f}% fraud")

    return X_train, X_val, X_test


def save_provider_splits(X_train, X_val, X_test,
                         provider_labels: pd.DataFrame) -> None:
    for name, ids in [("train", X_train), ("validation", X_val), ("test", X_test)]:
        df_out = provider_labels[provider_labels["ProviderID"].isin(ids)][
            ["ProviderID", "PotentialFraud"]
        ].copy()
        path = PROC / f"{name}_provider_ids.csv"
        df_out.to_csv(path, index=False)
        log.info(f"  Saved {path.name}: {len(df_out)} providers")


# ══════════════════════════════════════════════════════════════════════════════
# 9. PREPARE KAGGLE TEST DATA (no labels)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_kaggle_test(test_labels_file, test_bene_file,
                        test_inpat_file, test_outpat_file) -> pd.DataFrame:
    """
    Load and integrate Kaggle test data (no PotentialFraud label).
    Applies identical anomaly flags, LOS, and per-claim age logic.
    """
    log.info("  Preparing Kaggle test data (no labels)…")
    test_providers = load_csv(test_labels_file)
    test_providers = _standardize_provider(test_providers)

    test_bene = load_clean_beneficiaries(test_bene_file)
    test_inpat = load_clean_inpatient(test_inpat_file)
    test_outpat = load_clean_outpatient(test_outpat_file)
    test_claims = combine_claims(test_inpat, test_outpat)
    test_claims = _standardize_provider(test_claims)

    # Date anomaly flags
    test_claims["ClaimDateAnomaly"] = 0
    if "ClaimStartDt" in test_claims.columns and "ClaimEndDt" in test_claims.columns:
        bad_dates = (
            test_claims["ClaimStartDt"].notna() &
            test_claims["ClaimEndDt"].notna() &
            (test_claims["ClaimEndDt"] < test_claims["ClaimStartDt"])
        )
        test_claims["ClaimDateAnomaly"] = bad_dates.astype(int)

    test_claims["DischargeBeforeAdmission"] = 0
    ip_mask = (test_claims["ClaimType"] == "Inpatient")
    if "AdmissionDt" in test_claims.columns and "DischargeDt" in test_claims.columns:
        bad_ip = (
            ip_mask &
            test_claims["AdmissionDt"].notna() &
            test_claims["DischargeDt"].notna() &
            (test_claims["DischargeDt"] < test_claims["AdmissionDt"])
        )
        test_claims["DischargeBeforeAdmission"] = bad_ip.astype(int)

    test_claims["InpatientLOS"] = np.nan
    if "AdmissionDt" in test_claims.columns and "DischargeDt" in test_claims.columns:
        valid_ip = (
            ip_mask &
            test_claims["AdmissionDt"].notna() &
            test_claims["DischargeDt"].notna() &
            (test_claims["DischargeDt"] >= test_claims["AdmissionDt"])
        )
        test_claims.loc[valid_ip, "InpatientLOS"] = (
            test_claims.loc[valid_ip, "DischargeDt"] - test_claims.loc[valid_ip, "AdmissionDt"]
        ).dt.days

    # Merge beneficiary info (exclude direct demographic columns from feature set)
    bene_cols = [c for c in test_bene.columns if c not in ["Gender", "Race", "State"]]
    test_claims = test_claims.merge(
        test_bene[[c for c in bene_cols if c in test_bene.columns]],
        on="BeneID", how="left"
    )

    # Calculate beneficiary age per claim
    test_claims["BeneAgeAtClaim"] = np.nan
    if "ClaimStartDt" in test_claims.columns and "DOB" in test_claims.columns:
        valid_dob = (
            test_claims["ClaimStartDt"].notna() &
            test_claims["DOB"].notna() &
            (test_claims["ClaimStartDt"] >= test_claims["DOB"])
        )
        age_days = (test_claims.loc[valid_dob, "ClaimStartDt"] - test_claims.loc[valid_dob, "DOB"]).dt.days
        test_claims.loc[valid_dob, "BeneAgeAtClaim"] = age_days / 365.25

    # Mark claims after death
    if "DOD" in test_claims.columns and "ClaimStartDt" in test_claims.columns:
        after_death = test_claims["IsDeceased"].eq(1) & \
                      (test_claims["ClaimStartDt"] > test_claims["DOD"])
        test_claims["ClaimAfterDeath"] = after_death.astype(int)

    test_claims.to_csv(PROC / "kaggle_test_claims.csv", index=False)
    test_bene.to_csv(PROC / "kaggle_test_beneficiaries.csv", index=False)
    test_providers.to_csv(PROC / "kaggle_test_providers.csv", index=False)
    log.info(f"  Kaggle test claims saved: {len(test_claims):,} rows")
    return test_claims


# ══════════════════════════════════════════════════════════════════════════════
# 10. REPORTS: RECORD COUNTS, MISSING VALUES, DUPLICATES
# ══════════════════════════════════════════════════════════════════════════════

def save_dataset_summary(inspect_results: list) -> None:
    rows = []
    for r in inspect_results:
        rows.append({
            "file": r["file"],
            "rows": r["rows"],
            "cols": r["cols"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(REP / "evaluation" / "dataset_record_counts.csv", index=False)
    log.info("  Saved dataset_record_counts.csv")


def save_missing_value_summary(inspect_results: list) -> None:
    rows = []
    for r in inspect_results:
        for col, cnt in r.get("missing", {}).items():
            rows.append({
                "file": r["file"],
                "column": col,
                "missing_count": cnt,
                "missing_pct": round(cnt / r["rows"] * 100, 2),
            })
    df = pd.DataFrame(rows)
    df.to_csv(REP / "evaluation" / "missing_value_summary.csv", index=False)
    log.info("  Saved missing_value_summary.csv")


def save_duplicate_summary(claims: pd.DataFrame, action_log: list) -> None:
    dup_actions = [a for a in action_log if "duplicate" in a.get("action", "").lower()]
    df = pd.DataFrame(dup_actions) if dup_actions else pd.DataFrame(
        [{"action": "no_duplicates_found", "rows_removed": 0}]
    )
    df.to_csv(REP / "evaluation" / "duplicate_record_summary.csv", index=False)
    log.info("  Saved duplicate_record_summary.csv")


def save_cleaning_log(action_log: list) -> None:
    with open(REP / "evaluation" / "cleaning_action_log.json", "w") as f:
        json.dump(action_log, f, indent=2, default=str)
    log.info("  Saved cleaning_action_log.json")


# ══════════════════════════════════════════════════════════════════════════════
# 11. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_preprocessing() -> dict:
    log.info("=" * 70)
    log.info("HEALTHGUARD AI — Phase 1: Preprocessing Pipeline (Updated)")
    log.info("=" * 70)

    # ── Step 1: Inspect all 8 source files ────────────────────────────────────
    log.info("\n[1] Inspecting source files…")
    all_files = [
        TRAIN_LABELS, TRAIN_BENE, TRAIN_INPAT, TRAIN_OUTPAT,
        TEST_LABELS, TEST_BENE, TEST_INPAT, TEST_OUTPAT,
    ]
    inspect_results = [inspect_file(f) for f in all_files]
    save_dataset_summary(inspect_results)
    save_missing_value_summary(inspect_results)

    # ── Step 2: Load provider labels ─────────────────────────────────────────
    log.info("\n[2] Loading provider fraud labels…")
    provider_labels = load_provider_labels(TRAIN_LABELS)
    provider_labels.to_csv(PROC / "provider_labels.csv", index=False)

    # ── Step 3: Load & clean beneficiaries ───────────────────────────────────
    log.info("\n[3] Loading and cleaning beneficiaries…")
    bene = load_clean_beneficiaries(TRAIN_BENE)

    # ── Step 4: Load, combine and clean claims ────────────────────────────────
    log.info("\n[4] Loading inpatient and outpatient claims…")
    inpat  = load_clean_inpatient(TRAIN_INPAT)
    outpat = load_clean_outpatient(TRAIN_OUTPAT)
    log.info(f"  Inpatient rows: {len(inpat):,}")
    log.info(f"  Outpatient rows: {len(outpat):,}")
    claims = combine_claims(inpat, outpat)
    log.info(f"  Combined claims: {len(claims):,}")

    # ── Step 5: Clean combined claims ─────────────────────────────────────────
    log.info("\n[5] Cleaning combined claims…")
    cleaned_claims, action_log = clean_claims(claims, bene, provider_labels)
    save_duplicate_summary(cleaned_claims, action_log)
    save_cleaning_log(action_log)

    # ── Step 6: Save cleaned datasets ─────────────────────────────────────────
    log.info("\n[6] Saving cleaned datasets…")
    cleaned_claims.to_csv(PROC / "cleaned_training_claims.csv", index=False)
    bene.to_csv(PROC / "cleaned_training_beneficiaries.csv", index=False)
    log.info(f"  Saved cleaned_training_claims.csv: {len(cleaned_claims):,} rows")
    log.info(f"  Saved cleaned_training_beneficiaries.csv: {len(bene):,} rows")

    # ── Step 7: Provider-level split ─────────────────────────────────────────
    log.info("\n[7] Splitting providers (stratified 70/15/15)…")
    X_train, X_val, X_test = split_providers(provider_labels)
    save_provider_splits(X_train, X_val, X_test, provider_labels)

    # ── Step 8: Kaggle test data preparation ──────────────────────────────────
    log.info("\n[8] Preparing Kaggle test data (unlabeled)…")
    prepare_kaggle_test(TEST_LABELS, TEST_BENE, TEST_INPAT, TEST_OUTPAT)

    log.info("\n[✓] Preprocessing pipeline complete.")
    return {
        "cleaned_claims": cleaned_claims,
        "bene": bene,
        "provider_labels": provider_labels,
        "train_ids": X_train,
        "val_ids": X_val,
        "test_ids": X_test,
        "action_log": action_log,
        "inspect_results": inspect_results,
    }


if __name__ == "__main__":
    result = run_preprocessing()
    print("\nDone. Output files in data/processed/ and reports/evaluation/")
