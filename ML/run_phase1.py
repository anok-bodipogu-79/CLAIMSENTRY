"""
HealthGuard AI - Phase 1: Complete Pipeline Runner
===================================================
Runs the full Phase 1 pipeline in order:
  1. Preprocessing (data loading, cleaning, anomaly flagging, per-claim age, provider split)
  2. Feature engineering (tabular features, training peer reference table, LSTM sequences & masks)
  3. Report & chart generation
  4. Final verification

Usage:
    python ML/run_phase1.py
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ML.preprocessing import run_preprocessing
from ML.feature_engineering import run_feature_engineering
from ML.generate_reports import run_reports

PROC = ROOT / "data" / "processed"
REP  = ROOT / "reports"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("run_phase1")

RANDOM_SEED = 42

EXPECTED_PROCESSED_FILES = [
    "cleaned_training_claims.csv",
    "cleaned_training_beneficiaries.csv",
    "provider_labels.csv",
    "train_provider_features.csv",
    "validation_provider_features.csv",
    "test_provider_features.csv",
    "train_provider_sequences.npy",
    "validation_provider_sequences.npy",
    "test_provider_sequences.npy",
    "train_provider_masks.npy",
    "validation_provider_masks.npy",
    "test_provider_masks.npy",
    "train_provider_ids.csv",
    "validation_provider_ids.csv",
    "test_provider_ids.csv",
    "train_sequence_provider_ids.csv",
    "validation_sequence_provider_ids.csv",
    "test_sequence_provider_ids.csv",
    "peer_reference_table.json",
    "feature_spec.json",
    "sequence_metadata.json",
]

EXPECTED_REPORT_FILES = [
    "evaluation/dataset_record_counts.csv",
    "evaluation/missing_value_summary.csv",
    "evaluation/duplicate_record_summary.csv",
    "evaluation/cleaning_action_log.json",
    "evaluation/feature_checks.csv",
    "evaluation/provider_feature_summary.csv",
    "evaluation/peer_reference_table.json",
    "charts/fraud_label_distribution.png",
    "charts/claim_type_distribution.png",
    "charts/avg_reimbursement_by_claim_type.png",
    "charts/reimbursement_distribution.png",
    "charts/reimbursement_by_fraud_label.png",
    "charts/provider_feature_distributions.png",
    "charts/feature_means_fraud_vs_notfraud.png",
    "charts/split_fraud_distribution.png",
]


def verify_outputs() -> list:
    """Check that all expected output files exist and have non-zero size."""
    results = []

    for f in EXPECTED_PROCESSED_FILES:
        path = PROC / f
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        ok = exists and size > 0
        results.append({
            "file": str(path.relative_to(ROOT)),
            "exists": exists,
            "size_bytes": size,
            "status": "OK" if ok else "MISSING/EMPTY",
        })

    for f in EXPECTED_REPORT_FILES:
        path = REP / f
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        ok = exists and size > 0
        results.append({
            "file": str(path.relative_to(ROOT)),
            "exists": exists,
            "size_bytes": size,
            "status": "OK" if ok else "MISSING/EMPTY",
        })

    return results


def verify_no_provider_overlap() -> dict:
    """Verify provider splits do not overlap."""
    train_df = pd.read_csv(PROC / "train_provider_ids.csv")
    val_df   = pd.read_csv(PROC / "validation_provider_ids.csv")
    test_df  = pd.read_csv(PROC / "test_provider_ids.csv")

    t = set(train_df["ProviderID"])
    v = set(val_df["ProviderID"])
    s = set(test_df["ProviderID"])

    return {
        "train_count":    len(t),
        "val_count":      len(v),
        "test_count":     len(s),
        "train_val_overlap":  len(t & v),
        "train_test_overlap": len(t & s),
        "val_test_overlap":   len(v & s),
        "overlap_free":       len(t & v) == 0 and len(t & s) == 0 and len(v & s) == 0,
    }


def verify_fraud_distribution() -> dict:
    """Check stratification preserved fraud ratio across splits."""
    results = {}
    for split in ["train", "validation", "test"]:
        df = pd.read_csv(PROC / f"{split}_provider_ids.csv")
        if "PotentialFraud" in df.columns:
            fraud_pct = df["PotentialFraud"].mean() * 100
            results[split] = {
                "total": len(df),
                "fraud": int(df["PotentialFraud"].sum()),
                "fraud_pct": round(fraud_pct, 2),
            }
    return results


def verify_sequences_and_masks() -> dict:
    """Check LSTM sequence and mask shapes and properties."""
    results = {}
    seq_shapes = {}
    mask_shapes = {}

    for split in ["train", "validation", "test"]:
        s_path = PROC / f"{split}_provider_sequences.npy"
        m_path = PROC / f"{split}_provider_masks.npy"

        if s_path.exists() and m_path.exists():
            s_arr = np.load(s_path)
            m_arr = np.load(m_path)
            seq_shapes[split] = s_arr.shape
            mask_shapes[split] = m_arr.shape

            results[split] = {
                "sequence_shape": list(s_arr.shape),
                "mask_shape": list(m_arr.shape),
                "seq_has_nan": bool(np.isnan(s_arr).any()),
                "mask_has_nan": bool(np.isnan(m_arr).any()),
                "mask_values_valid": bool(np.isin(m_arr, [0.0, 1.0]).all()),
                "mask_mean_valid_months": float(m_arr.sum(axis=1).mean()),
            }

    if seq_shapes and mask_shapes:
        results["consistent_seq_len"] = len(set(s[1] for s in seq_shapes.values())) == 1
        results["consistent_features"] = len(set(s[2] for s in seq_shapes.values())) == 1
        results["mask_seq_dimension_match"] = all(
            seq_shapes[k][0] == mask_shapes[k][0] and seq_shapes[k][1] == mask_shapes[k][1]
            for k in seq_shapes
        )
    return results


def verify_no_data_leakage(claims: pd.DataFrame) -> dict:
    """Check that claims never overlap across splits."""
    train_df = pd.read_csv(PROC / "train_provider_ids.csv")
    val_df   = pd.read_csv(PROC / "validation_provider_ids.csv")
    test_df  = pd.read_csv(PROC / "test_provider_ids.csv")

    train_pids = set(train_df["ProviderID"])
    val_pids   = set(val_df["ProviderID"])
    test_pids  = set(test_df["ProviderID"])

    train_claims = claims[claims["ProviderID"].isin(train_pids)]
    val_claims   = claims[claims["ProviderID"].isin(val_pids)]
    test_claims  = claims[claims["ProviderID"].isin(test_pids)]

    train_cids = set(train_claims["ClaimID"])
    val_cids   = set(val_claims["ClaimID"])
    test_cids  = set(test_claims["ClaimID"])

    return {
        "train_val_claim_overlap":  len(train_cids & val_cids),
        "train_test_claim_overlap": len(train_cids & test_cids),
        "val_test_claim_overlap":   len(val_cids & test_cids),
        "leakage_free": (
            len(train_cids & val_cids) == 0 and
            len(train_cids & test_cids) == 0 and
            len(val_cids & test_cids) == 0
        ),
    }


def verify_feature_specification() -> dict:
    """Verify that feature_spec.json matches actual generated feature columns."""
    spec_path = PROC / "feature_spec.json"
    train_feat = pd.read_csv(PROC / "train_provider_features.csv")

    with open(spec_path) as f:
        spec = json.load(f)

    spec_cols = set(spec.keys())
    feat_cols = set(train_feat.columns) - {"ProviderID", "PotentialFraud"}

    missing_in_spec = feat_cols - spec_cols
    extra_in_spec = spec_cols - feat_cols

    return {
        "n_features_in_spec": len(spec),
        "n_features_in_data": len(feat_cols),
        "spec_matches_data": (len(missing_in_spec) == 0 and len(extra_in_spec) == 0),
        "missing_in_spec": list(missing_in_spec),
        "extra_in_spec": list(extra_in_spec),
    }


def verify_no_demographic_features() -> dict:
    """Verify no protected demographic attributes are used as model features."""
    train_feat = pd.read_csv(PROC / "train_provider_features.csv")
    forbidden = {"Gender", "Race", "State", "County"}
    found = forbidden & set(train_feat.columns)
    return {
        "clean": len(found) == 0,
        "found_demographics": list(found),
    }


def print_final_summary(
    output_checks, overlap_check, fraud_dist, seq_check, leakage_check,
    feature_checks_path, spec_check, demo_check,
):
    sep = "=" * 70
    print(f"\n{sep}")
    print("  HEALTHGUARD AI — PHASE 1 CORRECTION & VERIFICATION SUMMARY")
    print(sep)

    n_ok   = sum(1 for r in output_checks if r["status"] == "OK")
    n_fail = sum(1 for r in output_checks if r["status"] != "OK")
    print(f"\n📁  OUTPUT FILES ({n_ok} OK, {n_fail} missing/empty):")
    for r in output_checks:
        icon = "✅" if r["status"] == "OK" else "❌"
        sz = f"{r['size_bytes']:,} B" if r["exists"] else "MISSING"
        print(f"  {icon} {r['file']}  [{sz}]")

    print(f"\n🔀  PROVIDER SPLIT:")
    print(f"  Train:      {overlap_check['train_count']:,} providers")
    print(f"  Validation: {overlap_check['val_count']:,} providers")
    print(f"  Test:       {overlap_check['test_count']:,} providers")
    icon = "✅" if overlap_check["overlap_free"] else "❌"
    print(f"  {icon} Overlap free: {overlap_check['overlap_free']}")

    print(f"\n🎯  FRAUD LABEL DISTRIBUTION:")
    for split, stats in fraud_dist.items():
        print(f"  {split:12s}: {stats['total']:4d} providers, "
              f"{stats['fraud']:3d} fraud ({stats['fraud_pct']:.1f}%)")

    print(f"\n🔢  LSTM SEQUENCES & MASKS:")
    for split in ["train", "validation", "test"]:
        if split in seq_check:
            s_shape = seq_check[split]["sequence_shape"]
            m_shape = seq_check[split]["mask_shape"]
            avg_m = seq_check[split]["mask_mean_valid_months"]
            nan_s = "⚠️ NaN" if seq_check[split]["seq_has_nan"] else "✅ clean"
            print(f"  {split:12s}: seq {s_shape} ({nan_s}), mask {m_shape} (avg valid months: {avg_m:.1f})")
    icon1 = "✅" if seq_check.get("consistent_seq_len") else "❌"
    icon2 = "✅" if seq_check.get("consistent_features") else "❌"
    icon3 = "✅" if seq_check.get("mask_seq_dimension_match") else "❌"
    print(f"  {icon1} Consistent seq_len: {seq_check.get('consistent_seq_len')}")
    print(f"  {icon2} Consistent n_features: {seq_check.get('consistent_features')}")
    print(f"  {icon3} Sequence/mask dimension match: {seq_check.get('mask_seq_dimension_match')}")

    print(f"\n🔒  DATA LEAKAGE CHECK:")
    icon = "✅" if leakage_check["leakage_free"] else "❌"
    print(f"  {icon} Leakage free: {leakage_check['leakage_free']}")
    for k, v in leakage_check.items():
        if "overlap" in k:
            print(f"    {k}: {v}")

    print(f"\n🛡️  ETHICAL / DEMOGRAPHIC FILTER:")
    icon = "✅" if demo_check["clean"] else "❌"
    print(f"  {icon} Protected demographic attributes excluded: {demo_check['clean']}")

    print(f"\n📋  FEATURE SPECIFICATION:")
    icon = "✅" if spec_check["spec_matches_data"] else "❌"
    print(f"  {icon} Spec matches generated features ({spec_check['n_features_in_spec']} features): {spec_check['spec_matches_data']}")

    print(f"\n🧪  AUTOMATED FEATURE CHECKS:")
    if feature_checks_path.exists():
        fc = pd.read_csv(feature_checks_path)
        for _, row in fc.iterrows():
            icon = "✅" if row["status"] == "PASS" else "❌"
            detail = f" — {row['detail']}" if pd.notna(row.get("detail")) and row["detail"] else ""
            print(f"  {icon} {row['check']}{detail}")
    else:
        print("  ⚠️  Feature checks file not found")

    print(f"\n{sep}")
    all_files_ok = (n_fail == 0)
    no_overlap   = overlap_check["overlap_free"]
    no_leakage   = leakage_check["leakage_free"]
    seqs_ok      = seq_check.get("consistent_seq_len") and seq_check.get("mask_seq_dimension_match")
    spec_ok      = spec_check["spec_matches_data"]
    demo_ok      = demo_check["clean"]
    overall = all_files_ok and no_overlap and no_leakage and seqs_ok and spec_ok and demo_ok
    status_msg = "✅ PHASE 1 CORRECTIONS COMPLETE & VERIFIED" if overall else "⚠️  PHASE 1 COMPLETE WITH WARNINGS"
    print(f"\n  {status_msg}")
    print(f"\n{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("\n" + "=" * 70)
    log.info("  HEALTHGUARD AI — PHASE 1 FULL PIPELINE (CORRECTED)")
    log.info("=" * 70)

    # ── Step 1: Preprocessing ─────────────────────────────────────────────────
    log.info("\n▶ STEP 1: PREPROCESSING")
    prep_result = run_preprocessing()

    # ── Step 2: Feature Engineering ───────────────────────────────────────────
    log.info("\n▶ STEP 2: FEATURE ENGINEERING")
    feat_result, feature_checks = run_feature_engineering()

    # ── Step 3: Reports & Charts ──────────────────────────────────────────────
    log.info("\n▶ STEP 3: REPORTS & CHARTS")
    run_reports()

    # ── Step 4: Final Verification ────────────────────────────────────────────
    log.info("\n▶ STEP 4: FINAL VERIFICATION")

    output_checks  = verify_outputs()
    overlap_check  = verify_no_provider_overlap()
    fraud_dist     = verify_fraud_distribution()
    seq_check      = verify_sequences_and_masks()

    claims = pd.read_csv(PROC / "cleaned_training_claims.csv", low_memory=False)
    leakage_check  = verify_no_data_leakage(claims)
    spec_check     = verify_feature_specification()
    demo_check     = verify_no_demographic_features()

    ver_report = {
        "output_files": output_checks,
        "provider_overlap": overlap_check,
        "fraud_distribution": fraud_dist,
        "sequences_and_masks": seq_check,
        "leakage": leakage_check,
        "feature_spec": spec_check,
        "demographic_check": demo_check,
    }
    with open(REP / "evaluation" / "phase1_verification.json", "w") as f:
        json.dump(ver_report, f, indent=2, default=str)

    print_final_summary(
        output_checks, overlap_check, fraud_dist, seq_check, leakage_check,
        REP / "evaluation" / "feature_checks.csv", spec_check, demo_check,
    )

    log.info("Phase 1 corrected pipeline finished.")
