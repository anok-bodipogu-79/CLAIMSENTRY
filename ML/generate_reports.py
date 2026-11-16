"""
HealthGuard AI - Phase 1: Report & Chart Generator
====================================================
Generates all required reports and charts from actual data.
Outputs saved to reports/charts/ and reports/evaluation/.

Usage:
    python ML/generate_reports.py     (run after preprocessing + feature_engineering)
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).resolve().parent.parent
PROC   = ROOT / "data" / "processed"
CHARTS = ROOT / "reports" / "charts"
EVAL   = ROOT / "reports" / "evaluation"

CHARTS.mkdir(parents=True, exist_ok=True)
EVAL.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("generate_reports")

# Style
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (9, 5)})


def save_fig(fig, name: str):
    path = CHARTS / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. PROVIDER FRAUD LABEL DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def chart_fraud_distribution(provider_labels: pd.DataFrame):
    counts = provider_labels["PotentialFraud"].map({1: "Fraud", 0: "Not Fraud"}).value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Bar chart
    bars = axes[0].bar(counts.index, counts.values,
                       color=["#e74c3c", "#2ecc71"], edgecolor="white", linewidth=0.8)
    axes[0].set_title("Provider Fraud Label Distribution", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Number of Providers")
    axes[0].set_xlabel("")
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:,}", ha="center", va="bottom", fontsize=11)

    # Pie chart
    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=["#e74c3c", "#2ecc71"], startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[1].set_title("Fraud vs Non-Fraud Split", fontsize=13, fontweight="bold")

    fig.suptitle("HealthGuard AI — Provider Fraud Labels", fontsize=14, y=1.01)
    fig.tight_layout()
    save_fig(fig, "fraud_label_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. INPATIENT vs OUTPATIENT DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def chart_claim_type_distribution(claims: pd.DataFrame):
    counts = claims["ClaimType"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    bars = axes[0].bar(counts.index, counts.values,
                       color=["#3498db", "#f39c12"], edgecolor="white", linewidth=0.8)
    axes[0].set_title("Inpatient vs Outpatient Claim Count", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Number of Claims")
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                     f"{val:,}", ha="center", va="bottom", fontsize=11)

    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%",
                colors=["#3498db", "#f39c12"], startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[1].set_title("Claim Type Split", fontsize=13, fontweight="bold")

    # Reimbursement by type
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    reimb_by_type = claims.groupby("ClaimType")["InscClaimAmtReimbursed"].mean()
    bars2 = ax2.bar(reimb_by_type.index, reimb_by_type.values,
                    color=["#3498db", "#f39c12"], edgecolor="white")
    ax2.set_title("Average Reimbursement by Claim Type", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Average Reimbursement ($)")
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for bar, val in zip(bars2, reimb_by_type.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                 f"${val:,.0f}", ha="center", va="bottom", fontsize=11)
    fig2.tight_layout()
    save_fig(fig2, "avg_reimbursement_by_claim_type.png")

    fig.suptitle("HealthGuard AI — Claim Type Distribution", fontsize=14, y=1.01)
    fig.tight_layout()
    save_fig(fig, "claim_type_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. REIMBURSEMENT DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def chart_reimbursement_distribution(claims: pd.DataFrame):
    reimb = claims["InscClaimAmtReimbursed"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Raw distribution (capped at 99th percentile for readability)
    cap = reimb.quantile(0.99)
    capped = reimb[reimb <= cap]
    axes[0].hist(capped, bins=60, color="#2980b9", edgecolor="white", linewidth=0.3, alpha=0.85)
    axes[0].set_title(f"Reimbursement Distribution (≤ 99th pct, cap=${cap:,.0f})",
                      fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Reimbursement ($)")
    axes[0].set_ylabel("Claim Count")
    axes[0].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Log distribution
    log_reimb = np.log1p(reimb)
    axes[1].hist(log_reimb, bins=60, color="#8e44ad", edgecolor="white", linewidth=0.3, alpha=0.85)
    axes[1].set_title("Log(1 + Reimbursement) Distribution", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("log(1 + Reimbursement)")
    axes[1].set_ylabel("Claim Count")

    fig.suptitle("HealthGuard AI — Claim Reimbursement Distribution", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, "reimbursement_distribution.png")

    # By fraud label (provider-level)
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    if "PotentialFraud" in claims.columns:
        for label, grp in claims.groupby("PotentialFraud"):
            lname = "Fraud" if label == 1 else "Not Fraud"
            ax3.hist(np.log1p(grp["InscClaimAmtReimbursed"].dropna()),
                     bins=50, alpha=0.6, label=lname, density=True)
        ax3.set_title("Log-Reimbursement Distribution by Fraud Label",
                      fontsize=12, fontweight="bold")
        ax3.set_xlabel("log(1 + Reimbursement)")
        ax3.set_ylabel("Density")
        ax3.legend()
        fig3.tight_layout()
        save_fig(fig3, "reimbursement_by_fraud_label.png")
    else:
        plt.close(fig3)


# ══════════════════════════════════════════════════════════════════════════════
# 4. PROVIDER-LEVEL FEATURE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def chart_provider_feature_summary(train_feat: pd.DataFrame):
    if train_feat.empty:
        log.warning("  Empty train features — skipping provider feature charts")
        return

    # Summary statistics CSV
    numeric_cols = train_feat.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {"PotentialFraud"}
    numeric_cols = [c for c in numeric_cols if c not in exclude]
    summary = train_feat[numeric_cols].describe().T
    summary.to_csv(EVAL / "provider_feature_summary.csv")
    log.info("  Saved provider_feature_summary.csv")

    # Feature distributions: top 6 by std deviation
    top_features = (
        summary["std"].sort_values(ascending=False).head(6).index.tolist()
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feat in zip(axes.flat, top_features):
        data_all = train_feat[feat].dropna()
        if "PotentialFraud" in train_feat.columns:
            for label, grp in train_feat.groupby("PotentialFraud"):
                lname = "Fraud" if label == 1 else "Not Fraud"
                ax.hist(np.log1p(grp[feat].clip(lower=0).dropna()),
                        bins=30, alpha=0.6, label=lname, density=True)
            ax.legend(fontsize=7)
        else:
            ax.hist(np.log1p(data_all.clip(lower=0)), bins=30, alpha=0.8)
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel("log(1+x)")
        ax.set_ylabel("Density")
        ax.tick_params(labelsize=7)

    fig.suptitle("Top Provider Features by Variance (Train Split)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "provider_feature_distributions.png")

    # Fraud vs non-fraud mean comparison
    if "PotentialFraud" in train_feat.columns:
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        means = train_feat.groupby("PotentialFraud")[top_features].mean()
        means.index = ["Not Fraud", "Fraud"]
        norm_means = means.div(means.max(axis=0), axis=1).T.fillna(0)
        norm_means.plot(kind="bar", ax=ax4, color=["#2ecc71", "#e74c3c"],
                        edgecolor="white")
        ax4.set_title("Normalised Mean of Top Features: Fraud vs Not Fraud",
                      fontsize=12, fontweight="bold")
        ax4.set_ylabel("Normalised Mean (0–1)")
        ax4.set_xlabel("Feature")
        ax4.tick_params(axis="x", rotation=30, labelsize=8)
        ax4.legend(title="Label")
        fig4.tight_layout()
        save_fig(fig4, "feature_means_fraud_vs_notfraud.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAIN / VAL / TEST SPLIT DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def chart_split_distribution(train_feat, val_feat, test_feat):
    if "PotentialFraud" not in train_feat.columns:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, (name, feat) in zip(axes, [("Train", train_feat),
                                        ("Validation", val_feat),
                                        ("Test", test_feat)]):
        counts = feat["PotentialFraud"].map({1: "Fraud", 0: "Not Fraud"}).value_counts()
        ax.bar(counts.index, counts.values,
               color=["#e74c3c", "#2ecc71"], edgecolor="white")
        ax.set_title(f"{name} Split\n({len(feat)} providers)", fontsize=11)
        ax.set_ylabel("Count")
        for i, (val) in enumerate(counts.values):
            ax.text(i, val + 0.5, f"{val}", ha="center", va="bottom", fontsize=10)
    fig.suptitle("Fraud Label Distribution Across Splits", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "split_fraud_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_reports():
    log.info("=" * 70)
    log.info("HEALTHGUARD AI — Phase 1: Report Generation")
    log.info("=" * 70)

    # Load data
    log.info("\n[1] Loading processed data for reporting…")

    claims = pd.read_csv(PROC / "cleaned_training_claims.csv", low_memory=False)
    for col in ["ClaimStartDt", "ClaimEndDt"]:
        if col in claims.columns:
            claims[col] = pd.to_datetime(claims[col], errors="coerce")
    log.info(f"  Claims: {len(claims):,}")

    provider_labels = pd.read_csv(PROC / "provider_labels.csv")
    log.info(f"  Providers: {len(provider_labels):,}")

    # Feature files
    try:
        train_feat = pd.read_csv(PROC / "train_provider_features.csv")
        val_feat   = pd.read_csv(PROC / "validation_provider_features.csv")
        test_feat  = pd.read_csv(PROC / "test_provider_features.csv")
    except FileNotFoundError as e:
        log.warning(f"  Feature file not found: {e} — skipping feature charts")
        train_feat = val_feat = test_feat = pd.DataFrame()

    # Merge fraud labels into claims for label-stratified charts
    if "PotentialFraud" not in claims.columns and "ProviderID" in claims.columns:
        claims = claims.merge(provider_labels[["ProviderID", "PotentialFraud"]],
                              on="ProviderID", how="left")

    log.info("\n[2] Generating charts…")
    chart_fraud_distribution(provider_labels)
    chart_claim_type_distribution(claims)
    chart_reimbursement_distribution(claims)

    if not train_feat.empty:
        chart_provider_feature_summary(train_feat)
        chart_split_distribution(train_feat, val_feat, test_feat)

    log.info("\n[✓] Report generation complete.")


if __name__ == "__main__":
    run_reports()
    print("\nDone. Charts in reports/charts/, stats in reports/evaluation/")
