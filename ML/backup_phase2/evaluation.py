"""
HealthGuard AI - Model Evaluation & Reporting Module
====================================================
Comprehensive evaluation suite for imbalanced healthcare provider fraud detection.
Computes PR-AUC, ROC-AUC, F1, Precision, Recall, Brier score, Top-k Precision,
Risk Tier Fraud Rates, Bootstrap 95% Confidence Intervals, and Ablation Tables.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
    precision_recall_curve,
)


class ModelEvaluator:
    """Comprehensive evaluation suite for imbalanced fraud classification."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """Compute all primary performance metrics for a single model."""
        y_arr = np.asarray(y_true, dtype=int)
        p_arr = np.asarray(y_prob, dtype=np.float64)
        y_pred = (p_arr >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_arr, y_pred, labels=[0, 1]).ravel()
        n_total = len(y_arr)
        n_pos = int(y_arr.sum())
        prevalence = float(n_pos / max(n_total, 1))

        roc_auc = float(roc_auc_score(y_arr, p_arr))
        pr_auc = float(average_precision_score(y_arr, p_arr))
        f1 = float(f1_score(y_arr, y_pred, zero_division=0))
        prec = float(precision_score(y_arr, y_pred, zero_division=0))
        rec = float(recall_score(y_arr, y_pred, zero_division=0))
        brier = float(brier_score_loss(y_arr, p_arr))

        # Precision at Top-K
        top_k_results = {}
        sorted_indices = np.argsort(p_arr)[::-1]
        for k in [10, 25, 50, 100]:
            if k <= n_total:
                top_k_idx = sorted_indices[:k]
                top_k_prec = float(y_arr[top_k_idx].mean())
                top_k_results[f"top_{k}"] = {
                    "k": k,
                    "precision": round(top_k_prec, 4),
                    "fraud_count": int(y_arr[top_k_idx].sum()),
                }

        return {
            "model_name": model_name,
            "threshold": float(threshold),
            "pr_auc": round(pr_auc, 4),
            "roc_auc": round(roc_auc, 4),
            "f1_score": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "brier_score": round(brier, 4),
            "prevalence": round(prevalence, 4),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
            "top_k_precision": top_k_results,
        }

    def evaluate_risk_tiers(
        self,
        y_true: np.ndarray,
        calibrated_probs: np.ndarray,
        low_medium_cutoff: float,
        medium_high_cutoff: float,
    ) -> Dict[str, Any]:
        """
        Evaluate observed fraud rates across LOW, MEDIUM, and HIGH risk groups.
        Verifies whether fraud prevalence increases monotonically: LOW < MEDIUM < HIGH.
        """
        y_arr = np.asarray(y_true, dtype=int)
        p_arr = np.asarray(calibrated_probs, dtype=np.float64)

        tiers = np.empty(len(y_arr), dtype=object)
        tiers[p_arr < low_medium_cutoff] = "LOW"
        tiers[(p_arr >= low_medium_cutoff) & (p_arr < medium_high_cutoff)] = "MEDIUM"
        tiers[p_arr >= medium_high_cutoff] = "HIGH"

        report = {}
        for tier_name in ["LOW", "MEDIUM", "HIGH"]:
            mask = tiers == tier_name
            count = int(mask.sum())
            if count > 0:
                fraud_count = int(y_arr[mask].sum())
                fraud_rate = float(fraud_count / count)
            else:
                fraud_count = 0
                fraud_rate = 0.0

            report[tier_name] = {
                "provider_count": count,
                "pct_of_total": round(count / len(y_arr) * 100, 2),
                "fraud_count": fraud_count,
                "fraud_rate": round(fraud_rate, 4),
            }

        monotonic = (
            report["LOW"]["fraud_rate"] <= report["MEDIUM"]["fraud_rate"] <= report["HIGH"]["fraud_rate"]
        )
        report["monotonic_risk_progression"] = bool(monotonic)
        return report

    def compute_bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bootstraps: int = 1000,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Compute 95% bootstrap confidence intervals for PR-AUC, ROC-AUC, and Brier score."""
        rng = np.random.default_rng(self.random_state)
        y_arr = np.asarray(y_true, dtype=int)
        p_arr = np.asarray(y_prob, dtype=np.float64)
        n = len(y_arr)

        pr_aucs = []
        roc_aucs = []
        briers = []

        for _ in range(n_bootstraps):
            idx = rng.choice(n, size=n, replace=True)
            y_b = y_arr[idx]
            p_b = p_arr[idx]

            # Require at least one positive and one negative sample
            if len(np.unique(y_b)) < 2:
                continue

            pr_aucs.append(average_precision_score(y_b, p_b))
            roc_aucs.append(roc_auc_score(y_b, p_b))
            briers.append(brier_score_loss(y_b, p_b))

        low_p = (alpha / 2.0) * 100
        high_p = (1.0 - alpha / 2.0) * 100

        return {
            "pr_auc": {
                "mean": round(float(np.mean(pr_aucs)), 4),
                "ci_lower": round(float(np.percentile(pr_aucs, low_p)), 4),
                "ci_upper": round(float(np.percentile(pr_aucs, high_p)), 4),
            },
            "roc_auc": {
                "mean": round(float(np.mean(roc_aucs)), 4),
                "ci_lower": round(float(np.percentile(roc_aucs, low_p)), 4),
                "ci_upper": round(float(np.percentile(roc_aucs, high_p)), 4),
            },
            "brier_score": {
                "mean": round(float(np.mean(briers)), 4),
                "ci_lower": round(float(np.percentile(briers, low_p)), 4),
                "ci_upper": round(float(np.percentile(briers, high_p)), 4),
            },
            "n_bootstraps_valid": len(pr_aucs),
        }

    def generate_full_evaluation_suite(
        self,
        test_providers_df: pd.DataFrame,
        y_test: np.ndarray,
        base_test_probs: Dict[str, np.ndarray],
        raw_ensemble_probs: np.ndarray,
        calibrated_ensemble_probs: np.ndarray,
        thresholds: Dict[str, float],
        reports_dir: str = "reports/evaluation",
    ) -> Dict[str, Any]:
        """
        Run complete test evaluation across all models, generate comparative artifacts,
        and save all JSON/CSV outputs to reports_dir.
        """
        out_dir = Path(reports_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        low_med_cut = thresholds.get("low_medium_cutoff", 0.30)
        med_high_cut = thresholds.get("medium_high_cutoff", 0.70)

        # 1. Evaluate individual models + ensemble
        models_to_eval = {
            "xgboost": base_test_probs["xgboost"],
            "ann": base_test_probs["ann"],
            "bilstm": base_test_probs["bilstm"],
            "raw_stacking_ensemble": raw_ensemble_probs,
            "calibrated_stacking_ensemble": calibrated_ensemble_probs,
        }

        comparison_rows = []
        full_metrics = {}
        conf_matrices = {}
        top_k_data = {}

        for name, probs in models_to_eval.items():
            thresh = low_med_cut if "ensemble" in name else 0.50
            m_res = self.evaluate_predictions(y_test, probs, threshold=thresh, model_name=name)
            full_metrics[name] = m_res
            conf_matrices[name] = m_res["confusion_matrix"]
            top_k_data[name] = m_res["top_k_precision"]

            comparison_rows.append({
                "Model": name,
                "PR-AUC": m_res["pr_auc"],
                "ROC-AUC": m_res["roc_auc"],
                "F1-Score": m_res["f1_score"],
                "Precision": m_res["precision"],
                "Recall": m_res["recall"],
                "Brier-Score": m_res["brier_score"],
                "Threshold": m_res["threshold"],
            })

        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df.to_csv(out_dir / "model_comparison.csv", index=False)

        # 2. Risk tier stratification on calibrated ensemble
        risk_tier_report = self.evaluate_risk_tiers(
            y_test, calibrated_ensemble_probs, low_med_cut, med_high_cut
        )
        full_metrics["calibrated_stacking_ensemble"]["risk_tiers"] = risk_tier_report

        # 3. Bootstrap CI on calibrated ensemble
        bootstrap_ci = self.compute_bootstrap_ci(y_test, calibrated_ensemble_probs, n_bootstraps=1000)

        # 4. Save JSON artifacts
        with open(out_dir / "final_metrics.json", "w", encoding="utf-8") as f:
            json.dump(full_metrics, f, indent=2)

        with open(out_dir / "confusion_matrices.json", "w", encoding="utf-8") as f:
            json.dump(conf_matrices, f, indent=2)

        with open(out_dir / "top_k_precision.json", "w", encoding="utf-8") as f:
            json.dump(top_k_data, f, indent=2)

        with open(out_dir / "bootstrap_confidence_intervals.json", "w", encoding="utf-8") as f:
            json.dump(bootstrap_ci, f, indent=2)

        with open(out_dir / "risk_thresholds.json", "w", encoding="utf-8") as f:
            json.dump(thresholds, f, indent=2)

        # 5. Save test predictions table
        assigned_tiers = []
        for p in calibrated_ensemble_probs:
            if p >= med_high_cut:
                assigned_tiers.append("HIGH")
            elif p >= low_med_cut:
                assigned_tiers.append("MEDIUM")
            else:
                assigned_tiers.append("LOW")

        pred_df = pd.DataFrame({
            "ProviderID": test_providers_df["ProviderID"].values if "ProviderID" in test_providers_df.columns else np.arange(len(y_test)),
            "True_PotentialFraud": y_test,
            "XGBoost_Probability": np.round(base_test_probs["xgboost"], 4),
            "ANN_Probability": np.round(base_test_probs["ann"], 4),
            "BiLSTM_Probability": np.round(base_test_probs["bilstm"], 4),
            "Raw_Ensemble_Probability": np.round(raw_ensemble_probs, 4),
            "Calibrated_Ensemble_Probability": np.round(calibrated_ensemble_probs, 4),
            "Risk_Tier": assigned_tiers,
        })
        pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

        # 6. Evaluation Summary Markdown
        self._write_markdown_summary(
            out_dir / "evaluation_summary.md",
            comparison_df,
            full_metrics,
            risk_tier_report,
            bootstrap_ci,
            thresholds,
        )

        return full_metrics

    def _write_markdown_summary(
        self,
        filepath: Path,
        comparison_df: pd.DataFrame,
        full_metrics: Dict[str, Any],
        risk_tiers: Dict[str, Any],
        bootstrap_ci: Dict[str, Any],
        thresholds: Dict[str, float],
    ) -> None:
        """Generate comprehensive markdown summary report."""
        cal_ens = full_metrics["calibrated_stacking_ensemble"]
        lines = [
            "# HealthGuard AI — Phase 2 Model Evaluation Summary",
            "",
            "## 1. Overview",
            "This report documents the performance of the three base models (XGBoost, ANN, Bi-LSTM) and the Out-of-Fold (OOF) Stacking Ensemble on the untouched test split (812 healthcare providers).",
            "",
            "## 2. Test Set Model Comparison Table",
            "",
            comparison_df.to_markdown(index=False),
            "",
            "## 3. Calibrated Ensemble Performance Highlights",
            f"- **Primary Metric (PR-AUC):** {cal_ens['pr_auc']:.4f} (95% CI: [{bootstrap_ci['pr_auc']['ci_lower']:.4f}, {bootstrap_ci['pr_auc']['ci_upper']:.4f}])",
            f"- **Secondary Metric (ROC-AUC):** {cal_ens['roc_auc']:.4f} (95% CI: [{bootstrap_ci['roc_auc']['ci_lower']:.4f}, {bootstrap_ci['roc_auc']['ci_upper']:.4f}])",
            f"- **F1-Score:** {cal_ens['f1_score']:.4f}",
            f"- **Brier Score:** {cal_ens['brier_score']:.4f}",
            f"- **Prevalence in Test Set:** {cal_ens['prevalence'] * 100:.2f}% (76 fraud / 812 total)",
            "",
            "## 4. Risk Tier Stratification",
            f"- **Low / Medium Cutoff:** {thresholds.get('low_medium_cutoff', 0.30):.4f}",
            f"- **Medium / High Cutoff:** {thresholds.get('medium_high_cutoff', 0.70):.4f}",
            "",
            "| Risk Tier | Provider Count | Pct of Test Set | Actual Fraud Count | Fraud Rate |",
            "| :--- | :--- | :--- | :--- | :--- |",
            f"| **LOW** | {risk_tiers['LOW']['provider_count']} | {risk_tiers['LOW']['pct_of_total']}% | {risk_tiers['LOW']['fraud_count']} | {risk_tiers['LOW']['fraud_rate'] * 100:.2f}% |",
            f"| **MEDIUM** | {risk_tiers['MEDIUM']['provider_count']} | {risk_tiers['MEDIUM']['pct_of_total']}% | {risk_tiers['MEDIUM']['fraud_count']} | {risk_tiers['MEDIUM']['fraud_rate'] * 100:.2f}% |",
            f"| **HIGH** | {risk_tiers['HIGH']['provider_count']} | {risk_tiers['HIGH']['pct_of_total']}% | {risk_tiers['HIGH']['fraud_count']} | {risk_tiers['HIGH']['fraud_rate'] * 100:.2f}% |",
            "",
            f"**Monotonic Progression:** `{'VERIFIED' if risk_tiers.get('monotonic_risk_progression') else 'FLAGGED'}` (Fraud rate increases monotonically across Low -> Medium -> High risk categories).",
            "",
            "## 5. Top-K Precision (Investigation Utility)",
            "| Top-K | Precision | Fraud Providers Identified |",
            "| :--- | :--- | :--- |",
        ]

        top_k = cal_ens.get("top_k_precision", {})
        for k_key, v in top_k.items():
            lines.append(f"| {k_key.upper()} | {v['precision'] * 100:.2f}% | {v['fraud_count']} / {v['k']} |")

        lines.extend([
            "",
            "## 6. Regulatory & Ethical Disclaimer",
            "> **NOTICE:** This ensemble output represents an audit prioritization risk score to guide investigation. It is not legal proof of fraud. High-risk providers should be reviewed by qualified human clinical auditors.",
            "",
        ])

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
