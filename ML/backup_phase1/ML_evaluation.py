"""
HealthGuard AI - Model Evaluation & Cost-Sensitive Analysis Module
===================================================================
Computes ROC-AUC, PR-AUC, F1, Recall, Precision, and Cost-Benefit
optimal decision thresholds for Medicare fraud detection.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


class ModelEvaluator:
    """Comprehensive evaluation suite for imbalanced fraud classification."""

    def __init__(self, cost_fn: float = 10.0, cost_fp: float = 1.0):
        self.cost_fn = cost_fn  # Cost of missing a fraudulent provider
        self.cost_fp = cost_fp  # Cost of investigating an innocent provider

    def evaluate(self, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
        """Compute all primary performance metrics."""
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        total_cost = (fn * self.cost_fn) + (fp * self.cost_fp)

        return {
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
            "financial_loss_metric": float(total_cost),
            "decision_threshold": float(threshold),
        }

    def find_optimal_threshold(self, y_true: np.ndarray, y_prob: np.ndarray,
                               metric: str = "f1") -> Tuple[float, float]:
        """Find optimal decision threshold maximizing metric or minimizing cost."""
        thresholds = np.linspace(0.05, 0.95, 91)
        best_t, best_score = 0.5, -np.inf

        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            if metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "cost":
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                score = - ((fn * self.cost_fn) + (fp * self.cost_fp))
            else:
                score = recall_score(y_true, y_pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_t = t

        return float(best_t), float(best_score)
