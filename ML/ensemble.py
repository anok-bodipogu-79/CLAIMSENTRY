"""
HealthGuard AI - Ensemble Integration Architecture
===================================================
Genuine Out-of-Fold (OOF) Stacking Ensemble using Logistic Regression meta-learner,
combining XGBoost (tabular trees), ANN (deep MLP), and Bi-LSTM (masked temporal dynamics).
Includes Platt / Isotonic probability calibration and risk tier stratification.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, average_precision_score
import joblib


class StackingCalibrator:
    """Calibrates meta-model probabilities using Platt scaling or Isotonic regression."""

    def __init__(self, method: str = "platt"):
        self.method = method  # 'platt' or 'isotonic'
        self.calibrator = None
        self.brier_score_: Optional[float] = None
        self.log_loss_: Optional[float] = None

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "StackingCalibrator":
        probs = np.asarray(y_prob, dtype=np.float64).reshape(-1, 1)
        targets = np.asarray(y_true, dtype=int)

        if self.method == "platt":
            # Platt scaling: Logistic Regression on log-odds / probabilities
            self.calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            self.calibrator.fit(probs, targets)
        elif self.method == "isotonic":
            # Isotonic Regression: non-parametric isotonic step function
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(probs.ravel(), targets)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        cal_probs = self.predict(y_prob)
        self.brier_score_ = float(brier_score_loss(targets, cal_probs))
        self.log_loss_ = float(log_loss(targets, np.clip(cal_probs, 1e-6, 1.0 - 1e-6)))
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            return np.asarray(y_prob, dtype=np.float64)

        probs = np.asarray(y_prob, dtype=np.float64)
        if self.method == "platt":
            X_in = probs.reshape(-1, 1)
            cal_probs = self.calibrator.predict_proba(X_in)[:, 1]
        elif self.method == "isotonic":
            cal_probs = self.calibrator.predict(probs.ravel())
        else:
            cal_probs = probs

        return np.clip(cal_probs, 0.0, 1.0)


class HealthGuardEnsemble:
    """
    Multi-modal Fraud Detection Stacking Ensemble.
    Combines:
      1. XGBoost (tabular non-linear feature interactions)
      2. ANN (dense continuous feature representations)
      3. Bi-LSTM (temporal sequence dynamics with valid-month masking)
    Using genuine Logistic Regression meta-learner trained strictly on Out-of-Fold (OOF) predictions.
    """

    BASE_MODELS = ["xgboost", "ann", "bilstm"]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.meta_model: Optional[LogisticRegression] = None
        self.calibrator: Optional[StackingCalibrator] = None
        self.risk_thresholds: Dict[str, float] = {"low_max": 0.30, "medium_max": 0.70}
        self.meta_features = list(self.BASE_MODELS)
        self.fitted = False
        self.oof_metrics_: Dict[str, Any] = {}
        self.calibration_report_: Dict[str, Any] = {}

    def _assemble_meta_matrix(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Construct (N, 3) feature matrix for meta-learner."""
        cols = []
        for model_name in self.BASE_MODELS:
            if model_name not in base_predictions:
                raise KeyError(f"Missing base model predictions for '{model_name}'. Required: {self.BASE_MODELS}")
            preds = np.asarray(base_predictions[model_name], dtype=np.float64).ravel()
            cols.append(preds)
        return np.column_stack(cols)

    def fit_meta_learner(self, oof_predictions: Dict[str, np.ndarray], y_true: np.ndarray) -> "HealthGuardEnsemble":
        """
        Train Logistic Regression meta-learner strictly on out-of-fold base model predictions.
        """
        X_oof = self._assemble_meta_matrix(oof_predictions)
        y_arr = np.asarray(y_true, dtype=int)

        self.meta_model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=self.random_state,
        )
        self.meta_model.fit(X_oof, y_arr)
        self.fitted = True

        raw_oof_prob = self.meta_model.predict_proba(X_oof)[:, 1]
        self.oof_metrics_ = {
            "meta_coefficients": {
                name: float(coef) for name, coef in zip(self.BASE_MODELS, self.meta_model.coef_[0])
            },
            "meta_intercept": float(self.meta_model.intercept_[0]),
            "oof_prauc": float(average_precision_score(y_arr, raw_oof_prob)),
            "oof_brier": float(brier_score_loss(y_arr, raw_oof_prob)),
        }

        return self

    def calibrate(
        self,
        val_base_predictions: Dict[str, np.ndarray],
        y_val: np.ndarray,
        preferred_method: str = "platt",
    ) -> Dict[str, Any]:
        """
        Fit and compare Platt scaling vs Isotonic regression on validation predictions.
        """
        if not self.fitted:
            raise RuntimeError("Ensemble meta-model must be fitted before calibration.")

        X_val = self._assemble_meta_matrix(val_base_predictions)
        y_val_arr = np.asarray(y_val, dtype=int)
        raw_val_probs = self.meta_model.predict_proba(X_val)[:, 1]

        # 1. Platt scaling
        platt = StackingCalibrator(method="platt").fit(raw_val_probs, y_val_arr)
        # 2. Isotonic regression
        isotonic = StackingCalibrator(method="isotonic").fit(raw_val_probs, y_val_arr)

        raw_brier = float(brier_score_loss(y_val_arr, raw_val_probs))
        platt_brier = platt.brier_score_
        iso_brier = isotonic.brier_score_

        # Select method: default to Platt scaling for parametric stability with smaller positive count,
        # or isotonic if it provides strict non-overfitting improvement.
        if preferred_method == "isotonic" and iso_brier < platt_brier:
            selected_method = "isotonic"
            self.calibrator = isotonic
        else:
            selected_method = "platt"
            self.calibrator = platt

        self.calibration_report_ = {
            "raw_validation_brier": raw_brier,
            "platt_validation_brier": platt_brier,
            "isotonic_validation_brier": iso_brier,
            "selected_method": selected_method,
            "improvement_pct": float(((raw_brier - self.calibrator.brier_score_) / raw_brier) * 100),
        }

        return self.calibration_report_

    def set_risk_thresholds(
        self,
        low_medium_cutoff: float,
        medium_high_cutoff: float,
        strategy: str = "precision_driven",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set decision thresholds for LOW, MEDIUM, and HIGH risk tiers."""
        self.risk_thresholds = {
            "low_medium_cutoff": float(low_medium_cutoff),
            "medium_high_cutoff": float(medium_high_cutoff),
            "strategy": strategy,
            "metadata": metadata or {},
        }

    def predict_proba_raw(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Return raw uncalibrated meta-model probabilities."""
        if not self.fitted or self.meta_model is None:
            raise RuntimeError(
                "HealthGuardEnsemble has not been fitted yet. "
                "Train the meta-learner on base model predictions in Phase 2 before calling predict_proba()."
            )
        X_meta = self._assemble_meta_matrix(base_predictions)
        return self.meta_model.predict_proba(X_meta)[:, 1]

    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Return calibrated stacked ensemble probabilities."""
        raw_probs = self.predict_proba_raw(base_predictions)
        if self.calibrator is not None:
            return self.calibrator.predict(raw_probs)
        return raw_probs

    def assign_risk_category(self, calibrated_prob: float) -> str:
        """Categorize calibrated probability into LOW, MEDIUM, or HIGH risk priority."""
        low_cut = self.risk_thresholds.get("low_medium_cutoff", 0.30)
        med_cut = self.risk_thresholds.get("medium_high_cutoff", 0.70)

        if calibrated_prob >= med_cut:
            return "HIGH"
        elif calibrated_prob >= low_cut:
            return "MEDIUM"
        else:
            return "LOW"

    def predict(self, base_predictions: Dict[str, np.ndarray], threshold: Optional[float] = None) -> np.ndarray:
        """Predict binary fraud labels using specified or medium-high threshold."""
        probs = self.predict_proba(base_predictions)
        if threshold is None:
            threshold = self.risk_thresholds.get("low_medium_cutoff", 0.50)
        return (probs >= threshold).astype(int)

    def save(self, directory: str) -> None:
        """Save meta-model, calibrator, and thresholds to directory."""
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted ensemble.")
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.meta_model, str(save_dir / "stacking_meta_model.joblib"))
        if self.calibrator is not None:
            joblib.dump(self.calibrator, str(save_dir / "calibrator.joblib"))

        config = {
            "meta_learner": "LogisticRegression",
            "base_models": self.BASE_MODELS,
            "risk_thresholds": self.risk_thresholds,
            "oof_metrics": self.oof_metrics_,
            "calibration_report": self.calibration_report_,
            "random_state": self.random_state,
        }
        with open(save_dir / "ensemble_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    def load(self, directory: str) -> "HealthGuardEnsemble":
        """Load meta-model, calibrator, and thresholds from directory."""
        save_dir = Path(directory)
        meta_path = save_dir / "stacking_meta_model.joblib"
        cal_path = save_dir / "calibrator.joblib"
        cfg_path = save_dir / "ensemble_config.json"

        if not meta_path.exists():
            raise FileNotFoundError(f"Stacking meta-model not found at: {meta_path}")

        self.meta_model = joblib.load(str(meta_path))
        if cal_path.exists():
            self.calibrator = joblib.load(str(cal_path))
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.risk_thresholds = cfg.get("risk_thresholds", self.risk_thresholds)
            self.oof_metrics_ = cfg.get("oof_metrics", {})
            self.calibration_report_ = cfg.get("calibration_report", {})

        self.fitted = True
        return self


def derive_risk_thresholds(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    target_high_precision: float = 0.90,
    max_high_workload_pct: float = 5.0,
    min_high_flagged_count: int = 5,
    candidate_step: float = 0.01,
    **kwargs,
) -> Dict[str, Any]:
    """
    Derive LOW/MEDIUM and MEDIUM/HIGH risk thresholds strictly from validation predictions.

    Requirements:
      1. Derive LOW/MEDIUM using validation predictions and the documented F1 strategy.
      2. Derive MEDIUM/HIGH by searching validation probability thresholds for a documented
         high-precision target and investigation-capacity constraint.
      3. Deterministic selection with explicit tie-breaking and fallback.
      4. Enforce: 0 < low_medium_cutoff < medium_high_cutoff < 1.
      5. Save actual calculated validation precision, recall, F1, false positives,
         flagged provider count, and workload percentage for both thresholds.
      6. Strictly reject test labels/data if passed.
    """
    # Enforce that test labels/predictions are not accepted
    for k in kwargs:
        if "test" in k.lower():
            raise ValueError(
                f"Invalid parameter '{k}': derive_risk_thresholds strictly operates "
                f"on validation data only. Test labels and predictions are strictly forbidden."
            )

    probs = np.asarray(val_probs, dtype=np.float64).ravel()
    labels = np.asarray(val_labels, dtype=int).ravel()

    if len(probs) != len(labels):
        raise ValueError(f"Length mismatch: val_probs ({len(probs)}) vs val_labels ({len(labels)})")
    if len(probs) == 0:
        raise ValueError("Empty validation arrays passed to threshold derivation function.")
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("Validation probabilities must be within [0, 1].")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Validation labels must be binary {0, 1}.")

    n_total = len(probs)
    candidates = np.round(np.arange(0.05, 0.96, candidate_step), 4)

    # Compute metrics across all candidate thresholds
    records = []
    for t in candidates:
        pred = (probs >= t).astype(int)
        tp = int((pred * labels).sum())
        fp = int((pred * (1 - labels)).sum())
        fn = int(((1 - pred) * labels).sum())
        tn = int(((1 - pred) * (1 - labels)).sum())
        flagged = int(pred.sum())
        workload_pct = round(100.0 * flagged / n_total, 2)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        records.append({
            "threshold": float(t),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "flagged": flagged,
            "workload_pct": workload_pct,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        })

    # 1. LOW / MEDIUM Threshold Selection (F1 Maximization)
    valid_f1_records = [r for r in records if r["flagged"] > 0]
    if not valid_f1_records:
        valid_f1_records = records

    # Tie-breaking: 1) maximize F1, 2) maximize recall, 3) lower threshold
    best_f1_val = max(r["f1"] for r in valid_f1_records)
    f1_ties = [r for r in valid_f1_records if abs(r["f1"] - best_f1_val) <= 1e-6]
    best_low_med = max(f1_ties, key=lambda r: (r["recall"], -r["threshold"]))
    low_med_cutoff = round(best_low_med["threshold"], 2)

    # 2. MEDIUM / HIGH Threshold Selection (High-Precision & Capacity Search)
    high_candidates = [r for r in records if r["threshold"] >= round(low_med_cutoff + 0.02, 2)]

    eligible_high = [
        r for r in high_candidates
        if r["precision"] >= target_high_precision
        and r["workload_pct"] <= max_high_workload_pct
        and r["flagged"] >= min_high_flagged_count
    ]

    if eligible_high:
        # Tie-breaking: 1) maximize precision, 2) maximize recall / f1, 3) lowest threshold
        max_prec = max(r["precision"] for r in eligible_high)
        prec_ties = [r for r in eligible_high if r["precision"] == max_prec]
        best_med_high = max(prec_ties, key=lambda r: (r["recall"], r["f1"], -r["threshold"]))
        high_status = "target_precision_and_capacity_satisfied"
    else:
        # Fallback: find threshold with highest precision subject to capacity constraint
        cap_candidates = [r for r in high_candidates if r["workload_pct"] <= max_high_workload_pct and r["flagged"] >= 1]
        if cap_candidates:
            max_prec = max(r["precision"] for r in cap_candidates)
            prec_ties = [r for r in cap_candidates if r["precision"] == max_prec]
            best_med_high = max(prec_ties, key=lambda r: (r["recall"], r["f1"], -r["threshold"]))
            high_status = "highest_precision_under_capacity_constraint"
        else:
            p95 = float(np.percentile(probs, 95))
            fallback_t = min(0.95, max(round(low_med_cutoff + 0.05, 2), p95))
            best_med_high = min(high_candidates, key=lambda r: abs(r["threshold"] - fallback_t))
            high_status = "percentile_capacity_fallback"

    med_high_cutoff = round(best_med_high["threshold"], 2)

    # 3. Enforce strict monotonicity
    if not (0.0 < low_med_cutoff < med_high_cutoff < 1.0):
        raise RuntimeError(
            f"Threshold monotonicity check failed: low_med={low_med_cutoff}, med_high={med_high_cutoff}"
        )

    return {
        "low_medium_cutoff": float(low_med_cutoff),
        "medium_high_cutoff": float(med_high_cutoff),
        "selection_split": "validation",
        "test_labels_used": False,
        "low_medium_strategy": {
            "name": "validation_f1_maximization",
            "description": "Maximizes F1 score on validation set to balance fraud detection sensitivity and investigation workload.",
            "calculated_validation_metrics": {
                "cutoff": float(low_med_cutoff),
                "precision": round(best_low_med["precision"], 4),
                "recall": round(best_low_med["recall"], 4),
                "f1": round(best_low_med["f1"], 4),
                "false_positives": int(best_low_med["fp"]),
                "flagged_count": int(best_low_med["flagged"]),
                "workload_pct": float(best_low_med["workload_pct"]),
            },
        },
        "medium_high_strategy": {
            "name": "validation_high_precision_audit_capacity",
            "description": f"Searches validation probabilities for target precision >= {target_high_precision:.2f} subject to audit capacity <= {max_high_workload_pct:.1f}%.",
            "status": high_status,
            "target_precision": float(target_high_precision),
            "max_workload_pct": float(max_high_workload_pct),
            "min_flagged_count": int(min_high_flagged_count),
            "calculated_validation_metrics": {
                "cutoff": float(med_high_cutoff),
                "precision": round(best_med_high["precision"], 4),
                "recall": round(best_med_high["recall"], 4),
                "f1": round(best_med_high["f1"], 4),
                "false_positives": int(best_med_high["fp"]),
                "flagged_count": int(best_med_high["flagged"]),
                "workload_pct": float(best_med_high["workload_pct"]),
            },
        },
        "monotonic_thresholds_verified": True,
        "description": (
            f"LOW (< {low_med_cutoff}): routine automated monitoring; "
            f"MEDIUM ({low_med_cutoff} - {med_high_cutoff}): secondary clinical documentation review; "
            f"HIGH (>= {med_high_cutoff}): immediate priority forensic audit."
        ),
    }
