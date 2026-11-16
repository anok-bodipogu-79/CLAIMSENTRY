"""
HealthGuard AI - XGBoost Model Architecture & Training Module
==============================================================
Gradient Boosted Decision Trees with scale_pos_weight for handling
imbalanced provider fraud detection.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import xgboost as xgb


class XGBoostFraudDetector:
    """XGBoost Fraud Classifier with class weighting, PR-AUC monitoring, and early stopping."""

    def __init__(self, params: Optional[Dict[str, Any]] = None, random_state: int = 42):
        self.random_state = random_state
        self.params = params or {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.85,
            "colsample_bytree": 0.80,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "eval_metric": "aucpr",
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: List[str] = []
        self.evals_result_: Dict[str, Any] = {}
        self.best_iteration_: Optional[int] = None
        self.best_score_: Optional[float] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        early_stopping_rounds: int = 30,
        calculate_pos_weight: bool = True,
    ) -> "XGBoostFraudDetector":
        """
        Fit XGBoost model with class imbalance handling and validation monitoring.
        """
        self.feature_names = list(X_train.columns)

        # Dynamic positive class weighting from training data
        fit_params = dict(self.params)
        if calculate_pos_weight and "scale_pos_weight" not in fit_params:
            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())
            pos_weight = n_neg / max(n_pos, 1)
            fit_params["scale_pos_weight"] = round(pos_weight, 4)

        if eval_set is not None:
            fit_params["early_stopping_rounds"] = early_stopping_rounds

        self.model = xgb.XGBClassifier(**fit_params)

        evals = [(X_train, y_train)]
        if eval_set is not None:
            evals.append((eval_set[0], eval_set[1]))

        self.model.fit(
            X_train,
            y_train,
            eval_set=evals,
            verbose=False,
        )

        if hasattr(self.model, "evals_result"):
            self.evals_result_ = self.model.evals_result()
        if hasattr(self.model, "best_iteration"):
            self.best_iteration_ = self.model.best_iteration
        if hasattr(self.model, "best_score"):
            self.best_score_ = float(self.model.best_score)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return fraud probabilities [0, 1] for positive class."""
        if self.model is None:
            raise RuntimeError(
                "XGBoostFraudDetector has not been trained yet. "
                "Fit the model using Phase 2 training routines before calling predict_proba()."
            )
        if isinstance(X, pd.DataFrame):
            # Ensure correct column ordering
            X_ordered = X[self.feature_names]
        else:
            X_ordered = X
        probs = self.model.predict_proba(X_ordered)
        return probs[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary fraud labels based on decision threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def save(self, filepath: str) -> None:
        """Save model and metadata to disk."""
        if self.model is None:
            raise RuntimeError("Cannot save untrained model.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save native XGBoost model
        model_path = path.with_suffix(".json")
        self.model.save_model(str(model_path))

        # Save metadata
        meta = {
            "feature_names": self.feature_names,
            "params": self.params,
            "best_iteration": self.best_iteration_,
            "best_score": self.best_score_,
            "random_state": self.random_state,
        }
        meta_path = path.with_name(path.stem + "_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def load(self, filepath: str) -> "XGBoostFraudDetector":
        """Load model and metadata from disk."""
        path = Path(filepath)
        model_path = path.with_suffix(".json")
        meta_path = path.with_name(path.stem + "_meta.json")

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.feature_names = meta.get("feature_names", [])
            self.params = meta.get("params", self.params)
            self.best_iteration_ = meta.get("best_iteration")
            self.best_score_ = meta.get("best_score")
            self.random_state = meta.get("random_state", 42)

        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_path))
        return self
