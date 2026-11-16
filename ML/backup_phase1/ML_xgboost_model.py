"""
HealthGuard AI - XGBoost Model Architecture & Training Module
==============================================================
Gradient Boosted Decision Trees with scale_pos_weight for handling
imbalanced provider fraud detection.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


class XGBoostFraudDetector:
    """XGBoost Fraud Classifier with hyperparameter tuning & early stopping."""

    def __init__(self, params: Dict[str, Any] = None, random_state: int = 42):
        self.random_state = random_state
        self.params = params or {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 9.7,  # Ratio of non-fraud (4904) to fraud (506)
            "eval_metric": "aucpr",
            "random_state": self.random_state,
        }
        self.model = None
        self.feature_names = []

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            eval_set: Tuple[pd.DataFrame, pd.Series] = None):
        """Fit XGBoost model with validation monitoring."""
        import xgboost as xgb
        self.feature_names = list(X_train.columns)
        self.model = xgb.XGBClassifier(**self.params)
        evals = [eval_set] if eval_set is not None else None
        self.model.fit(X_train, y_train, eval_set=evals, verbose=False)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return fraud probabilities."""
        if self.model is None:
            raise RuntimeError(
                "XGBoostFraudDetector has not been trained yet. "
                "Fit the model using Phase 2 training routines before calling predict_proba()."
            )
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary fraud labels based on decision threshold."""
        if self.model is None:
            raise RuntimeError(
                "XGBoostFraudDetector has not been trained yet. "
                "Fit the model using Phase 2 training routines before calling predict()."
            )
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
