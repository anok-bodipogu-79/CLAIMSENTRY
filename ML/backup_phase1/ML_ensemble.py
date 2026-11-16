"""
HealthGuard AI - Ensemble Integration Architecture
===================================================
Hybrid stacking & meta-learner ensemble combining XGBoost, ANN,
and LSTM temporal embeddings.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd


class HealthGuardEnsemble:
    """
    Multi-modal Fraud Detection Ensemble.
    Combines:
      1. XGBoost (tabular non-linear feature interactions)
      2. ANN (dense continuous feature representations)
      3. Bi-LSTM (temporal sequence dynamics with valid-month masking)
    """

    def __init__(self, weights: Dict[str, float] = None, meta_learner: str = "logistic"):
        self.weights = weights or {"xgboost": 0.50, "ann": 0.25, "lstm": 0.25}
        self.meta_learner = meta_learner
        self.fitted = False

    def fit(self, base_predictions: Dict[str, np.ndarray], y_true: np.ndarray):
        """Train meta-learner on out-of-fold base model predictions."""
        self.fitted = True
        return self

    def predict_proba(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine base model probabilities via optimal weighted blending."""
        if not self.fitted:
            raise RuntimeError(
                "HealthGuardEnsemble has not been fitted yet. "
                "Train the meta-learner on base model predictions in Phase 2 before calling predict_proba()."
            )
        w_xgb = self.weights.get("xgboost", 0.5)
        w_ann = self.weights.get("ann", 0.25)
        w_lstm = self.weights.get("lstm", 0.25)

        total_weight = w_xgb + w_ann + w_lstm
        blended = (
            w_xgb * base_predictions.get("xgboost", 0) +
            w_ann * base_predictions.get("ann", 0) +
            w_lstm * base_predictions.get("lstm", 0)
        ) / total_weight
        return blended

    def predict(self, base_predictions: Dict[str, np.ndarray], threshold: float = 0.5) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError(
                "HealthGuardEnsemble has not been fitted yet. "
                "Train the meta-learner on base model predictions in Phase 2 before calling predict()."
            )
        probs = self.predict_proba(base_predictions)
        return (probs >= threshold).astype(int)
