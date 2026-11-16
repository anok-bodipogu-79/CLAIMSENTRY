"""
HealthGuard AI - Explainability and Interpretability Module
===========================================================
SHAP (SHapley Additive exPlanations) attribution for local case
investigation and global fraud risk drivers.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd


class FraudExplainer:
    """SHAP Explainer wrapper for provider fraud investigation."""

    def __init__(self, model: Any = None, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names or []
        self.explainer = None

    def initialize_tree_explainer(self, background_data: pd.DataFrame = None):
        """Initialize TreeExplainer for XGBoost."""
        if self.model is None:
            raise RuntimeError(
                "Cannot initialize TreeExplainer: Provided model is None. "
                "Train the model in Phase 2 before initializing explainer."
            )
        import shap
        self.explainer = shap.TreeExplainer(self.model, data=background_data)
        return self

    def explain_provider(self, provider_features: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate feature attributions for a single provider.
        Returns top contributing risk factors with signed contributions.
        """
        if self.explainer is None:
            raise RuntimeError(
                "FraudExplainer has not been initialized. "
                "Model training and explainer setup in Phase 2 is required before calling explain_provider()."
            )

        shap_values = self.explainer.shap_values(provider_features)
        vals = shap_values[0] if isinstance(shap_values, list) else shap_values

        contributions = []
        for name, val in zip(self.feature_names, vals[0]):
            contributions.append({
                "feature": name,
                "shap_value": float(val),
                "direction": "INCREASES_FRAUD_RISK" if val > 0 else "DECREASES_FRAUD_RISK",
            })

        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return {
            "top_risk_factors": contributions[:10],
            "base_value": float(self.explainer.expected_value) if hasattr(self.explainer, "expected_value") else 0.0,
        }
