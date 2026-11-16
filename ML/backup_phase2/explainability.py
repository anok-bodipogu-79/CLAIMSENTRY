"""
HealthGuard AI - Explainability and Interpretability Module
===========================================================
SHAP (SHapley Additive exPlanations) TreeExplainer attribution for XGBoost tabular features,
combined with Phase 1 peer reference benchmark distributions and temporal occlusion analysis.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
import shap


class FraudExplainer:
    """
    SHAP-based Feature Attribution and Temporal Occlusion Explainer.
    Explains the XGBoost tabular model and Bi-LSTM temporal sequence dynamics with peer benchmarks.
    """

    def __init__(
        self,
        xgboost_model: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
        peer_reference_path: Optional[str] = None,
    ):
        self.xgboost_model = xgboost_model
        self.feature_names = feature_names or []
        self.peer_reference_path = peer_reference_path
        self.peer_reference_table: Dict[str, Any] = {}
        self.tree_explainer: Optional[shap.TreeExplainer] = None

        if self.peer_reference_path:
            self._load_peer_reference()

        if self.xgboost_model is not None:
            self.initialize_tree_explainer()

    def _load_peer_reference(self) -> None:
        """Load Phase 1 training peer reference distribution table."""
        path = Path(self.peer_reference_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.peer_reference_table = data.get("feature_distributions", data)

    def initialize_tree_explainer(self, background_data: Optional[pd.DataFrame] = None) -> "FraudExplainer":
        """Initialize TreeExplainer for the trained XGBoost model."""
        if self.xgboost_model is None:
            raise RuntimeError(
                "Cannot initialize TreeExplainer: XGBoost model is None. "
                "Train the model in Phase 2 before initializing explainer."
            )

        underlying_model = getattr(self.xgboost_model, "model", self.xgboost_model)
        if underlying_model is None:
            raise RuntimeError("Underlying XGBoost model instance is None.")

        self.tree_explainer = shap.TreeExplainer(underlying_model, data=background_data)
        if hasattr(self.xgboost_model, "feature_names") and self.xgboost_model.feature_names:
            self.feature_names = list(self.xgboost_model.feature_names)

        return self

    def explain_provider(
        self,
        provider_features: Union[pd.DataFrame, pd.Series, Dict[str, float]],
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """
        Generate local feature attributions for a single provider using XGBoost SHAP TreeExplainer.
        Includes peer percentile comparisons and direction of risk attribution.
        """
        if self.tree_explainer is None:
            raise RuntimeError(
                "FraudExplainer has not been initialized. "
                "Fit the XGBoost model and call initialize_tree_explainer() before calling explain_provider()."
            )

        # Prepare 1-row DataFrame with correct column names
        if isinstance(provider_features, dict):
            df_in = pd.DataFrame([provider_features])
        elif isinstance(provider_features, pd.Series):
            df_in = pd.DataFrame([provider_features])
        elif isinstance(provider_features, pd.DataFrame):
            df_in = provider_features.iloc[0:1].copy()
        else:
            df_in = pd.DataFrame(provider_features)

        if self.feature_names:
            # Reindex to ensure strict feature ordering matching training
            for col in self.feature_names:
                if col not in df_in.columns:
                    df_in[col] = 0.0
            df_in = df_in[self.feature_names]

        shap_vals = self.tree_explainer.shap_values(df_in)
        if isinstance(shap_vals, list):
            vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        else:
            vals = shap_vals

        if vals.ndim == 2:
            row_vals = vals[0]
        else:
            row_vals = vals

        contributions = []
        for feat_name, val in zip(self.feature_names, row_vals):
            actual_val = float(df_in[feat_name].iloc[0])
            shap_score = float(val)

            # Peer comparison lookup
            peer_stats = self.peer_reference_table.get(feat_name, {})
            p50 = peer_stats.get("p50", peer_stats.get("median", peer_stats.get("mean", None)))
            p75 = peer_stats.get("p75", None)
            p90 = peer_stats.get("p90", None)
            p95 = peer_stats.get("p95", None)
            p25 = peer_stats.get("p25", None)
            p10 = peer_stats.get("p10", None)

            # Estimate percentile
            pct = 50.0
            if p95 is not None and actual_val >= p95:
                pct = 95.0
            elif p90 is not None and actual_val >= p90:
                pct = 90.0
            elif p75 is not None and actual_val >= p75:
                pct = 75.0
            elif p50 is not None and actual_val >= p50:
                pct = 50.0
            elif p25 is not None and actual_val >= p25:
                pct = 25.0
            elif p10 is not None and actual_val <= p10:
                pct = 10.0
            else:
                pct = 40.0

            summary_str = f"{pct:.0f}th percentile"
            if p50 is not None:
                summary_str += f" (Peer median: {round(p50, 2)})"

            contributions.append({
                "feature": feat_name,
                "shap_value": round(shap_score, 5),
                "actual_value": round(actual_val, 4),
                "feature_value": round(actual_val, 4),
                "direction": "INCREASES_FRAUD_RISK" if shap_score > 0 else "DECREASES_FRAUD_RISK",
                "peer_percentile": pct,
                "peer_median": round(p50, 4) if p50 is not None else None,
                "peer_p75": round(p75, 4) if p75 is not None else None,
                "peer_p90": round(p90, 4) if p90 is not None else None,
                "benchmark_summary": summary_str,
            })

        # Sort by absolute impact
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        expected_val = (
            float(self.tree_explainer.expected_value[1])
            if isinstance(self.tree_explainer.expected_value, (list, np.ndarray)) and len(self.tree_explainer.expected_value) > 1
            else float(self.tree_explainer.expected_value)
        )

        return {
            "top_risk_factors": contributions[:top_n],
            "total_features_analyzed": len(contributions),
            "base_value": round(expected_val, 5),
            "explanation_method": "SHAP_TreeExplainer_XGBoost",
            "scope_disclosure": (
                "SHAP feature attribution strictly explains the XGBoost tabular model component, "
                "which provides structured non-linear pattern analysis to the multi-modal ensemble. "
                "It does not directly explain ANN or Bi-LSTM representations."
            ),
        }

    def explain_temporal_sequence(
        self,
        lstm_model: Any,
        sequence: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Explain monthly temporal impact using genuine feature ablation / occlusion on the Bi-LSTM.
        For each valid month, measures change in predicted probability when occluding that month.
        """
        if lstm_model is None or getattr(lstm_model, "model", None) is None:
            raise RuntimeError("Trained LSTMFraudDetector required for temporal explanation.")

        seq_3d = np.asarray(sequence, dtype=np.float32)
        if seq_3d.ndim == 2:
            seq_3d = seq_3d[np.newaxis, :, :]

        if mask is None:
            mask = np.ones((seq_3d.shape[0], seq_3d.shape[1]), dtype=np.float32)
        elif mask.ndim == 1:
            mask = mask[np.newaxis, :]

        # Baseline probability
        baseline_prob = float(lstm_model.predict_proba(seq_3d, mask)[0])

        monthly_impacts = []
        n_months = seq_3d.shape[1]

        for m_idx in range(n_months):
            is_valid = bool(mask[0, m_idx] > 0)
            if not is_valid:
                monthly_impacts.append({
                    "month_index": m_idx + 1,
                    "is_valid_observation": False,
                    "impact": 0.0,
                    "note": "Leading padding (zero weight)",
                })
                continue

            # Occlude month m_idx by setting features to 0 and temporaily zeroing mask
            occluded_seq = seq_3d.copy()
            occluded_mask = mask.copy()
            occluded_seq[0, m_idx, :] = 0.0
            occluded_mask[0, m_idx] = 0.0

            if occluded_mask.sum() == 0:
                occluded_mask[0, m_idx] = 1.0  # Prevent empty mask division

            occluded_prob = float(lstm_model.predict_proba(occluded_seq, occluded_mask)[0])
            impact = baseline_prob - occluded_prob

            monthly_impacts.append({
                "month_index": m_idx + 1,
                "is_valid_observation": True,
                "impact": round(impact, 4),
                "direction": "ELEVATES_RISK" if impact > 0 else "LOWERS_RISK",
            })

        return {
            "baseline_lstm_probability": round(baseline_prob, 4),
            "monthly_temporal_impacts": monthly_impacts,
            "explanation_method": "BiLSTM_Temporal_Occlusion",
            "scope_disclosure": (
                "Temporal occlusion attribution strictly explains the Bi-LSTM sequence component. "
                "It isolates the billing dynamics across the 12-month sequence window."
            ),
        }
