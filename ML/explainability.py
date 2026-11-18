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

        # Validate finiteness
        feat_vals = df_in.values.astype(np.float64)
        if not np.all(np.isfinite(feat_vals)):
            raise ValueError("Feature vector contains invalid numerical values: NaN or Infinity detected.")

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

            # Feature provenance & human-readable scope
            is_peer_metric = "to_peer_median" in feat_name
            feat_scope = "PEER_RELATIVE_BENCHMARK" if is_peer_metric else "PROVIDER_HISTORY_AGGREGATE"
            scope_badge = "Peer Benchmark" if is_peer_metric else "Provider History"

            display_title = feat_name.replace("_", " ").title()
            if "To Peer Median" in display_title:
                display_title = display_title.replace("To Peer Median", "(Ratio vs Peer Median)")
            elif feat_name == "total_reimbursement":
                display_title = "Provider Cumulative Reimbursement"
            elif feat_name == "total_claim_count":
                display_title = "Provider Cumulative Claim Count"
            elif feat_name == "unique_beneficiaries":
                display_title = "Provider Distinct Patients"
            elif feat_name == "avg_reimbursement":
                display_title = "Provider Average Reimbursement"

            contributions.append({
                "feature": feat_name,
                "display_name": display_title,
                "feature_scope": feat_scope,
                "scope_badge": scope_badge,
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

        # Top positive risk factors (increases fraud risk, largest positive first)
        positive_factors = [c for c in contributions if c["shap_value"] > 0]
        positive_factors.sort(key=lambda x: x["shap_value"], reverse=True)

        # Top negative risk factors (decreases fraud risk, most negative first)
        negative_factors = [c for c in contributions if c["shap_value"] < 0]
        negative_factors.sort(key=lambda x: x["shap_value"])

        # Sort all by absolute impact for primary top_risk_factors
        contributions_abs = sorted(contributions, key=lambda x: abs(x["shap_value"]), reverse=True)

        expected_val = (
            float(self.tree_explainer.expected_value[1])
            if isinstance(self.tree_explainer.expected_value, (list, np.ndarray)) and len(self.tree_explainer.expected_value) > 1
            else float(self.tree_explainer.expected_value)
        )

        return {
            "top_risk_factors": contributions_abs[:top_n],
            "top_positive_risk_factors": positive_factors[:5],
            "top_negative_risk_factors": negative_factors[:5],
            "positive_contributors": positive_factors[:5],
            "negative_contributors": negative_factors[:5],
            "total_features_analyzed": len(contributions),
            "base_value": round(expected_val, 5),
            "explanation_method": "SHAP_TreeExplainer_XGBoost",
            "scope_disclosure": (
                "SHAP feature attribution strictly explains the XGBoost tabular model component, "
                "which provides structured non-linear pattern analysis to the multi-modal ensemble. "
                "It does not directly explain ANN or Bi-LSTM representations. "
                "SHAP values represent statistical attribution, not direct causal effects."
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
        if not np.all(np.isfinite(seq_3d)):
            raise ValueError("Sequence array contains invalid numerical values: NaN or Infinity detected.")

        if seq_3d.ndim == 2:
            if seq_3d.shape != (12, 6):
                raise ValueError(f"Invalid sequence shape: expected (12, 6), got {seq_3d.shape}.")
            seq_3d = seq_3d[np.newaxis, :, :]
        elif seq_3d.ndim == 3:
            if seq_3d.shape != (1, 12, 6):
                raise ValueError(f"Invalid sequence shape: expected (1, 12, 6), got {seq_3d.shape}.")
        else:
            raise ValueError(f"Invalid sequence dimensions: expected 2D or 3D, got {seq_3d.ndim}D.")

        if mask is None:
            mask = np.ones((seq_3d.shape[0], seq_3d.shape[1]), dtype=np.float32)
        else:
            mask = np.asarray(mask, dtype=np.float32)
            if not np.all(np.isfinite(mask)):
                raise ValueError("Mask array contains invalid numerical values: NaN or Infinity detected.")
            if mask.ndim == 1:
                if mask.shape != (12,):
                    raise ValueError(f"Invalid mask shape: expected (12,), got {mask.shape}.")
                mask = mask[np.newaxis, :]
            elif mask.ndim == 2:
                if mask.shape != (1, 12):
                    raise ValueError(f"Invalid mask shape: expected (1, 12), got {mask.shape}.")
            else:
                raise ValueError(f"Invalid mask dimensions: expected 1D or 2D, got {mask.ndim}D.")

        # Baseline probability
        baseline_prob = float(lstm_model.predict_proba(seq_3d, mask)[0])

        sequence_feature_names = [
            "monthly_claim_count",
            "log_total_reimbursement",
            "unique_beneficiaries",
            "ip_claim_pct",
            "avg_reimbursement",
            "repeat_bene_count",
        ]

        monthly_impacts = []
        n_months = seq_3d.shape[1]

        for m_idx in range(n_months):
            is_valid = bool(mask[0, m_idx] > 0)
            if not is_valid:
                monthly_impacts.append({
                    "month_index": m_idx + 1,
                    "is_valid_observation": False,
                    "observation_type": "PADDING",
                    "impact": 0.0,
                    "direction": "NEUTRAL",
                    "note": "Leading padding (unobserved month)",
                })
                continue

            # Capture observed monthly features
            month_vals = {
                sequence_feature_names[f_i]: round(float(seq_3d[0, m_idx, f_i]), 4)
                for f_i in range(len(sequence_feature_names))
            }
            clm_cnt = month_vals.get("monthly_claim_count", 0.0)
            obs_type = "ACTIVE_CLAIMS" if clm_cnt > 0 else "ZERO_CLAIMS_IN_WINDOW"

            # Occlude month m_idx by setting features to 0 and temporarily zeroing mask
            occluded_seq = seq_3d.copy()
            occluded_mask = mask.copy()
            occluded_seq[0, m_idx, :] = 0.0
            occluded_mask[0, m_idx] = 0.0

            if occluded_mask.sum() == 0:
                occluded_mask[0, m_idx] = 1.0  # Prevent empty mask division

            occluded_prob = float(lstm_model.predict_proba(occluded_seq, occluded_mask)[0])
            impact = baseline_prob - occluded_prob

            if impact > 0.0001:
                direction_str = "ELEVATES_RISK"
            elif impact < -0.0001:
                direction_str = "LOWERS_RISK"
            else:
                direction_str = "NEUTRAL"

            monthly_impacts.append({
                "month_index": m_idx + 1,
                "is_valid_observation": True,
                "observation_type": obs_type,
                "impact": round(impact, 4),
                "direction": direction_str,
                "observed_features": month_vals,
            })

        # Identify most influential months
        valid_impacts = [m for m in monthly_impacts if m["is_valid_observation"]]
        most_influential = sorted(valid_impacts, key=lambda x: abs(x["impact"]), reverse=True)

        if not valid_impacts:
            top_month_desc = "No valid observation months in provider sequence."
        elif all(abs(m["impact"]) < 1e-4 for m in valid_impacts):
            top_month_desc = "Temporal sequence shows consistent billing patterns across active observation months."
        else:
            top_m = most_influential[0]
            action_desc = "elevating" if top_m["impact"] > 0 else "moderating"
            obs_note = " (zero claims in window)" if top_m.get("observation_type") == "ZERO_CLAIMS_IN_WINDOW" else ""
            top_month_desc = (
                f"Month {top_m['month_index']} had the highest temporal impact ({top_m['impact']:+.4f}){obs_note}, "
                f"{action_desc} fraud risk."
            )

        return {
            "baseline_lstm_probability": round(baseline_prob, 4),
            "monthly_temporal_impacts": monthly_impacts,
            "most_influential_months": most_influential[:3],
            "sequence_features_analyzed": sequence_feature_names,
            "trajectory_summary": top_month_desc,
            "explanation_method": "BiLSTM_Temporal_Occlusion",
            "scope_disclosure": (
                "Temporal occlusion attribution strictly explains the Bi-LSTM sequence component. "
                "It isolates billing dynamics across the 12-month sequence window."
            ),
        }
