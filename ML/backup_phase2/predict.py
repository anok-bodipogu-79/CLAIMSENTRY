"""
HealthGuard AI - Shared Production Inference Pipeline
======================================================
Unified prediction service providing deterministic fraud probability scoring,
multi-modal base model breakdown, calibrated stacked ensemble risk tier,
and SHAP explanation factor attribution.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

from ML.xgboost_model import XGBoostFraudDetector
from ML.ann_model import ANNFraudDetector
from ML.lstm_model import LSTMFraudDetector
from ML.ensemble import HealthGuardEnsemble
from ML.explainability import FraudExplainer


REQUIRED_ARTIFACT_FILENAMES = [
    "xgboost_model.json",
    "xgboost_model_meta.json",
    "ann_model.pt",
    "ann_model_meta.json",
    "ann_model_scaler.joblib",
    "lstm_model.pt",
    "lstm_model_meta.json",
    "lstm_model_scaler.joblib",
    "stacking_meta_model.joblib",
    "calibrator.joblib",
    "ensemble_config.json",
    "oof_predictions.csv",
]


class HealthGuardPredictor:
    """Production inference engine loading verified trained artifacts with cryptographic integrity checks."""

    DISCLAIMER = "This is a risk assessment to guide investigation. It is not proof of fraud."

    def __init__(self, artifact_version: Optional[str] = None):
        if artifact_version is None:
            if (ROOT / "ML" / "artifacts" / "v2" / "model_manifest.json").exists():
                artifact_version = "v2"
            else:
                artifact_version = "v1"
        self.artifact_version = artifact_version
        self.artifacts_dir = ROOT / "ML" / "artifacts" / self.artifact_version
        self.proc_dir = ROOT / "data" / "processed"

        self.manifest: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.xgboost_detector: Optional[XGBoostFraudDetector] = None
        self.ann_detector: Optional[ANNFraudDetector] = None
        self.lstm_detector: Optional[LSTMFraudDetector] = None
        self.ensemble: Optional[HealthGuardEnsemble] = None
        self.explainer: Optional[FraudExplainer] = None
        self.is_ready: bool = False
        self.verification_error: Optional[str] = None

        self._load_artifacts()

    def verify_artifact_integrity(self, target_dir: Optional[Path] = None) -> Tuple[bool, Optional[str]]:
        """
        Verify every SHA-256 checksum in model_manifest.json before loading models.
        Rejects manifest if:
          - model_manifest.json is missing or corrupted
          - training_status != "TRAINED"
          - sha256_checksums dictionary is missing or empty
          - any required model artifact file is missing
          - any computed SHA-256 hash mismatches the manifest
        """
        art_dir = target_dir or self.artifacts_dir
        manifest_path = art_dir / "model_manifest.json"
        if not manifest_path.exists():
            return False, f"model_manifest.json missing in {art_dir}"

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception as e:
            return False, f"Failed to parse model_manifest.json: {e}"

        # 1. Training status check
        if manifest_data.get("training_status") != "TRAINED":
            return False, f"Manifest training_status is '{manifest_data.get('training_status')}', expected 'TRAINED'."

        # 2. Checksum dictionary check
        checksums = manifest_data.get("sha256_checksums")
        if not checksums or not isinstance(checksums, dict):
            return False, "model_manifest.json lacks required 'sha256_checksums' dictionary or is empty."

        # 3. File existence and cryptographic checksum verification
        for fname in REQUIRED_ARTIFACT_FILENAMES:
            fpath = art_dir / fname
            if not fpath.exists():
                return False, f"Required model artifact file missing: {fname}"

            expected_hash = checksums.get(fname)
            if not expected_hash:
                return False, f"Missing SHA-256 checksum for required artifact: {fname}"

            actual_hash = hashlib.sha256(fpath.read_bytes()).hexdigest()
            if actual_hash.lower() != expected_hash.lower():
                return False, (
                    f"Cryptographic checksum mismatch for '{fname}': "
                    f"expected {expected_hash}, calculated {actual_hash}"
                )

        return True, None

    def _load_artifacts(self) -> None:
        """Load and verify all trained Phase 2 artifacts."""
        self.is_ready = False
        self.verification_error = None

        # Verify integrity before attempting any model loading
        is_valid, err_msg = self.verify_artifact_integrity()
        if not is_valid:
            self.is_ready = False
            self.verification_error = err_msg
            return

        with open(self.artifacts_dir / "model_manifest.json", "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.feature_names = self.manifest.get("feature_names", [])

        # Load models
        try:
            self.xgboost_detector = XGBoostFraudDetector().load(str(self.artifacts_dir / "xgboost_model"))
            self.ann_detector = ANNFraudDetector().load(str(self.artifacts_dir / "ann_model"))
            self.lstm_detector = LSTMFraudDetector().load(str(self.artifacts_dir / "lstm_model"))
            self.ensemble = HealthGuardEnsemble().load(str(self.artifacts_dir))

            peer_ref_path = str(self.proc_dir / "peer_reference_table.json")
            self.explainer = FraudExplainer(
                xgboost_model=self.xgboost_detector,
                feature_names=self.feature_names,
                peer_reference_path=peer_ref_path,
            )
            # Set is_ready=True only after successful checksum verification AND successful model instantiation
            self.is_ready = True
            self.verification_error = None
        except Exception as e:
            self.is_ready = False
            self.verification_error = f"Model instantiation failed: {str(e)}"

    def validate_features(self, features: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
        """Validate input features against expected feature specification."""
        if not self.feature_names:
            raise RuntimeError("Model artifacts not initialized; cannot validate features.")

        missing_cols = [c for c in self.feature_names if c not in features]
        if missing_cols:
            raise ValueError(f"Missing {len(missing_cols)} required features: {missing_cols[:5]}...")

        # Assemble single row DataFrame in exact training order
        row_dict = {}
        for col in self.feature_names:
            val = features.get(col, 0.0)
            try:
                row_dict[col] = float(val) if val is not None else 0.0
            except (ValueError, TypeError):
                row_dict[col] = 0.0

        df_row = pd.DataFrame([row_dict], columns=self.feature_names)
        warnings = []
        return df_row, warnings

    def validate_sequence_and_mask(
        self,
        sequence: Any,
        mask: Any,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Validate genuine provider monthly sequence and validity mask.
        Ensures:
          - Sequence and mask are provided (not None).
          - Sequence shape is (12, 6) or (1, 12, 6).
          - Mask shape is (12,) or (1, 12).
          - Finite numerical values (no NaN, Inf, or nulls).
          - Mask contains valid binary indicators (0 or 1).
        """
        if sequence is None:
            raise ValueError(
                "Missing monthly sequence: Genuine provider 12-month temporal sequence is strictly required "
                "for multi-modal ensemble prediction. Fallback or synthetic sequences are disabled."
            )

        if mask is None:
            raise ValueError(
                "Missing sequence validity mask: Monthly sequence validity mask is strictly required "
                "for temporal Bi-LSTM inference."
            )

        try:
            seq_arr = np.asarray(sequence, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Sequence conversion to float array failed: {e}")

        try:
            mask_arr = np.asarray(mask, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Mask conversion to float array failed: {e}")

        # Check finiteness (no NaN or Inf)
        if not np.all(np.isfinite(seq_arr)):
            raise ValueError("Sequence contains invalid numerical values: NaN or Infinity detected.")
        if not np.all(np.isfinite(mask_arr)):
            raise ValueError("Mask contains invalid numerical values: NaN or Infinity detected.")

        # Validate sequence dimensions: (12, 6) or (1, 12, 6)
        if seq_arr.ndim == 2:
            if seq_arr.shape != (12, 6):
                raise ValueError(f"Invalid sequence shape: expected (12, 6), got {seq_arr.shape}.")
            seq_arr = seq_arr[np.newaxis, :, :]
        elif seq_arr.ndim == 3:
            if seq_arr.shape != (1, 12, 6):
                raise ValueError(f"Invalid sequence shape: expected (1, 12, 6), got {seq_arr.shape}.")
        else:
            raise ValueError(
                f"Invalid sequence dimensions: expected 2D (12, 6) or 3D (1, 12, 6), "
                f"got {seq_arr.ndim}D shape {seq_arr.shape}."
            )

        # Validate mask dimensions: (12,) or (1, 12)
        if mask_arr.ndim == 1:
            if mask_arr.shape != (12,):
                raise ValueError(f"Invalid mask shape: expected (12,), got {mask_arr.shape}.")
            mask_arr = mask_arr[np.newaxis, :]
        elif mask_arr.ndim == 2:
            if mask_arr.shape != (1, 12):
                raise ValueError(f"Invalid mask shape: expected (1, 12), got {mask_arr.shape}.")
        else:
            raise ValueError(
                f"Invalid mask dimensions: expected 1D (12,) or 2D (1, 12), "
                f"got {mask_arr.ndim}D shape {mask_arr.shape}."
            )

        # Validate mask values are binary (0 or 1)
        valid_values = np.isin(mask_arr, [0.0, 1.0])
        if not np.all(valid_values):
            raise ValueError("Mask values must be binary indicators (0 for padding, 1 for genuine observation).")

        valid_months = int(mask_arr.sum())
        return seq_arr, mask_arr, valid_months

    def predict_provider(
        self,
        provider_id: str,
        features: Dict[str, Any],
        sequence: Optional[Union[np.ndarray, List[List[float]]]] = None,
        mask: Optional[Union[np.ndarray, List[int]]] = None,
        include_shap: bool = True,
        top_n_shap: int = 5,
    ) -> Dict[str, Any]:
        """
        Score a single healthcare provider for fraud risk across all modalities.
        Requires genuine tabular features, 12-month sequence, and validity mask.
        """
        if not self.is_ready:
            raise RuntimeError(
                f"HealthGuardPredictor: Trained model artifacts are not ready or verification failed: {self.verification_error}"
            )

        # 1. Validate tabular features
        df_row, val_warnings = self.validate_features(features)

        # 2. Validate genuine monthly sequence and validity mask (synthetic fallback removed)
        seq_arr, mask_arr, valid_months = self.validate_sequence_and_mask(sequence, mask)

        # 3. Base model predictions
        p_xgb = float(self.xgboost_detector.predict_proba(df_row)[0])
        p_ann = float(self.ann_detector.predict_proba(df_row.values)[0])
        p_lstm = float(self.lstm_detector.predict_proba(seq_arr, mask_arr)[0])

        # 4. Meta-model Stacking
        base_preds = {
            "xgboost": np.array([p_xgb]),
            "ann": np.array([p_ann]),
            "bilstm": np.array([p_lstm]),
        }
        raw_stacked = float(self.ensemble.predict_proba_raw(base_preds)[0])
        calibrated_prob = float(self.ensemble.predict_proba(base_preds)[0])

        if not (0.0 <= calibrated_prob <= 1.0):
            raise ValueError(f"Calibrated probability {calibrated_prob} is outside [0, 1].")
        if not (0.0 <= raw_stacked <= 1.0):
            raise ValueError(f"Raw stacked probability {raw_stacked} is outside [0, 1].")

        risk_tier = self.ensemble.assign_risk_category(calibrated_prob)

        # 5. Limited history check
        claim_count = float(features.get("total_claims", features.get("total_claim_count", 0)))
        active_m = float(features.get("active_months", valid_months))
        has_limited_history = bool(active_m < 3 or valid_months < 3 or claim_count < 10)
        if has_limited_history:
            val_warnings.append("Limited claim history: prediction confidence may be reduced.")

        # 6. SHAP explanations
        shap_explanation = None
        if include_shap and self.explainer is not None:
            shap_explanation = self.explainer.explain_provider(df_row, top_n=top_n_shap)

        return {
            "provider_id": str(provider_id),
            "fraud_probability": round(calibrated_prob, 4),
            "calibrated_fraud_probability": round(calibrated_prob, 4),
            "raw_stacked_probability": round(raw_stacked, 4),
            "risk_tier": risk_tier,
            "predicted_fraud": int(calibrated_prob >= self.ensemble.risk_thresholds.get("low_medium_cutoff", 0.22)),
            "base_model_probabilities": {
                "xgboost": round(p_xgb, 4),
                "ann": round(p_ann, 4),
                "bilstm": round(p_lstm, 4),
            },
            "risk_thresholds_applied": self.ensemble.risk_thresholds,
            "has_limited_history": has_limited_history,
            "warnings": val_warnings if val_warnings else None,
            "top_risk_factors": shap_explanation.get("top_risk_factors", []) if shap_explanation else [],
            "scope_disclosure": shap_explanation.get("scope_disclosure") if shap_explanation else None,
            "model_version": self.manifest.get("model_version", "1.0.0"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "disclaimer": self.DISCLAIMER,
        }

    def get_provider_data(self, provider_id: str) -> Optional[Tuple[Dict[str, Any], np.ndarray, np.ndarray]]:
        """Look up provider features, sequence, and mask from stored Phase 1 datasets."""
        splits = [
            ("test_provider_features.csv", "test_provider_sequences.npy", "test_provider_masks.npy"),
            ("validation_provider_features.csv", "validation_provider_sequences.npy", "validation_provider_masks.npy"),
            ("train_provider_features.csv", "train_provider_sequences.npy", "train_provider_masks.npy"),
        ]
        for feat_name, seq_name, mask_name in splits:
            f_path = self.proc_dir / feat_name
            if not f_path.exists():
                continue
            df = pd.read_csv(f_path)
            match = df[df["ProviderID"] == provider_id]
            if len(match) > 0:
                idx = match.index[0]
                feat_dict = match.iloc[0].to_dict()
                seq_path = self.proc_dir / seq_name
                mask_path = self.proc_dir / mask_name
                if not (seq_path.exists() and mask_path.exists()):
                    return None
                seq_arr = np.load(seq_path)[idx:idx+1]
                mask_arr = np.load(mask_path)[idx:idx+1]
                return feat_dict, seq_arr, mask_arr
        return None

    def predict_batch(
        self,
        providers_df: pd.DataFrame,
        sequences: np.ndarray,
        masks: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Run batch inference across multiple providers."""
        if not self.is_ready:
            raise RuntimeError(f"HealthGuardPredictor: Trained model artifacts are not ready: {self.verification_error}")

        results = []
        n_rows = len(providers_df)
        pids = providers_df["ProviderID"].values if "ProviderID" in providers_df.columns else [f"PRV_{i}" for i in range(n_rows)]

        for i in range(n_rows):
            row_dict = providers_df.iloc[i].to_dict()
            seq_i = sequences[i:i+1] if sequences is not None else None
            mask_i = masks[i:i+1] if masks is not None else None
            res = self.predict_provider(
                provider_id=str(pids[i]),
                features=row_dict,
                sequence=seq_i,
                mask=mask_i,
                include_shap=False,
            )
            results.append(res)
        return results


# Global singleton instance
_predictor_instance: Optional[HealthGuardPredictor] = None


def get_predictor() -> HealthGuardPredictor:
    global _predictor_instance
    if _predictor_instance is None or not _predictor_instance.is_ready:
        _predictor_instance = HealthGuardPredictor()
    return _predictor_instance
