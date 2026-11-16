"""
HealthGuard AI - Shared Production Inference Pipeline
======================================================
Unified prediction service providing deterministic fraud probability scoring,
multi-modal base model breakdown, calibrated stacked ensemble risk tier,
and SHAP explanation factor attribution.
"""

import hashlib
import json
import uuid
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
from ML.claim_anomaly import detect_claim_anomalies, CLAIM_DISCLAIMER
from ML.feature_engineering import build_provider_features, build_monthly_sequences


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
        self.peer_reference_table: Dict[str, Any] = {}
        self._provider_history_cache: Dict[str, pd.DataFrame] = {}
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
            if (self.proc_dir / "peer_reference_table.json").exists():
                with open(self.proc_dir / "peer_reference_table.json", "r", encoding="utf-8") as f:
                    self.peer_reference_table = json.load(f)
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
            "positive_contributors": shap_explanation.get("positive_contributors", []) if shap_explanation else [],
            "negative_contributors": shap_explanation.get("negative_contributors", []) if shap_explanation else [],
            "top_positive_risk_factors": shap_explanation.get("top_positive_risk_factors", []) if shap_explanation else [],
            "top_negative_risk_factors": shap_explanation.get("top_negative_risk_factors", []) if shap_explanation else [],
            "shap_explanation": shap_explanation,
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

    def _load_provider_history(self, provider_id: str) -> pd.DataFrame:
        """
        Retrieve historical claims for a provider from PostgreSQL database and repository datasets.
        """
        matched_chunks = []

        # 1. Check live PostgreSQL database for previously recorded claims
        try:
            from backend.app.db.database import SessionLocal
            from backend.app.db.models import Claim
            db = SessionLocal()
            try:
                db_claims = db.query(Claim).filter(Claim.provider_id == provider_id).all()
                if db_claims:
                    db_rows = []
                    for c in db_claims:
                        diag_1 = c.diagnosis_codes[0] if (c.diagnosis_codes and len(c.diagnosis_codes) > 0) else None
                        proc_1 = c.procedure_codes[0] if (c.procedure_codes and len(c.procedure_codes) > 0) else None
                        db_rows.append({
                            "ClaimID": c.claim_id,
                            "ProviderID": c.provider_id,
                            "BeneID": c.beneficiary_id,
                            "ClaimType": c.claim_type or "Outpatient",
                            "InscClaimAmtReimbursed": float(c.reimbursement_amount or 0.0),
                            "DeductibleAmtPaid": float(c.deductible_amount or 0.0),
                            "ClaimStartDt": c.claim_date.strftime("%Y-%m-%d") if c.claim_date else None,
                            "ClaimEndDt": c.claim_end_date.strftime("%Y-%m-%d") if c.claim_end_date else None,
                            "AdmissionDt": c.admission_date.strftime("%Y-%m-%d") if c.admission_date else None,
                            "DischargeDt": c.discharge_date.strftime("%Y-%m-%d") if c.discharge_date else None,
                            "InpatientLOS": int(c.inpatient_los or 0),
                            "AttendingPhysician": c.attending_physician,
                            "OperatingPhysician": c.operating_physician,
                            "OtherPhysician": c.other_physician,
                            "ClmDiagnosisCode_1": diag_1,
                            "ClmProcedureCode_1": proc_1,
                        })
                    if db_rows:
                        matched_chunks.append(pd.DataFrame(db_rows))
            finally:
                db.close()
        except Exception:
            pass

        # 2. Check repository datasets
        candidate_files = [
            self.proc_dir / "cleaned_training_claims.csv",
            self.proc_dir / "kaggle_test_claims.csv",
        ]
        for c_file in candidate_files:
            if not c_file.exists():
                continue
            try:
                for chunk in pd.read_csv(c_file, chunksize=50000, low_memory=False):
                    sub = chunk[chunk["ProviderID"] == provider_id]
                    if len(sub) > 0:
                        matched_chunks.append(sub)
                if len(matched_chunks) > (1 if matched_chunks and "ClaimID" in matched_chunks[0] else 0):
                    break
            except Exception:
                continue

        if matched_chunks:
            p_history = pd.concat(matched_chunks, ignore_index=True)
            if "ClaimID" in p_history.columns:
                p_history = p_history.drop_duplicates(subset=["ClaimID"], keep="first")
        else:
            p_history = pd.DataFrame()

        return p_history.copy()

    def predict_claims(
        self,
        claims_input: Union[Dict[str, Any], pd.Series, pd.DataFrame, List[Dict[str, Any]], str],
        provider_id: Optional[str] = None,
        existing_claims_df: Optional[pd.DataFrame] = None,
        historical_claims_df: Optional[pd.DataFrame] = None,
        include_shap: bool = True,
        include_temporal: bool = True,
        include_anomalies: bool = True,
        top_n_shap: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Shared prediction pipeline for health insurance claims (single claim or batch CSV).
        Aggregates claim data with provider history, rebuilds 51 tabular features and
        12-month sequence, executes multi-modal stacking ensemble, and runs claim anomaly detection.

        Enforces:
          - Provider-level fraud probability (claims do NOT receive fraud probabilities).
          - Single-claim and CSV consistency.
          - Reconciliation: analysed_claims + rejected_rows == total_uploaded_rows.
          - Mandatory regulatory disclaimer.
        """
        if not self.is_ready:
            raise RuntimeError(
                f"HealthGuardPredictor: Trained model artifacts not ready: {self.verification_error}"
            )

        hist_ref_df = historical_claims_df if historical_claims_df is not None else existing_claims_df

        # 1. Parse claims_input to DataFrame
        if isinstance(claims_input, str):
            p = Path(claims_input)
            if p.exists() and p.is_file():
                df_raw = pd.read_csv(p, low_memory=False)
            else:
                try:
                    parsed_json = json.loads(claims_input)
                    if isinstance(parsed_json, dict):
                        df_raw = pd.DataFrame([parsed_json])
                    elif isinstance(parsed_json, list):
                        df_raw = pd.DataFrame(parsed_json)
                    else:
                        raise ValueError("Invalid JSON format for claims input.")
                except json.JSONDecodeError:
                    import io
                    df_raw = pd.read_csv(io.StringIO(claims_input), low_memory=False)
        elif isinstance(claims_input, dict):
            df_raw = pd.DataFrame([claims_input])
        elif isinstance(claims_input, pd.Series):
            df_raw = pd.DataFrame([claims_input])
        elif isinstance(claims_input, list):
            df_raw = pd.DataFrame(claims_input)
        elif isinstance(claims_input, pd.DataFrame):
            df_raw = claims_input.copy()
        else:
            raise ValueError(f"Unsupported claims input type: {type(claims_input)}")

        total_uploaded_rows = len(df_raw)
        if total_uploaded_rows == 0:
            return {
                "total_uploaded_rows": 0,
                "analysed_claims": 0,
                "rejected_rows": 0,
                "rejected_records": [],
                "rejection_reasons": [],
                "unique_providers": 0,
                "analysed_providers": 0,
                "low_risk_providers": 0,
                "medium_risk_providers": 0,
                "high_risk_providers": 0,
                "claim_anomaly_counts": {},
                "provider_results": {},
                "claims_result": [],
                "disclaimer": self.DISCLAIMER,
            }

        # Normalize column names
        col_map = {
            "claim_id": "ClaimID",
            "claimid": "ClaimID",
            "provider_id": "ProviderID",
            "providerid": "ProviderID",
            "bene_id": "BeneID",
            "beneid": "BeneID",
            "reimbursement": "InscClaimAmtReimbursed",
            "inscclaimamtreimbursed": "InscClaimAmtReimbursed",
            "claimstartdt": "ClaimStartDt",
            "claimenddt": "ClaimEndDt",
            "claimtype": "ClaimType",
        }
        df_raw = df_raw.rename(columns={c: col_map[c.lower()] for c in df_raw.columns if c.lower() in col_map})

        # 2. Row-level validation, duplicate detection, and rejection tracking (Fix 4)
        valid_indices = []
        rejected_records = []
        seen_claim_ids = set()

        for idx, row in df_raw.iterrows():
            cid = row.get("ClaimID", None)
            pid = row.get("ProviderID", None) or provider_id
            reimb = row.get("InscClaimAmtReimbursed", None)
            start_dt = row.get("ClaimStartDt", None)

            reasons = []
            if pd.isna(cid) or cid is None or str(cid).strip() in ["", "nan", "None"]:
                reasons.append("Missing or empty ClaimID")
            else:
                cid_str = str(cid).strip()
                if cid_str in seen_claim_ids:
                    reasons.append(f"Duplicate ClaimID within uploaded batch (ClaimID '{cid_str}' already submitted in earlier row)")
                else:
                    seen_claim_ids.add(cid_str)

            if pd.isna(pid) or pid is None or str(pid).strip() in ["", "nan", "None"]:
                reasons.append("Missing or empty ProviderID")

            if reimb is None or pd.isna(reimb):
                reasons.append("Missing reimbursement amount")
            else:
                try:
                    float(reimb)
                except (ValueError, TypeError):
                    reasons.append("Invalid or non-numeric reimbursement amount")

            if start_dt is None or pd.isna(start_dt) or str(start_dt).strip() in ["", "nan", "None"]:
                reasons.append("Missing ClaimStartDt")
            else:
                try:
                    pd.to_datetime(start_dt)
                except Exception:
                    reasons.append("Unparseable ClaimStartDt")

            if reasons:
                rejected_records.append({
                    "row_index": int(idx),
                    "claim_id": str(cid) if pd.notna(cid) and cid is not None else None,
                    "ClaimID": str(cid) if pd.notna(cid) and cid is not None else None,
                    "provider_id": str(pid) if pd.notna(pid) and pid is not None else None,
                    "ProviderID": str(pid) if pd.notna(pid) and pid is not None else None,
                    "reason": "; ".join(reasons),
                    "rejection_reason": "; ".join(reasons),
                })
            else:
                valid_indices.append(idx)

        valid_claims_df = df_raw.loc[valid_indices].copy()
        if "ProviderID" not in valid_claims_df.columns and provider_id:
            valid_claims_df["ProviderID"] = provider_id

        # Ensure numeric reimbursement
        if "InscClaimAmtReimbursed" in valid_claims_df.columns:
            valid_claims_df["InscClaimAmtReimbursed"] = pd.to_numeric(
                valid_claims_df["InscClaimAmtReimbursed"], errors="coerce"
            ).fillna(0.0)

        # 3. Provider Grouping, Error Handling, and Reconciliation (Fix 5 & Fix 6)
        provider_results = {}
        all_claim_results = []
        successfully_analysed_claim_indices = []

        if not valid_claims_df.empty:
            grouped_providers = valid_claims_df.groupby("ProviderID")
            for pid_val, p_new_claims in grouped_providers:
                pid_str = str(pid_val)

                try:
                    # A. Retrieve historical claims
                    if hist_ref_df is not None:
                        if "ProviderID" in hist_ref_df.columns:
                            p_hist = hist_ref_df[hist_ref_df["ProviderID"] == pid_str].copy()
                        else:
                            p_hist = hist_ref_df.copy()
                    else:
                        p_hist = self._load_provider_history(pid_str)

                    # B. Merge and deduplicate by ClaimID (new claims supersede historical duplicates)
                    if not p_hist.empty:
                        new_claim_ids = set(p_new_claims["ClaimID"].astype(str).tolist())
                        p_hist_dedup = p_hist[~p_hist["ClaimID"].astype(str).isin(new_claim_ids)].copy()
                        p_combined = pd.concat([p_hist_dedup, p_new_claims], ignore_index=True)
                        history_count = len(p_hist_dedup)
                    else:
                        p_hist_dedup = pd.DataFrame()
                        p_combined = p_new_claims.copy()
                        history_count = 0

                    # Ensure clean numerical and categorical types across combined history
                    for num_col in ["InscClaimAmtReimbursed", "DeductibleAmtPaid", "InpatientLOS"]:
                        if num_col in p_combined.columns:
                            p_combined[num_col] = pd.to_numeric(p_combined[num_col], errors="coerce").fillna(0.0)
                        else:
                            p_combined[num_col] = 0.0

                    if "ClaimType" not in p_combined.columns:
                        p_combined["ClaimType"] = "Outpatient"
                    else:
                        p_combined["ClaimType"] = p_combined["ClaimType"].fillna("Outpatient")

                    if "BeneID" not in p_combined.columns:
                        p_combined["BeneID"] = [f"BENE_AUTO_{i}" for i in range(len(p_combined))]
                    else:
                        # Ensure any missing BeneID values are filled so grouping does not drop rows
                        p_combined["BeneID"] = p_combined["BeneID"].fillna("").apply(
                            lambda x: f"BENE_AUTO_{uuid.uuid4().hex[:6]}" if not str(x).strip() else str(x).strip()
                        )

                    # C. Check limited history
                    has_limited_history = bool(len(p_combined) < 10 or history_count == 0)

                    # D. Rebuild tabular provider features (51 features)
                    feat_df = build_provider_features(p_combined, peer_ref=self.peer_reference_table)
                    feat_dict = feat_df.iloc[0].to_dict()

                    # E. Rebuild 12-month temporal sequence and validity mask
                    seqs, masks, plist, meta = build_monthly_sequences(p_combined, pd.Series([pid_str]))

                    # F. Predict provider via multi-modal stacking ensemble
                    p_eval = self.predict_provider(
                        provider_id=pid_str,
                        features=feat_dict,
                        sequence=seqs,
                        mask=masks,
                        include_shap=include_shap,
                        top_n_shap=top_n_shap,
                    )

                    # G. Generate Bi-LSTM temporal explanation if requested
                    temporal_expl = None
                    if include_temporal and self.explainer is not None:
                        try:
                            temporal_expl = self.explainer.explain_temporal_sequence(
                                self.lstm_detector, seqs, masks
                            )
                        except Exception as e:
                            temporal_expl = {"error": f"Temporal explanation failed: {e}"}

                    p_eval["temporal_explanation"] = temporal_expl
                    p_eval["provider_history_count"] = history_count
                    p_eval["new_claims_count"] = len(p_new_claims)
                    p_eval["total_claims_analyzed"] = len(p_combined)
                    if history_count == 0:
                        p_eval["warnings"] = (p_eval.get("warnings") or []) + [
                            "No prior historical claims found for provider in repository; evaluating based exclusively on new claims snapshot."
                        ]

                    provider_results[pid_str] = p_eval

                    # H. Run claim-level anomaly detection on newly uploaded claims using provider history (Fix 5)
                    if include_anomalies:
                        p_claim_anomalies = detect_claim_anomalies(
                            p_new_claims,
                            provider_reference_data=self.peer_reference_table,
                            historical_claims_df=p_hist_dedup,
                        )
                        for ca in p_claim_anomalies:
                            ca["provider_risk_tier"] = p_eval["risk_tier"]
                            ca["provider_fraud_probability"] = p_eval["calibrated_fraud_probability"]
                            all_claim_results.append(ca)

                    # Record successfully analysed rows
                    successfully_analysed_claim_indices.extend(p_new_claims.index.tolist())

                except Exception as e:
                    # Provider analysis failed: reject associated claims and do not count as analysed (Fix 6)
                    for c_idx, c_row in p_new_claims.iterrows():
                        fail_reason = f"Provider analysis failed for {pid_str}: {str(e)}"
                        rejected_records.append({
                            "row_index": int(c_idx),
                            "claim_id": str(c_row.get("ClaimID", f"CLAIM_{c_idx}")),
                            "ClaimID": str(c_row.get("ClaimID", f"CLAIM_{c_idx}")),
                            "provider_id": pid_str,
                            "ProviderID": pid_str,
                            "reason": fail_reason,
                            "rejection_reason": fail_reason,
                        })

        # Final Reconciliation Invariant: analysed_claims + rejected_rows == total_uploaded_rows
        analysed_claims = len(successfully_analysed_claim_indices)
        rejected_rows = len(rejected_records)
        assert analysed_claims + rejected_rows == total_uploaded_rows, (
            f"Reconciliation invariant violated: {analysed_claims} analysed + {rejected_rows} rejected != {total_uploaded_rows} total"
        )

        # 5. Summarize anomaly rule counts
        anomaly_flag_counts = {}
        for ca in all_claim_results:
            for flag in ca.get("anomaly_flags", []):
                anomaly_flag_counts[flag] = anomaly_flag_counts.get(flag, 0) + 1

        low_risk_cnt = sum(1 for p in provider_results.values() if p["risk_tier"] == "LOW")
        med_risk_cnt = sum(1 for p in provider_results.values() if p["risk_tier"] == "MEDIUM")
        high_risk_cnt = sum(1 for p in provider_results.values() if p["risk_tier"] == "HIGH")

        # 6. Structured Return
        is_single_claim = (total_uploaded_rows == 1 and analysed_claims == 1)

        result_dict = {
            "total_uploaded_rows": total_uploaded_rows,
            "analysed_claims": analysed_claims,
            "rejected_rows": rejected_rows,
            "rejected_records": rejected_records,
            "rejection_reasons": rejected_records,
            "unique_providers": len(set(df_raw["ProviderID"].dropna().astype(str).tolist())) if "ProviderID" in df_raw.columns else (1 if provider_id else 0),
            "analysed_providers": len(provider_results),
            "low_risk_providers": low_risk_cnt,
            "medium_risk_providers": med_risk_cnt,
            "high_risk_providers": high_risk_cnt,
            "claim_anomaly_counts": anomaly_flag_counts,
            "provider_results": provider_results,
            "claims_result": all_claim_results,
            "disclaimer": self.DISCLAIMER,
        }

        if is_single_claim:
            first_pid = list(provider_results.keys())[0] if provider_results else None
            result_dict["provider_result"] = provider_results[first_pid] if first_pid else None

        return result_dict


# Global singleton instance
_predictor_instance: Optional[HealthGuardPredictor] = None


def get_predictor() -> HealthGuardPredictor:
    global _predictor_instance
    if _predictor_instance is None or not _predictor_instance.is_ready:
        _predictor_instance = HealthGuardPredictor()
    return _predictor_instance


def predict_claims(
    claims_input: Union[Dict[str, Any], pd.Series, pd.DataFrame, List[Dict[str, Any]], str],
    provider_id: Optional[str] = None,
    existing_claims_df: Optional[pd.DataFrame] = None,
    historical_claims_df: Optional[pd.DataFrame] = None,
    include_shap: bool = True,
    include_temporal: bool = True,
    include_anomalies: bool = True,
    top_n_shap: int = 5,
    **kwargs,
) -> Dict[str, Any]:
    """Shared prediction entry point for single claims and batch CSV uploads."""
    return get_predictor().predict_claims(
        claims_input=claims_input,
        provider_id=provider_id,
        existing_claims_df=existing_claims_df,
        historical_claims_df=historical_claims_df,
        include_shap=include_shap,
        include_temporal=include_temporal,
        include_anomalies=include_anomalies,
        top_n_shap=top_n_shap,
        **kwargs,
    )
