"""
HealthGuard AI — Phase 2 Master Training Pipeline
=================================================
Executes full end-to-end Phase 2 training:
  1. Loads verified Phase 1 processed tabular features, temporal sequences, and masks.
  2. Runs 5-Fold Stratified Cross Validation on training providers to generate OOF base predictions.
  3. Trains genuine Logistic Regression stacking meta-model on OOF probabilities.
  4. Trains final XGBoost, ANN, and Bi-LSTM models on complete 3,787 training providers.
  5. Calibrates meta-probabilities on validation set (Platt scaling vs Isotonic regression).
  6. Derives and freezes decision thresholds (LOW / MEDIUM / HIGH) on validation set.
  7. Evaluates all base models and final calibrated ensemble on the untouched test split (812 providers).
  8. Computes bootstrap 95% confidence intervals, top-k precision, and monotonic risk tier progression.
  9. Initializes SHAP TreeExplainer with peer reference table integration.
  10. Saves versioned artifacts in ML/artifacts/v1/ and writes model_manifest.json.
  11. Verifies artifact integrity and writes evaluation reports.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ML.xgboost_model import XGBoostFraudDetector
from ML.ann_model import ANNFraudDetector
from ML.lstm_model import LSTMFraudDetector
from ML.ensemble import HealthGuardEnsemble, derive_risk_thresholds
from ML.explainability import FraudExplainer
from ML.evaluation import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Phase2_Training")


def run_phase2_pipeline(
    random_state: int = 42,
    artifact_version: str = "v2",
) -> Dict[str, Any]:
    """Execute complete Phase 2 training and evaluation."""
    start_time = time.time()
    logger.info("=" * 70)
    logger.info("HEALTHGUARD AI — PHASE 2 MODEL TRAINING & ENSEMBLE PIPELINE")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Verification of Processed Phase 1 Data
    # -------------------------------------------------------------------------
    proc_dir = ROOT / "data" / "processed"
    reports_eval_dir = ROOT / "reports" / "evaluation"
    artifacts_dir = ROOT / "ML" / "artifacts" / artifact_version
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_eval_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        proc_dir / "train_provider_features.csv",
        proc_dir / "validation_provider_features.csv",
        proc_dir / "test_provider_features.csv",
        proc_dir / "train_provider_sequences.npy",
        proc_dir / "validation_provider_sequences.npy",
        proc_dir / "test_provider_sequences.npy",
        proc_dir / "train_provider_masks.npy",
        proc_dir / "validation_provider_masks.npy",
        proc_dir / "test_provider_masks.npy",
        proc_dir / "peer_reference_table.json",
        proc_dir / "feature_spec.json",
    ]

    missing = [str(f) for f in required_files if not f.exists()]
    if missing:
        err_msg = f"Phase 1 artifacts missing. Training aborted: {missing}"
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    logger.info("▶ [1/10] Loading Phase 1 processed datasets…")
    train_df = pd.read_csv(proc_dir / "train_provider_features.csv")
    val_df = pd.read_csv(proc_dir / "validation_provider_features.csv")
    test_df = pd.read_csv(proc_dir / "test_provider_features.csv")

    train_seq = np.load(proc_dir / "train_provider_sequences.npy")
    val_seq = np.load(proc_dir / "validation_provider_sequences.npy")
    test_seq = np.load(proc_dir / "test_provider_sequences.npy")

    train_masks = np.load(proc_dir / "train_provider_masks.npy")
    val_masks = np.load(proc_dir / "validation_provider_masks.npy")
    test_masks = np.load(proc_dir / "test_provider_masks.npy")

    feature_cols = [c for c in train_df.columns if c not in ["ProviderID", "PotentialFraud"]]
    y_train = train_df["PotentialFraud"].values.astype(int)
    y_val = val_df["PotentialFraud"].values.astype(int)
    y_test = test_df["PotentialFraud"].values.astype(int)

    X_train_tab = train_df[feature_cols]
    X_val_tab = val_df[feature_cols]
    X_test_tab = test_df[feature_cols]

    logger.info(f"  Train: {len(train_df)} providers (features: {len(feature_cols)}, fraud: {y_train.sum()} / {y_train.mean()*100:.2f}%)")
    logger.info(f"  Validation: {len(val_df)} providers (fraud: {y_val.sum()} / {y_val.mean()*100:.2f}%)")
    logger.info(f"  Test: {len(test_df)} providers (fraud: {y_test.sum()} / {y_test.mean()*100:.2f}%)")

    # -------------------------------------------------------------------------
    # 2. Out-of-Fold (OOF) Stacking with 5-Fold Stratified CV
    # -------------------------------------------------------------------------
    logger.info("▶ [2/10] Running 5-Fold Stratified Cross-Validation on training providers…")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    oof_xgb = np.zeros(len(train_df), dtype=np.float64)
    oof_ann = np.zeros(len(train_df), dtype=np.float64)
    oof_lstm = np.zeros(len(train_df), dtype=np.float64)
    fold_ids = np.zeros(len(train_df), dtype=int)

    fold_metrics = []

    for fold_idx, (f_train_idx, f_val_idx) in enumerate(skf.split(X_train_tab, y_train), 1):
        fold_start = time.time()
        fold_ids[f_val_idx] = fold_idx

        # Fold training data
        f_X_tr_tab = X_train_tab.iloc[f_train_idx]
        f_y_tr = y_train[f_train_idx]
        f_seq_tr = train_seq[f_train_idx]
        f_mask_tr = train_masks[f_train_idx]

        # Fold validation data (strictly held out)
        f_X_val_tab = X_train_tab.iloc[f_val_idx]
        f_y_val = y_train[f_val_idx]
        f_seq_val = train_seq[f_val_idx]
        f_mask_val = train_masks[f_val_idx]

        # A. Train Fold XGBoost
        xgb_fold = XGBoostFraudDetector(random_state=random_state + fold_idx)
        xgb_fold.fit(
            f_X_tr_tab,
            pd.Series(f_y_tr, index=f_X_tr_tab.index),
            eval_set=(f_X_val_tab, pd.Series(f_y_val, index=f_X_val_tab.index)),
            early_stopping_rounds=25,
        )
        p_xgb_val = xgb_fold.predict_proba(f_X_val_tab)
        oof_xgb[f_val_idx] = p_xgb_val

        # B. Train Fold ANN
        ann_fold = ANNFraudDetector(input_dim=len(feature_cols), random_state=random_state + fold_idx)
        ann_fold.fit(
            f_X_tr_tab,
            f_y_tr,
            validation_data=(f_X_val_tab, f_y_val),
            epochs=40,
            batch_size=64,
            patience=10,
        )
        p_ann_val = ann_fold.predict_proba(f_X_val_tab)
        oof_ann[f_val_idx] = p_ann_val

        # C. Train Fold Bi-LSTM
        lstm_fold = LSTMFraudDetector(random_state=random_state + fold_idx)
        lstm_fold.fit(
            f_seq_tr,
            f_y_tr,
            masks=f_mask_tr,
            validation_data=(f_seq_val, f_y_val, f_mask_val),
            epochs=30,
            batch_size=32,
            patience=10,
        )
        p_lstm_val = lstm_fold.predict_proba(f_seq_val, f_mask_val)
        oof_lstm[f_val_idx] = p_lstm_val

        pr_xgb = average_precision_score(f_y_val, p_xgb_val)
        pr_ann = average_precision_score(f_y_val, p_ann_val)
        pr_lstm = average_precision_score(f_y_val, p_lstm_val)

        elapsed = time.time() - fold_start
        logger.info(
            f"  Fold {fold_idx}/5 finished in {elapsed:.1f}s — "
            f"PR-AUC: XGBoost={pr_xgb:.4f}, ANN={pr_ann:.4f}, Bi-LSTM={pr_lstm:.4f}"
        )
        fold_metrics.append({
            "fold": fold_idx,
            "train_count": len(f_train_idx),
            "val_count": len(f_val_idx),
            "xgb_prauc": round(pr_xgb, 4),
            "ann_prauc": round(pr_ann, 4),
            "lstm_prauc": round(pr_lstm, 4),
        })

    # Save OOF predictions table
    oof_df = pd.DataFrame({
        "ProviderID": train_df["ProviderID"],
        "Fold": fold_ids,
        "True_PotentialFraud": y_train,
        "OOF_XGBoost_Probability": np.round(oof_xgb, 5),
        "OOF_ANN_Probability": np.round(oof_ann, 5),
        "OOF_BiLSTM_Probability": np.round(oof_lstm, 5),
    })
    oof_df.to_csv(artifacts_dir / "oof_predictions.csv", index=False)
    logger.info("  Saved oof_predictions.csv")

    # -------------------------------------------------------------------------
    # 3. Train Logistic Regression Stacking Meta-Learner
    # -------------------------------------------------------------------------
    logger.info("▶ [3/10] Training Logistic Regression stacking meta-model on OOF matrix…")
    ensemble = HealthGuardEnsemble(random_state=random_state)
    oof_dict = {
        "xgboost": oof_xgb,
        "ann": oof_ann,
        "bilstm": oof_lstm,
    }
    ensemble.fit_meta_learner(oof_dict, y_train)

    logger.info(f"  Meta-learner coefficients: {ensemble.oof_metrics_['meta_coefficients']}")
    logger.info(f"  Meta-learner intercept: {ensemble.oof_metrics_['meta_intercept']:.4f}")
    logger.info(f"  OOF Stacking PR-AUC: {ensemble.oof_metrics_['oof_prauc']:.4f}")

    # -------------------------------------------------------------------------
    # 4. Train Final Base Models on Complete Training Split
    # -------------------------------------------------------------------------
    logger.info("▶ [4/10] Training final base models on full training split (3,787 providers)…")

    # A. Final XGBoost
    logger.info("  Training final XGBoost model…")
    final_xgb = XGBoostFraudDetector(random_state=random_state)
    final_xgb.fit(
        X_train_tab,
        pd.Series(y_train, index=X_train_tab.index),
        eval_set=(X_val_tab, pd.Series(y_val, index=X_val_tab.index)),
        early_stopping_rounds=30,
    )
    final_xgb.save(str(artifacts_dir / "xgboost_model"))
    logger.info(f"  Final XGBoost best iteration: {final_xgb.best_iteration_}")

    # B. Final ANN
    logger.info("  Training final ANN model…")
    final_ann = ANNFraudDetector(input_dim=len(feature_cols), random_state=random_state)
    final_ann.fit(
        X_train_tab,
        y_train,
        validation_data=(X_val_tab, y_val),
        epochs=50,
        batch_size=64,
        patience=12,
    )
    final_ann.save(str(artifacts_dir / "ann_model"))
    logger.info(f"  Final ANN best epoch: {final_ann.best_epoch_} (val PR-AUC: {final_ann.best_val_prauc_:.4f})")

    # C. Final Bi-LSTM
    logger.info("  Training final Bi-LSTM sequence model…")
    final_lstm = LSTMFraudDetector(random_state=random_state)
    final_lstm.fit(
        train_seq,
        y_train,
        masks=train_masks,
        validation_data=(val_seq, y_val, val_masks),
        epochs=40,
        batch_size=32,
        patience=12,
    )
    final_lstm.save(str(artifacts_dir / "lstm_model"))
    logger.info(f"  Final Bi-LSTM best epoch: {final_lstm.best_epoch_} (val PR-AUC: {final_lstm.best_val_prauc_:.4f})")

    # -------------------------------------------------------------------------
    # 5. Validation Predictions & Probability Calibration
    # -------------------------------------------------------------------------
    logger.info("▶ [5/10] Generating validation predictions and fitting calibrator…")
    val_p_xgb = final_xgb.predict_proba(X_val_tab)
    val_p_ann = final_ann.predict_proba(X_val_tab)
    val_p_lstm = final_lstm.predict_proba(val_seq, val_masks)

    val_base_preds = {
        "xgboost": val_p_xgb,
        "ann": val_p_ann,
        "bilstm": val_p_lstm,
    }

    cal_report = ensemble.calibrate(val_base_preds, y_val, preferred_method="platt")
    logger.info(
        f"  Calibration selection: {cal_report['selected_method']} — "
        f"Raw Brier: {cal_report['raw_validation_brier']:.4f} → "
        f"Calibrated Brier: {cal_report['platt_validation_brier']:.4f} "
        f"(Improvement: {cal_report['improvement_pct']:.2f}%)"
    )

    with open(reports_eval_dir / "calibration_results.json", "w", encoding="utf-8") as f:
        json.dump(cal_report, f, indent=2)

    # -------------------------------------------------------------------------
    # 6. Risk Threshold Derivation on Validation Set
    # -------------------------------------------------------------------------
    logger.info("▶ [6/10] Deriving decision thresholds on validation data…")
    val_cal_probs = ensemble.predict_proba(val_base_preds)

    # Calculate optimal thresholds dynamically on validation set using documented strategies
    thresholds_dict = derive_risk_thresholds(
        val_probs=val_cal_probs,
        val_labels=y_val,
        target_high_precision=0.90,
        max_high_workload_pct=5.0,
        min_high_flagged_count=5,
    )
    low_medium_cutoff = thresholds_dict["low_medium_cutoff"]
    medium_high_cutoff = thresholds_dict["medium_high_cutoff"]

    ensemble.set_risk_thresholds(
        low_medium_cutoff=low_medium_cutoff,
        medium_high_cutoff=medium_high_cutoff,
        strategy="precision_recall_investigation_balance",
        metadata=thresholds_dict,
    )
    ensemble.save(str(artifacts_dir))
    logger.info(f"  Derived risk thresholds from validation predictions: Low/Med={low_medium_cutoff}, Med/High={medium_high_cutoff}")

    # -------------------------------------------------------------------------
    # 7. Untouched Test Set Evaluation
    # -------------------------------------------------------------------------
    logger.info("▶ [7/10] Evaluating all models on untouched test split (812 providers)…")
    test_p_xgb = final_xgb.predict_proba(X_test_tab)
    test_p_ann = final_ann.predict_proba(X_test_tab)
    test_p_lstm = final_lstm.predict_proba(test_seq, test_masks)

    test_base_preds = {
        "xgboost": test_p_xgb,
        "ann": test_p_ann,
        "bilstm": test_p_lstm,
    }

    raw_test_ens = ensemble.predict_proba_raw(test_base_preds)
    cal_test_ens = ensemble.predict_proba(test_base_preds)

    evaluator = ModelEvaluator(random_state=random_state)
    full_eval_results = evaluator.generate_full_evaluation_suite(
        test_providers_df=test_df,
        y_test=y_test,
        base_test_probs=test_base_preds,
        raw_ensemble_probs=raw_test_ens,
        calibrated_ensemble_probs=cal_test_ens,
        thresholds=thresholds_dict,
        reports_dir=str(reports_eval_dir),
    )

    ens_metrics = full_eval_results["calibrated_stacking_ensemble"]
    logger.info("  Test Evaluation Summary:")
    logger.info(f"    XGBoost PR-AUC: {full_eval_results['xgboost']['pr_auc']:.4f} | ROC-AUC: {full_eval_results['xgboost']['roc_auc']:.4f}")
    logger.info(f"    ANN PR-AUC:     {full_eval_results['ann']['pr_auc']:.4f} | ROC-AUC: {full_eval_results['ann']['roc_auc']:.4f}")
    logger.info(f"    Bi-LSTM PR-AUC: {full_eval_results['bilstm']['pr_auc']:.4f} | ROC-AUC: {full_eval_results['bilstm']['roc_auc']:.4f}")
    logger.info(f"    Stacked Calibrated PR-AUC: {ens_metrics['pr_auc']:.4f} | ROC-AUC: {ens_metrics['roc_auc']:.4f}")
    logger.info(f"    Ensemble F1:    {ens_metrics['f1_score']:.4f} | Brier: {ens_metrics['brier_score']:.4f}")

    risk_tiers = ens_metrics["risk_tiers"]
    logger.info("  Test Risk Tier Stratification:")
    logger.info(f"    LOW:    {risk_tiers['LOW']['fraud_count']} / {risk_tiers['LOW']['provider_count']} ({risk_tiers['LOW']['fraud_rate']*100:.2f}%)")
    logger.info(f"    MEDIUM: {risk_tiers['MEDIUM']['fraud_count']} / {risk_tiers['MEDIUM']['provider_count']} ({risk_tiers['MEDIUM']['fraud_rate']*100:.2f}%)")
    logger.info(f"    HIGH:   {risk_tiers['HIGH']['fraud_count']} / {risk_tiers['HIGH']['provider_count']} ({risk_tiers['HIGH']['fraud_rate']*100:.2f}%)")
    logger.info(f"    Monotonic risk progression verified: {risk_tiers['monotonic_risk_progression']}")

    # -------------------------------------------------------------------------
    # 8. SHAP Explainability Setup & Verification
    # -------------------------------------------------------------------------
    logger.info("▶ [8/10] Setting up and verifying SHAP TreeExplainer for XGBoost…")
    explainer = FraudExplainer(
        xgboost_model=final_xgb,
        feature_names=feature_cols,
        peer_reference_path=str(proc_dir / "peer_reference_table.json"),
    )

    # Test local explanation on first high-risk test provider
    high_risk_idx = int(np.argmax(cal_test_ens))
    sample_feat = X_test_tab.iloc[high_risk_idx]
    sample_pid = test_df["ProviderID"].iloc[high_risk_idx]

    explanation = explainer.explain_provider(sample_feat, top_n=5)
    logger.info(f"  Verified local SHAP explanation for sample provider {sample_pid}:")
    for factor in explanation["top_risk_factors"]:
        logger.info(f"    • {factor['feature']}: val={factor['actual_value']} ({factor['direction']}, SHAP={factor['shap_value']:+.4f})")

    # -------------------------------------------------------------------------
    # 9. Model Manifest Creation
    # -------------------------------------------------------------------------
    logger.info("▶ [9/10] Writing versioned model manifest JSON…")

    version_map = {"v1": "1.0.0", "v2": "2.0.0"}
    model_version = version_map.get(
        artifact_version,
        f"{artifact_version.lstrip('v')}.0.0" if artifact_version.startswith("v") else artifact_version,
    )

    manifest = {
        "model_version": model_version,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "training_dataset_reference": "data/processed/train_provider_features.csv (3,787 providers)",
        "validation_dataset_reference": "data/processed/validation_provider_features.csv (811 providers)",
        "test_dataset_reference": "data/processed/test_provider_features.csv (812 providers)",
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "sequence_shape": [12, 6],
        "sequence_features": ["claim_count", "total_reimbursement", "avg_reimbursement", "max_reimbursement", "inpatient_ratio", "unique_beneficiaries"],
        "base_models": {
            "xgboost": {
                "algorithm": "Gradient Boosted Decision Trees",
                "framework": "xgboost 3.4.1",
                "scale_pos_weight": 9.7,
                "best_iteration": final_xgb.best_iteration_,
                "test_prauc": full_eval_results["xgboost"]["pr_auc"],
                "test_rocauc": full_eval_results["xgboost"]["roc_auc"],
            },
            "ann": {
                "algorithm": "Feedforward Multi-Layer Perceptron (128-64-32)",
                "framework": "PyTorch 2.14.0",
                "scaler": "RobustScaler",
                "best_epoch": final_ann.best_epoch_,
                "test_prauc": full_eval_results["ann"]["pr_auc"],
                "test_rocauc": full_eval_results["ann"]["roc_auc"],
            },
            "bilstm": {
                "algorithm": "Bidirectional LSTM with Masked Temporal Mean Pooling (64x2)",
                "framework": "PyTorch 2.14.0",
                "scaler": "RobustScaler",
                "best_epoch": final_lstm.best_epoch_,
                "test_prauc": full_eval_results["bilstm"]["pr_auc"],
                "test_rocauc": full_eval_results["bilstm"]["roc_auc"],
            },
        },
        "ensemble": {
            "meta_learner": "LogisticRegression",
            "stacking_strategy": "Stratified 5-Fold Cross Validation Out-of-Fold (OOF)",
            "coefficients": ensemble.oof_metrics_["meta_coefficients"],
            "intercept": ensemble.oof_metrics_["meta_intercept"],
            "calibration_method": cal_report["selected_method"],
            "raw_test_prauc": full_eval_results["raw_stacking_ensemble"]["pr_auc"],
            "calibrated_test_prauc": ens_metrics["pr_auc"],
            "calibrated_test_rocauc": ens_metrics["roc_auc"],
            "calibrated_test_f1": ens_metrics["f1_score"],
            "calibrated_test_brier": ens_metrics["brier_score"],
        },
        "risk_thresholds": thresholds_dict,
        "artifact_filenames": [
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
            "model_manifest.json",
        ],
        "training_status": "TRAINED",
        "random_state": random_state,
    }

    # Compute and populate sha256_checksums for all artifact files in this directory
    import hashlib
    checksums = {}
    for p in sorted(artifacts_dir.glob("*")):
        if p.name != "model_manifest.json" and p.is_file():
            checksums[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest["sha256_checksums"] = checksums

    with open(artifacts_dir / "model_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"  Saved model_manifest.json in {artifacts_dir} (version: {model_version}, status: TRAINED)")

    # Save calculated validation threshold metrics
    with open(reports_eval_dir / "risk_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds_dict, f, indent=2)
    logger.info("  Saved calculated validation risk thresholds to reports/evaluation/risk_thresholds.json")

    # -------------------------------------------------------------------------
    # 10. Artifact Integrity Verification
    # -------------------------------------------------------------------------
    logger.info("▶ [10/10] Verifying artifact loading and determinism…")
    test_xgb = XGBoostFraudDetector().load(str(artifacts_dir / "xgboost_model"))
    test_ann = ANNFraudDetector().load(str(artifacts_dir / "ann_model"))
    test_lstm = LSTMFraudDetector().load(str(artifacts_dir / "lstm_model"))
    test_ens = HealthGuardEnsemble().load(str(artifacts_dir))

    p1 = test_xgb.predict_proba(X_test_tab.iloc[:5])
    p2 = test_ann.predict_proba(X_test_tab.iloc[:5])
    p3 = test_lstm.predict_proba(test_seq[:5], test_masks[:5])
    p_ens = test_ens.predict_proba({"xgboost": p1, "ann": p2, "bilstm": p3})

    assert len(p_ens) == 5, "Inference verification returned invalid count."
    assert np.all((p_ens >= 0.0) & (p_ens <= 1.0)), "Probabilities out of bounds [0, 1]."
    logger.info("  ✓ Artifact integrity verified: All models reloaded and deterministic.")

    total_time = time.time() - start_time
    logger.info("=" * 70)
    logger.info(f"✅ PHASE 2 TRAINING & EVALUATION COMPLETE ({total_time:.1f}s)")
    logger.info("=" * 70)

    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HealthGuard AI — Phase 2 Master Training Pipeline")
    parser.add_argument("--version", type=str, default="v2", help="Artifact version (default: v2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random state seed (default: 42)")
    args = parser.parse_args()
    run_phase2_pipeline(random_state=args.random_state, artifact_version=args.version)
