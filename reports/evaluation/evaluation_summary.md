# HealthGuard AI — Phase 2 Model Evaluation Summary

## 1. Overview
This report documents the performance of the three base models (XGBoost, ANN, Bi-LSTM) and the Out-of-Fold (OOF) Stacking Ensemble on the untouched test split (812 healthcare providers).

## 2. Test Set Model Comparison Table

| Model                        |   PR-AUC |   ROC-AUC |   F1-Score |   Precision |   Recall |   Brier-Score |   Threshold |
|:-----------------------------|---------:|----------:|-----------:|------------:|---------:|--------------:|------------:|
| xgboost                      |   0.6861 |    0.9446 |     0.5614 |      0.4211 |   0.8421 |        0.0929 |        0.5  |
| ann                          |   0.7031 |    0.9398 |     0.484  |      0.3317 |   0.8947 |        0.1144 |        0.5  |
| bilstm                       |   0.7301 |    0.9427 |     0.5036 |      0.3465 |   0.9211 |        0.1118 |        0.5  |
| raw_stacking_ensemble        |   0.7212 |    0.9433 |     0.6103 |      0.4745 |   0.8553 |        0.0481 |        0.22 |
| calibrated_stacking_ensemble |   0.7212 |    0.9433 |     0.6474 |      0.5773 |   0.7368 |        0.0472 |        0.22 |

## 3. Calibrated Ensemble Performance Highlights
- **Primary Metric (PR-AUC):** 0.7212 (95% CI: [0.6267, 0.8148])
- **Secondary Metric (ROC-AUC):** 0.9433 (95% CI: [0.9142, 0.9676])
- **F1-Score:** 0.6474
- **Brier Score:** 0.0472
- **Prevalence in Test Set:** 9.36% (76 fraud / 812 total)

## 4. Risk Tier Stratification
- **Low / Medium Cutoff:** 0.2200
- **Medium / High Cutoff:** 0.7000

| Risk Tier | Provider Count | Pct of Test Set | Actual Fraud Count | Fraud Rate |
| :--- | :--- | :--- | :--- | :--- |
| **LOW** | 715 | 88.05% | 20 | 2.80% |
| **MEDIUM** | 83 | 10.22% | 42 | 50.60% |
| **HIGH** | 14 | 1.72% | 14 | 100.00% |

**Monotonic Progression:** `VERIFIED` (Fraud rate increases monotonically across Low -> Medium -> High risk categories).

## 5. Top-K Precision (Investigation Utility)
| Top-K | Precision | Fraud Providers Identified |
| :--- | :--- | :--- |
| TOP_10 | 100.00% | 10 / 10 |
| TOP_25 | 88.00% | 22 / 25 |
| TOP_50 | 74.00% | 37 / 50 |
| TOP_100 | 57.00% | 57 / 100 |

## 6. Regulatory & Ethical Disclaimer
> **NOTICE:** This ensemble output represents an audit prioritization risk score to guide investigation. It is not legal proof of fraud. High-risk providers should be reviewed by qualified human clinical auditors.
