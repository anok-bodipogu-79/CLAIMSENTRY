# HealthGuard AI — Code Review & Architecture Guide

HealthGuard AI is an enterprise-grade healthcare insurance fraud detection and forensic claims investigation platform. It combines a multi-modal machine learning ensemble for provider-level risk stratification with an independent deterministic anomaly engine for claim-level billing integrity and compliance surveillance.

---

## 1. System Architecture

```
                                 ┌──────────────────────────────────────────────┐
                                 │   React 18 Frontend ("Liquid Glass" UI)      │
                                 │   - 5 Views: Overview, Providers, Claims,    │
                                 │     Claim Analyzer (Single/Batch), Settings  │
                                 │   - Visual Theme: healthguard_glass.html     │
                                 └──────────────────────┬───────────────────────┘
                                                        │ HTTP / JSON (JWT Auth)
                                                        ▼
                                 ┌──────────────────────────────────────────────┐
                                 │          FastAPI Backend Service             │
                                 │   - REST Endpoints (/api/auth, /api/claims,  │
                                 │     /api/providers, /api/analyzer, /api/csv) │
                                 │   - Background CSV Asynchronous Worker       │
                                 └──────────────┬───────────────────────────────┘
                                                │
                     ┌──────────────────────────┴───────────────────────────┐
                     ▼                                                      ▼
  ┌────────────────────────────────────┐                 ┌────────────────────────────────────┐
  │   PostgreSQL 18 Database           │                 │   Dual-Tier Assessment Engine      │
  │   - Providers, Claims, Beneficiary │                 │                                    │
  │   - ProviderRiskScores, Analyses   │                 │   [Tier 1: Provider ML Ensemble]   │
  │   - AuditLogs, InvestigatorReviews │                 │   - XGBoost (Tabular features)     │
  │   - CSVUploads & Reconciled Rows   │                 │   - PyTorch ANN (Dense embeddings) │
  │   - Alembic Managed Migrations     │                 │   - PyTorch Bi-LSTM (Sequences)    │
  └────────────────────────────────────┘                 │   - Stacking Meta-Learner + Platt  │
                                                         │   - TreeSHAP & Temporal Occlusion  │
                                                         │                                    │
                                                         │   [Tier 2: Claim Anomaly Engine]   │
                                                         │   - Deductible > Reimb Bounds      │
                                                         │   - Inpatient LOS Temporal Clashes │
                                                         │   - Exact & Cross-Claim Duplicates │
                                                         │   - Clinical Coding Syntax (ICD-9) │
                                                         └────────────────────────────────────┘
```

---

## 2. Directory Structure

```
HealthGuard_AI/
├── ML/                                 # Machine Learning core module
│   ├── artifacts/v2/                   # Production model weights & manifests (SHA-256 verified)
│   │   ├── xgboost_model.json          # Trained XGBoost Booster
│   │   ├── ann_model.pt                # PyTorch Artificial Neural Network weights
│   │   ├── lstm_model.pt               # PyTorch Bi-LSTM temporal sequence weights
│   │   ├── stacking_meta_model.joblib  # LogisticRegression Stacking Meta-Learner
│   │   ├── calibrator.joblib           # Platt probability scaling calibrator
│   │   ├── ensemble_config.json        # Ensemble hyperparameters & cutoffs (0.22 / 0.70)
│   │   └── model_manifest.json         # SHA-256 checksum manifest
│   ├── predict.py                      # Production inference engine (HealthGuardPredictor)
│   ├── ensemble.py                     # Multi-modal Stacking Ensemble orchestration
│   ├── claim_anomaly.py                # Individual claim-level rule-based anomaly detector
│   ├── explainability.py               # TreeSHAP explainer & temporal occlusion
│   ├── feature_engineering.py          # 51-feature tabular extractor & 12-month sequence builder
│   ├── preprocessing.py                # Missing imputation, scaling & temporal padding
│   └── train_phase2.py                 # Multi-modal model training & cross-validation script
├── backend/                            # FastAPI backend application
│   ├── app/
│   │   ├── api/                        # Route controllers
│   │   │   ├── auth.py                 # JWT login, registration, me, logout
│   │   │   ├── overview.py             # Executive dashboard KPIs, charts, surveillance
│   │   │   ├── providers.py            # Provider directory, risk scores, forensic detail
│   │   │   ├── claims.py               # Evaluated claims directory, filters, pagination
│   │   │   ├── claim_analyzer.py       # Single-claim ML inference & investigator decisions
│   │   │   ├── csv_upload.py           # Drag-and-drop CSV validation, preview, batch queue
│   │   │   └── audit.py                # Statutory audit logs & forensic event stream
│   │   ├── core/                       # Security, JWT encoding, config, dependencies
│   │   ├── db/                         # SQLAlchemy models, Pydantic schemas, database session
│   │   ├── services/                   # Business logic & ML pipeline orchestrator
│   │   │   ├── claim_analysis_pipeline.py  # Universal pipeline for single & batch claims
│   │   │   ├── anomaly_service.py      # Encapsulates claim_anomaly rule validation
│   │   │   ├── explanation_service.py  # SHAP factor enrichment & narrative formatting
│   │   │   ├── feature_explanation_service.py # Feature dictionary & peer comparison
│   │   │   └── csv_service.py          # CSV parsing, schema checks & preview generation
│   │   └── workers/
│   │       └── csv_worker.py           # Chronological batch claim processor
│   ├── migrations/                     # Alembic database migrations
│   │   └── versions/
│   │       ├── 0001_initial_schema.py
│   │       ├── 0002_add_currency_metadata.py
│   │       └── 0003_xai_investigator_gov.py
│   └── tests/                          # 195 Pytest unit and integration test suite
├── frontend/                           # React 18 + Vite Frontend SPA
│   ├── src/
│   │   ├── App.jsx                     # Complete application UI (5 core views, modals, forms)
│   │   ├── App.test.jsx                # 23 Vitest end-to-end integration tests
│   │   ├── index.css                   # Glassmorphism design system (--bg, --glass, --accent)
│   │   └── main.jsx                    # Application bootstrap
│   ├── index.html                      # Google Fonts (Manrope, Inter, JetBrains Mono)
│   ├── vite.config.js                  # Vite bundler configuration
│   └── package.json                    # Frontend dependencies
├── data/
│   └── processed/                      # Feature specifications and peer reference benchmarks
│       ├── feature_spec.json           # 51-feature specification
│       ├── peer_reference_table.json   # 95th-percentile benchmark distributions
│       └── sequence_metadata.json      # Temporal Bi-LSTM sequence metadata
├── healthguard_glass.html              # Reference design mockup (Liquid Glass design spec)
├── alembic.ini                         # Alembic database migration config
├── requirements.txt                    # Python dependencies
└── .env.example                        # Sanitized configuration template
```

---

## 3. Quick Start & Setup

### 3.1 Prerequisites
- Python 3.11+
- Node.js 18+ & npm
- PostgreSQL 15+ running on `localhost:5432`

### 3.2 Backend Setup
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your local PostgreSQL credentials

# 3. Apply database migrations
python -m alembic upgrade head

# 4. Start FastAPI server
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation will be live at: `http://localhost:8000/docs`

### 3.3 Frontend Setup
```bash
cd frontend

# 1. Install dependencies
npm install

# 2. Run integration test suite
npm test -- --run

# 3. Start development server
npm run dev
```
Application interface will be live at: `http://localhost:5173`

---

## 4. Machine Learning & Explainability Pipeline

### 4.1 Models & Stacking Architecture
1. **XGBoost (Tabular)**: Evaluates 51 aggregate provider features (reimbursement metrics, inpatient/outpatient ratios, chronic condition rates, unique physician and beneficiary ratios).
2. **Artificial Neural Network (ANN)**: 3-layer dense PyTorch model with batch normalization and dropout (`64 -> 32 -> 16 -> 1`) capturing non-linear feature interactions.
3. **Bi-LSTM (Temporal Dynamics)**: 2-layer Bidirectional LSTM taking `(12, 6)` monthly sequential tensors representing volume, total reimbursement, and beneficiary velocity across 12 calendar months.
4. **Stacking Meta-Learner**: Logistic Regression combining out-of-fold predictions from the base models.
5. **Platt Probability Calibrator**: Sigmoid probability scaling calibrating raw ensemble margins into reliable Bayesian probabilities.

### 4.2 Decision Thresholds
- **Low Risk**: Calibrated Probability $< 0.22$
- **Medium Risk**: $0.22 \le \text{Probability} < 0.70$
- **High Risk**: $\text{Probability} \ge 0.70$ (Mandatory clinical review)

---

## 5. Dual-Tier Fraud Assessment & Technical Transparency

HealthGuard AI strictly enforces a **two-tier independent evaluation architecture**:

1. **Tier 1 — Provider-Level ML Fraud Assessment**:
   - Predicts systemic, provider-level fraud likelihood using the multi-modal stacking ensemble.
   - Generates TreeSHAP local feature attribution on the XGBoost component.
   - Analyzes temporal trajectory using Bi-LSTM sequence dynamics.

2. **Tier 2 — Individual Claim-Level Anomaly Detection**:
   - Operates strictly independently of the provider's ML risk score.
   - Evaluates encounter-specific clinical and billing anomalies:
     - Deductible paid exceeding insurance reimbursement.
     - Inpatient admission/discharge date sequence clashes.
     - Excessive inpatient lengths of stay ($\ge 30$ days).
     - Same-day overlapping inpatient stays for the same beneficiary across different providers.
     - Attending, operating, or other physician role conflicts.
     - Duplicate submissions (exact duplicates and conflicting resubmissions).
   - **Key Invariant**: A Low provider-level ML score **never** suppresses claim-level anomalies.

### Known Limitations & Explainability Scope:
- **Claim-Level Detection is Rule-Based**: The individual claim anomaly engine is a deterministic, heuristic rule-based severity index, not a separately trained supervised claim classifier.
- **TreeSHAP Explains XGBoost**: `shap.TreeExplainer` attributes feature importance specifically with respect to the XGBoost component model's margin space. It does not explain the neural network activations or the final meta-learner weights. The platform discloses this scope directly in generated investigator narratives.
- **Statutory Regulatory Disclaimer**: Every report and assessment carries the mandatory notice:
  > *"This is a risk assessment to guide investigation. It is not proof of fraud."*

---

## 6. Verification & Test Suite

The codebase has undergone full automated test verification:
- **Backend**: `195 passed` in `pytest backend/tests`
- **Frontend**: `23 passed` in `vitest run`
- **Production Build**: Successfully compiled via `npm run build --prefix frontend` (zero JSX or CSS compilation errors).
