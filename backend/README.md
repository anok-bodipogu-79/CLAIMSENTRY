# HealthGuard AI — Phase 4 Backend & Database Service

Welcome to the backend service of **HealthGuard AI: An Explainable Ensemble Framework for Health Insurance Provider Fraud Detection and Claims Investigation**.

This service is built with **FastAPI**, **SQLAlchemy ORM**, **Alembic**, and **Pydantic V2**. It exposes RESTful APIs for real-time single-claim analysis, asynchronous CSV batch claims processing, dynamic investigation dashboards, explainability retrieval (SHAP feature contributions and Bi-LSTM temporal shifts), model version switching, and role-based access control (RBAC).

---

## 1. Architecture Overview

```
backend/
├── alembic.ini                   # Alembic database migration configuration
├── alembic/                      # Database migration scripts & environment
│   ├── env.py                    # Schema metadata & connection setup
│   └── versions/
│       └── 0001_initial_schema.py# 12-table initial schema migration
├── app/
│   ├── api/                      # REST API Routers
│   │   ├── auth.py               # /api/auth (Register, Login, /me)
│   │   ├── overview.py           # /api/overview (Aggregated stats, risk distribution)
│   │   ├── providers.py          # /api/providers (Listing, detail, claims, scores)
│   │   ├── claims.py             # /api/claims (Listing, filtering, details)
│   │   ├── claim_analyzer.py     # /api/analyzer/single (Live claim inference)
│   │   ├── csv_upload.py         # /api/analyzer/csv (Batch upload, status, rejections)
│   │   └── settings.py           # /api/settings (System config & version switching)
│   ├── core/
│   │   ├── config.py             # Pydantic BaseSettings & configuration
│   │   ├── dependencies.py       # JWT validation & role security dependencies
│   │   └── security.py           # Password hashing (bcrypt) & JWT token handling
│   ├── db/
│   │   ├── database.py           # Engine & SessionLocal with automatic SQLite fallback
│   │   ├── models.py             # 12 SQLAlchemy ORM models
│   │   └── schemas.py            # Pydantic request/response schemas
│   ├── services/
│   │   ├── prediction_service.py # Phase 3 ML predictor integration & persistence
│   │   ├── anomaly_service.py    # Unsupervised claim isolation & rule flagging
│   │   ├── explanation_service.py# SHAP explanations & temporal narrative formatting
│   │   └── csv_service.py        # Asynchronous CSV validation & queue management
│   ├── workers/
│   │   └── csv_worker.py         # Background worker enforcing reconciliation invariant
│   ├── utils/
│   │   └── data_importer.py      # Seed utility importing Phase 1 Kaggle samples
│   └── main.py                   # FastAPI application factory & middleware setup
├── main.py                       # Root backend entrypoint re-exporting `app`
├── requirements.txt              # Production and test Python dependencies
├── .env.example                  # Environment configuration template
└── tests/                        # 28 comprehensive automated test suites
    ├── conftest.py               # Test client & transactional in-memory fixtures
    ├── test_auth.py              # User registration, JWT login, & role access
    ├── test_db.py                # Table schemas, foreign keys, rollback isolation
    ├── test_ml_integration.py    # ML predictor readiness, scoring, version switching
    ├── test_providers_claims.py  # Provider & claim querying, filtering, pagination
    ├── test_claim_analyzer.py    # Single-claim real-time inference & explainability
    └── test_csv_upload.py        # CSV batch processing & reconciliation invariant
```

---

## 2. Database Schema (12 Tables)

The database schema is managed via Alembic and supports both PostgreSQL (production) and SQLite (embedded development / testing).

| Table Name | Description | Key Relationships |
| :--- | :--- | :--- |
| `users` | System accounts (Investigator, Admin) with bcrypt hashes | One-to-Many `case_notes`, `audit_logs` |
| `model_versions` | Registry of ML ensemble versions (`v1`, `v2`) | One-to-Many `provider_risk_scores` |
| `providers` | Healthcare providers under monitoring | One-to-Many `claims`, `provider_risk_scores`, `case_notes` |
| `beneficiaries` | Patient beneficiaries (HIPAA-compliant, zero protected attributes) | One-to-Many `claims` |
| `claims` | Individual medical claims (Inpatient / Outpatient) | Many-to-One `providers`, `beneficiaries`; One-to-One `claim_analyses` |
| `provider_risk_scores` | Provider evaluation snapshots (stacked probability, risk tier) | Many-to-One `providers`, `model_versions`; One-to-Many `explanations` |
| `claim_analyses` | Unsupervised claim anomaly scoring & flag breakdown | One-to-One `claims`; Many-to-One `provider_risk_scores` |
| `explanations` | SHAP feature contributions and temporal shift narratives | Many-to-One `provider_risk_scores`, `claim_analyses` |
| `csv_uploads` | Batch upload tracking records (status, total/successful/rejected) | One-to-Many `csv_upload_rows` |
| `csv_upload_rows` | Itemized row processing status and specific rejection reasons | Many-to-One `csv_uploads` |
| `case_notes` | Investigator case logs, flagged status, and follow-up notes | Many-to-One `providers`, `users` |
| `audit_logs` | Immutable audit trail of system events, logins, and overrides | Many-to-One `users` |

---

## 3. Key Invariants & Features

1. **Reconciliation Invariant**:
   For any batch upload processed via `/api/analyzer/csv`:
   $$\text{successful\_rows} + \text{rejected\_rows} = \text{total\_rows}$$
   Duplicate `ClaimID`s (either within the batch or pre-existing in the database) are itemized and rejected with informative error messages.
2. **Statutory Regulatory Disclaimer**:
   Every risk score and explanation returned includes:
   `"This is a risk assessment to guide investigation. It is not proof of fraud."`
3. **No Fabricated / Hardcoded Statistics**:
   The `/api/overview` dashboard dynamically aggregates actual database counts, risk tier distributions, and financial metrics.
4. **Dynamic Model Version Switching**:
   Admins can switch between `v1` and `v2` via `POST /api/settings/model-version`. Each version validates its 13 artifact SHA-256 checksums before serving predictions.

---

## 4. Setup and Run Guide

### 4.1 Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Optional: PostgreSQL 14+ (if PostgreSQL is unavailable, SQLite is used automatically)

### 4.2 Installation
```bash
# Activate virtual environment (if applicable)
pip install -r backend/requirements.txt
```

### 4.3 Database Migration & Seeding
```bash
# Run database migrations
alembic -c backend/alembic.ini upgrade head

# Seed database with sample Kaggle data and model versions
python -m backend.app.utils.data_importer
```

### 4.4 Start the FastAPI Server
```bash
# Run using uvicorn
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 5. Automated Test Suite

Run the full backend test suite:
```bash
python -m pytest backend/tests/ -v
```

Run the complete 110-test project regression suite:
```bash
python -m pytest ML/test_phase1.py ML/test_phase2.py ML/test_phase3.py backend/tests/ -v
```
