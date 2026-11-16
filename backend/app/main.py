"""
HealthGuard AI - FastAPI Application Entrypoint
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path regardless of working directory
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.db.database import init_db
from backend.app.api.auth import router as auth_router
from backend.app.api.overview import router as overview_router
from backend.app.api.providers import router as providers_router
from backend.app.api.claims import router as claims_router
from backend.app.api.claim_analyzer import router as claim_analyzer_router
from backend.app.api.csv_upload import router as csv_upload_router
from backend.app.api.settings import router as settings_router
from backend.routes import router as v1_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables safely
    init_db()
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Explainable Ensemble Framework for Health Insurance Provider Fraud Detection and Claims Investigation",
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Phase 4 Core Routers under /api
app.include_router(auth_router, prefix=settings.API_PREFIX)
app.include_router(overview_router, prefix=settings.API_PREFIX)
app.include_router(providers_router, prefix=settings.API_PREFIX)
app.include_router(claims_router, prefix=settings.API_PREFIX)
app.include_router(claim_analyzer_router, prefix=settings.API_PREFIX)
app.include_router(csv_upload_router, prefix=settings.API_PREFIX)
app.include_router(settings_router, prefix=settings.API_PREFIX)

# Mount Phase 2 router at /api/v1 for backward compatibility with Phase 1 & 2 test suites
app.include_router(v1_router)


@app.get("/health", status_code=200)
def health_check():
    """Application and database health check."""
    from backend.app.db.database import engine
    from ML.predict import get_predictor
    predictor = get_predictor()
    dialect = engine.dialect.name
    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "database": f"PostgreSQL 18 ({dialect})" if dialect == "postgresql" else f"SQLite ({dialect})",
        "models_ready": predictor.is_ready,
        "model_version": predictor.manifest.get("model_version", settings.MODEL_VERSION) if predictor.is_ready else None,
        "disclaimer": settings.REGULATORY_DISCLAIMER,
    }


@app.get("/api/models/readiness")
def get_models_readiness():
    """Alias for Phase 2 readiness check at /api/models/readiness."""
    from ML.predict import get_predictor
    predictor = get_predictor()
    return {
        "ready": predictor.is_ready,
        "model_version": predictor.manifest.get("model_version", settings.MODEL_VERSION) if predictor.is_ready else None,
        "artifacts_validated": predictor.is_ready,
    }


@app.get("/")
def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "model_version": settings.MODEL_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
        "disclaimer": settings.REGULATORY_DISCLAIMER,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
