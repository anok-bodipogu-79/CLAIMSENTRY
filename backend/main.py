"""
HealthGuard AI - FastAPI Application Entrypoint
Exposes the centralized application instance including Phase 2 and Phase 4 endpoints.
"""

from backend.app.main import app

# Re-export app for uvicorn and pytest test client
__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
