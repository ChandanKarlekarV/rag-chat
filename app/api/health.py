"""Liveness / readiness."""

from fastapi import APIRouter, Depends
from sqlalchemy import text

from app.api.deps import get_container
from app.schemas import HealthResponse
from app.services.container import Container

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(c: Container = Depends(get_container)) -> HealthResponse:
    """Checks the database and that models are loaded."""
    try:
        with c.db.session() as s:
            s.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    ok = db_ok and c.models_loaded
    return HealthResponse(
        status="ok" if ok else "degraded",
        database=db_ok,
        models_loaded=c.models_loaded,
        llm_configured=c.server_llm is not None,
        llm_provider=c.settings.llm_provider,
        llm_model=c.settings.llm_model,
    )
