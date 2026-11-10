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
    redis_ok: bool | None = None
    if c.redis is not None:
        try:
            redis_ok = bool(c.redis.ping())
        except Exception:
            redis_ok = False
    ok = db_ok and c.models_loaded and redis_ok is not False
    return HealthResponse(
        status="ok" if ok else "degraded",
        database=db_ok,
        redis=redis_ok,
        models_loaded=c.models_loaded,
        local_model=c.router.local.name if c.router.local else None,
        server_key=bool(c.settings.anthropic_api_key),
    )
