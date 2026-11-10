"""Shared FastAPI dependencies: container access, current user, CSRF header, client IP."""

from fastapi import Depends, Request

from app.db.models import User
from app.errors import AppError
from app.services.container import Container

SESSION_COOKIE = "rag_session"
CSRF_HEADER = "X-Requested-With"
CSRF_VALUE = "fetch"
API_KEY_HEADER = "X-Anthropic-Api-Key"


def get_container(request: Request) -> Container:
    """The service container created at startup."""
    return request.app.state.container


def get_current_user(request: Request, c: Container = Depends(get_container)) -> User:
    """Resolve the signed-in user from the httpOnly session cookie (401 otherwise)."""
    return c.auth.user_from_token(request.cookies.get(SESSION_COOKIE))


def require_fetch_header(request: Request) -> None:
    """CSRF defence for state-changing requests: SameSite=Lax cookie + a custom header.

    Browsers cannot attach custom headers to cross-site form posts, and CORS is not enabled.
    """
    if request.headers.get(CSRF_HEADER, "").lower() != CSRF_VALUE:
        raise AppError("csrf", "This request was blocked for your safety. Reload the page and try again.", 403)


def client_ip(request: Request) -> str:
    """Best-effort client address for rate limiting."""
    return request.client.host if request.client else "unknown"
