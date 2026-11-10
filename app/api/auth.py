"""Auth routes: register, login, guest, logout, me."""

from fastapi import APIRouter, Depends, Request, Response

from app.api.deps import SESSION_COOKIE, client_ip, get_container, get_current_user, require_fetch_header
from app.db.models import User
from app.errors import AppError
from app.schemas import Credentials, PreferencesIn, UserOut
from app.services.container import Container

router = APIRouter(prefix="/auth", tags=["auth"])
RATE_MESSAGE = "Too many attempts. Please wait a minute and try again."


def _set_session(response: Response, c: Container, user: User) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        c.auth.issue_token(user.id),
        max_age=c.settings.session_days * 86400,
        httponly=True,
        samesite="lax",
        secure=c.settings.is_prod,
        path="/",
    )


@router.post("/register", response_model=UserOut, status_code=201, dependencies=[Depends(require_fetch_header)])
def register(body: Credentials, request: Request, response: Response, c: Container = Depends(get_container)) -> User:
    """Create an account and sign in."""
    c.auth_limiter.hit(f"auth:{client_ip(request)}", RATE_MESSAGE)
    user = c.auth.register(body.email, body.password)
    _set_session(response, c, user)
    return user


@router.post("/login", response_model=UserOut, dependencies=[Depends(require_fetch_header)])
def login(body: Credentials, request: Request, response: Response, c: Container = Depends(get_container)) -> User:
    """Sign in with email and password."""
    c.auth_limiter.hit(f"auth:{client_ip(request)}", RATE_MESSAGE)
    user = c.auth.login(body.email, body.password)
    _set_session(response, c, user)
    return user


@router.post("/guest", response_model=UserOut, status_code=201, dependencies=[Depends(require_fetch_header)])
def guest(request: Request, response: Response, c: Container = Depends(get_container)) -> User:
    """Create a guest account with the sample PDF already indexed."""
    c.auth_limiter.hit(f"auth:{client_ip(request)}", RATE_MESSAGE)
    user = c.auth.create_guest()
    _set_session(response, c, user)
    return user


@router.post("/logout", status_code=204, dependencies=[Depends(require_fetch_header)])
def logout(response: Response, request: Request, c: Container = Depends(get_container)) -> Response:
    """Sign out. Guest accounts are deleted immediately."""
    try:
        user = c.auth.user_from_token(request.cookies.get(SESSION_COOKIE))
        if user.is_guest:
            c.auth.delete_user(user.id)
    except AppError:  # signing out must always succeed, even with a stale cookie
        pass
    response.status_code = 204
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> User:
    """The signed-in user (with their answer-mode preferences)."""
    return user


me_router = APIRouter(prefix="/me", tags=["auth"])


@me_router.put("/preferences", response_model=UserOut, dependencies=[Depends(require_fetch_header)])
def preferences(
    body: PreferencesIn, user: User = Depends(get_current_user), c: Container = Depends(get_container)
) -> User:
    """Save the default answer mode and Ollama model. Never the API key."""
    return c.auth.set_preferences(user.id, body.default_mode, body.ollama_model)
