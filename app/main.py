"""FastAPI app factory: routers, error shape, request ids, static UI and startup (models load once)."""

import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api import auth, chat, documents, health
from app.config import DEV_SECRET, get_settings
from app.errors import AppError
from app.logging_setup import configure_logging, request_id_var
from app.services.container import Container, build_container

log = logging.getLogger("rag-chat")
STATIC_DIR = Path(__file__).parent / "static"

_HTTP_MESSAGES = {
    404: ("not_found", "We couldn't find that."),
    405: ("method_not_allowed", "That action isn't supported here."),
    413: ("too_large", "That file is too large."),
}


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


def _validation_message(exc: RequestValidationError) -> str:
    fields = {str(e["loc"][-1]) for e in exc.errors() if e.get("loc")}
    if "question" in fields:
        return "Type a question (up to 2,000 characters)."
    if fields & {"email", "password"}:
        return "Enter your email and password."
    if "file" in fields:
        return "Choose a PDF file to upload."
    return "Some of the information sent was missing or invalid. Please check and try again."


def create_app(container: Container | None = None) -> FastAPI:
    """Application factory. Tests inject a container built with fakes."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = get_settings()
        weak = settings.secret_key == DEV_SECRET or "change-me" in settings.secret_key or len(settings.secret_key) < 32
        if container is None and settings.is_prod and weak:
            raise RuntimeError("Set a random SECRET_KEY of 32+ characters before running with APP_ENV=prod.")
        app.state.container = container or build_container(settings)
        app.state.container.auth.cleanup_guests()
        log.info("rag-chat ready")
        yield

    app = FastAPI(title="rag-chat", version="2.0.0", description="Chat with your PDFs.", lifespan=lifespan)

    @app.middleware("http")
    async def request_id(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        return response

    @app.exception_handler(AppError)
    async def app_error(_: Request, exc: AppError) -> JSONResponse:
        return _error(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(422, "invalid_request", _validation_message(exc))

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code, message = _HTTP_MESSAGES.get(exc.status_code, ("http_error", "Something went wrong. Please try again."))
        return _error(exc.status_code, code, message)

    @app.exception_handler(Exception)
    async def unexpected(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return _error(500, "internal", "Something went wrong on our side. Please try again.")

    for router in (auth.router, documents.router, chat.router, health.router):
        app.include_router(router)

    @app.get("/", include_in_schema=False)
    def index_page() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


configure_logging()
app = create_app()
