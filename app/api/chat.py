"""Chat routes: ask (SSE), answer modes, key test, and conversations."""

import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter, Depends, Header, Response
from fastapi.responses import StreamingResponse

from app.api.deps import API_KEY_HEADER, get_container, get_current_user, require_fetch_header
from app.db.models import User
from app.generation.anthropic import AnthropicProvider
from app.schemas import (
    AskRequest,
    ConversationDetail,
    ConversationOut,
    ModeOut,
    ModesResponse,
    RenameRequest,
    TestKeyResponse,
)
from app.services.container import Container

router = APIRouter(tags=["chat"])
log = logging.getLogger("rag-chat.api")


def _sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/ask", dependencies=[Depends(require_fetch_header)])
def ask(
    body: AskRequest,
    user: User = Depends(get_current_user),
    c: Container = Depends(get_container),
    user_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> StreamingResponse:
    """Stream an answer as SSE events: `mode`, `rewrite`?, `token`*, `citations`, `done` (or `error`)."""
    c.ask_limiter.hit(f"ask:{user.id}", "You're asking very quickly. Please wait a moment and try again.")
    engine = c.engine(user_key, body.mode, user.preferred_mode, user.ollama_model)
    events = c.chat.ask(user, body.question, engine, body.conversation_id, body.doc_ids)  # validates eagerly

    def stream() -> Iterator[str]:
        try:
            for ev in events:
                yield _sse(ev["event"], ev["data"])
        except Exception:
            log.exception("ask failed", extra={"user_id": user.id})
            yield _sse("error", {"code": "internal", "message": "Something went wrong while answering. Please retry."})

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)


@router.get("/chat/modes", response_model=ModesResponse)
def modes(
    user: User = Depends(get_current_user),
    c: Container = Depends(get_container),
    user_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> ModesResponse:
    """Each answer mode with availability and reason, the recommended one, and installed Ollama models."""
    infos = c.router.modes(user_key)
    return ModesResponse(
        modes=[
            ModeOut(mode=m.mode.value, title=m.title, available=m.available, reason=m.reason, description=m.description)
            for m in infos
        ],
        recommended=c.router.recommended(user_key).value,
        default_mode=user.preferred_mode,  # type: ignore[arg-type]
        ollama_models=c.router.ollama_status().models,
        ollama_model=user.ollama_model or c.settings.ollama_model,
    )


@router.post("/chat/test-key", response_model=TestKeyResponse, dependencies=[Depends(require_fetch_header)])
def test_key(
    user: User = Depends(get_current_user),
    c: Container = Depends(get_container),
    user_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> TestKeyResponse:
    """Validate a pasted key (sent in the header, never stored) with the free Models API."""
    key = (user_key or "").strip()
    if not key:
        return TestKeyResponse(ok=False, message="Paste your API key first.")
    ok, message = AnthropicProvider(c.router.api_factory(key)).is_available()
    return TestKeyResponse(ok=ok, message="It works. Claude API mode is ready." if ok else message)


@router.get("/conversations", response_model=list[ConversationOut])
def conversations(
    user: User = Depends(get_current_user), c: Container = Depends(get_container)
) -> list[ConversationOut]:
    """The user's conversation history."""
    return c.chat.list_conversations(user.id)


@router.get("/conversations/{conv_id}", response_model=ConversationDetail)
def conversation(
    conv_id: str, user: User = Depends(get_current_user), c: Container = Depends(get_container)
) -> ConversationDetail:
    """One conversation with its messages."""
    return c.chat.get_conversation(user.id, conv_id)


@router.patch("/conversations/{conv_id}", response_model=ConversationOut, dependencies=[Depends(require_fetch_header)])
def rename(
    conv_id: str, body: RenameRequest, user: User = Depends(get_current_user), c: Container = Depends(get_container)
) -> ConversationOut:
    """Rename a conversation."""
    return c.chat.rename(user.id, conv_id, body.title)


@router.delete("/conversations/{conv_id}", status_code=204, dependencies=[Depends(require_fetch_header)])
def delete_conversation(
    conv_id: str, user: User = Depends(get_current_user), c: Container = Depends(get_container)
) -> Response:
    """Delete a conversation and its messages."""
    c.chat.delete(user.id, conv_id)
    return Response(status_code=204)
