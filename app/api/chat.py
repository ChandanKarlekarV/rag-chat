"""Chat routes: ask (SSE), conversations, and the API-key check."""

import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter, Depends, Header, Response
from fastapi.responses import StreamingResponse

from app.api.deps import API_KEY_HEADER, get_container, get_current_user, require_fetch_header
from app.db.models import User
from app.generation.llm_base import LLMError
from app.schemas import AskRequest, ConversationDetail, ConversationOut, LLMCheckResponse, RenameRequest
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
    """Stream an answer as SSE events: `token`*, `citations`, `done` (or `error`)."""
    c.ask_limiter.hit(f"ask:{user.id}", "You're asking very quickly. Please wait a moment and try again.")
    llm, _ = c.resolve_llm(user_key)
    events = c.chat.ask(user, body.question, llm, body.conversation_id, body.doc_ids)  # validates eagerly

    def stream() -> Iterator[str]:
        try:
            for ev in events:
                yield _sse(ev["event"], ev["data"])
        except Exception:
            log.exception("ask failed", extra={"user_id": user.id})
            yield _sse("error", {"code": "internal", "message": "Something went wrong while answering. Please retry."})

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(stream(), media_type="text/event-stream", headers=headers)


@router.get("/conversations", response_model=list[ConversationOut])
def conversations(user: User = Depends(get_current_user), c: Container = Depends(get_container)) -> list[ConversationOut]:
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


@router.post("/llm/check", response_model=LLMCheckResponse, dependencies=[Depends(require_fetch_header)])
def llm_check(
    user: User = Depends(get_current_user),
    c: Container = Depends(get_container),
    user_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> LLMCheckResponse:
    """Does the browser's key (or else the server's) work? Uses the free Models API."""
    llm, source = c.resolve_llm(user_key)
    provider, model = c.settings.llm_provider, c.settings.llm_model
    if llm is None:
        return LLMCheckResponse(ok=False, source="none", provider=provider, model=model, message="No API key added yet.")
    try:
        llm.check()
    except LLMError as exc:
        return LLMCheckResponse(ok=False, source=source, provider=provider, model=model, message=str(exc))  # type: ignore[arg-type]
    label = "offline demo model" if llm.provider == "fake" else model
    return LLMCheckResponse(ok=True, source=source, provider=provider, model=llm.model, message=f"Connected to {label}.")  # type: ignore[arg-type]
