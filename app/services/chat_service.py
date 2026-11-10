"""Ask flow and conversation history."""

import json
import logging
import time
from collections.abc import Iterator
from typing import Any

from app.db.models import Conversation, Message, User
from app.db.repositories import ConversationRepo, DocumentRepo
from app.db.session import Database
from app.errors import AppError, NotFound
from app.generation.llm_base import LLMClient, LLMError
from app.generation.prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
    cited_numbers,
    could_become_no_answer,
    is_no_answer,
)
from app.schemas import Citation, ConversationDetail, ConversationOut, MessageOut, RetrievedChunk
from app.services.retrieval_service import RetrievalService, passes_gate

log = logging.getLogger("rag-chat.chat")

FALLBACK_MESSAGE = "I couldn't find this in your documents."
NO_KEY_MESSAGE = "Add your Anthropic API key in Settings to get written answers. Search already works."
NO_READY_DOCS = "Upload a PDF to start. Questions are answered from your documents only."
SNIPPET_CHARS = 300
TITLE_CHARS = 60

Event = dict[str, Any]


def to_citation(n: int, r: RetrievedChunk) -> Citation:
    """Build the citation payload for source number `n`."""
    text = r.chunk.text
    snippet = text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"
    return Citation(
        n=n, doc_id=r.chunk.doc_id, filename=r.chunk.filename, page_start=r.chunk.page_start,
        page_end=r.chunk.page_end, snippet=snippet, text=text, vector_score=r.vector_score,
        bm25_score=r.bm25_score, rrf_score=r.rrf_score, rerank_score=r.rerank_score,
    )


def extract_citations(answer: str, sources: list[RetrievedChunk]) -> list[Citation]:
    """Citations for the sources the answer actually cited (out-of-range numbers are ignored)."""
    return [to_citation(n, sources[n - 1]) for n in sorted(cited_numbers(answer)) if 1 <= n <= len(sources)]


def _event(name: str, data: dict[str, Any]) -> Event:
    return {"event": name, "data": data}


class ChatService:
    """Runs questions through retrieval → gate → LLM and stores the conversation."""

    def __init__(self, db: Database, retrieval: RetrievalService, min_rerank_score: float) -> None:
        self.db = db
        self.retrieval = retrieval
        self.min_rerank_score = min_rerank_score

    # ---- ask ----

    def ask(
        self,
        user: User,
        question: str,
        llm: LLMClient | None,
        conversation_id: str | None = None,
        doc_ids: list[str] | None = None,
    ) -> Iterator[Event]:
        """Validate the request up front (raising AppError), then return the SSE event stream."""
        question = question.strip()
        if not question:
            raise AppError("empty_question", "Type a question first.")
        with self.db.session() as s:
            ready = set(DocumentRepo(s).ready_ids(user.id))
            if conversation_id and ConversationRepo(s).get(user.id, conversation_id) is None:
                raise NotFound("not_found", "This conversation no longer exists. Start a new one.")
        if not ready:
            raise AppError("no_documents", NO_READY_DOCS)
        scope = None
        if doc_ids:
            scope = [d for d in doc_ids if d in ready]
            if not scope:
                raise AppError("no_documents", "The selected documents aren't ready yet. Pick a ready document.")
        return self._stream(user, question, llm, conversation_id, scope)

    def _stream(
        self, user: User, question: str, llm: LLMClient | None, conversation_id: str | None, scope: list[str] | None
    ) -> Iterator[Event]:
        result = self.retrieval.search(user.id, question, scope)
        sources = result.chunks
        retrieved = [to_citation(i, s).model_dump() for i, s in enumerate(sources, start=1)]

        if not passes_gate(sources, self.min_rerank_score):
            yield from self._finish(user, question, conversation_id, FALLBACK_MESSAGE, [], False, retrieved)
            return
        if llm is None:
            yield _event("error", {"code": "no_api_key", "message": NO_KEY_MESSAGE})
            return

        full, emitted = "", 0
        started = time.perf_counter()
        try:
            for delta in llm.stream(SYSTEM_PROMPT, build_user_prompt(question, sources)):
                full += delta
                if could_become_no_answer(full) or is_no_answer(full):
                    continue  # never let the sentinel reach the screen
                yield _event("token", {"text": full[emitted:]})
                emitted = len(full)
        except LLMError as exc:
            yield _event("error", {"code": "llm_error", "message": str(exc)})
            return
        log.info("llm", extra={"user_id": user.id, "latency_ms": round((time.perf_counter() - started) * 1000)})

        if is_no_answer(full) or not full.strip():
            yield from self._finish(user, question, conversation_id, FALLBACK_MESSAGE, [], False, retrieved)
            return
        if emitted < len(full):
            yield _event("token", {"text": full[emitted:]})
        answer = full.strip()
        yield from self._finish(user, question, conversation_id, answer, extract_citations(answer, sources), True, retrieved)

    def _finish(
        self,
        user: User,
        question: str,
        conversation_id: str | None,
        answer: str,
        citations: list[Citation],
        answerable: bool,
        retrieved: list[dict[str, Any]],
    ) -> Iterator[Event]:
        yield _event("citations", {"citations": [c.model_dump() for c in citations], "retrieved": retrieved})
        with self.db.session() as s:
            repo = ConversationRepo(s)
            conv = repo.get(user.id, conversation_id) if conversation_id else None
            if conv is None:
                conv = repo.add(Conversation(user_id=user.id, title=_title(question)))
            repo.add_message(Message(conversation_id=conv.id, role="user", content=question))
            msg = repo.add_message(
                Message(
                    conversation_id=conv.id, role="assistant", content=answer, answerable=answerable,
                    citations_json=json.dumps([c.model_dump() for c in citations]),
                )
            )
            conv_id, msg_id = conv.id, msg.id
        yield _event(
            "done", {"answer": answer, "answerable": answerable, "conversation_id": conv_id, "message_id": msg_id}
        )

    # ---- history ----

    def list_conversations(self, user_id: str) -> list[ConversationOut]:
        """The user's conversations, newest first."""
        with self.db.session() as s:
            return [ConversationOut.model_validate(c) for c in ConversationRepo(s).for_user(user_id)]

    def get_conversation(self, user_id: str, conv_id: str) -> ConversationDetail:
        """One conversation with its messages (404 if it isn't the user's)."""
        with self.db.session() as s:
            conv = ConversationRepo(s).get(user_id, conv_id)
            if conv is None:
                raise NotFound("not_found", "This conversation no longer exists.")
            messages = [
                MessageOut(
                    id=m.id, role=m.role, content=m.content, answerable=m.answerable,  # type: ignore[arg-type]
                    citations=[Citation(**c) for c in json.loads(m.citations_json or "[]")],
                    created_at=m.created_at,
                )
                for m in conv.messages
            ]
            return ConversationDetail(id=conv.id, title=conv.title, created_at=conv.created_at, messages=messages)

    def rename(self, user_id: str, conv_id: str, title: str) -> ConversationOut:
        """Rename one of the user's conversations."""
        with self.db.session() as s:
            conv = ConversationRepo(s).get(user_id, conv_id)
            if conv is None:
                raise NotFound("not_found", "This conversation no longer exists.")
            conv.title = title.strip()[:120] or conv.title
            return ConversationOut.model_validate(conv)

    def delete(self, user_id: str, conv_id: str) -> None:
        """Delete one of the user's conversations."""
        with self.db.session() as s:
            repo = ConversationRepo(s)
            conv = repo.get(user_id, conv_id)
            if conv is None:
                raise NotFound("not_found", "This conversation no longer exists.")
            repo.delete(conv)


def _title(question: str) -> str:
    q = " ".join(question.split())
    return q if len(q) <= TITLE_CHARS else q[: TITLE_CHARS - 1].rsplit(" ", 1)[0] + "…"
