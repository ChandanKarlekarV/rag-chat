"""Ask flow and conversation history.

The service works with any `AnswerProvider` (it never imports a concrete one): rewrite → retrieval
→ shared no-answer gate → provider → citations → persist. Every mode emits the same SSE events:
`mode`, `rewrite`?, `token`*, `citations`, `done` (or `error`).
"""

import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from app.db.models import Conversation, Message, User
from app.db.repositories import ConversationRepo, DocumentRepo
from app.db.session import Database
from app.errors import AppError, NotFound
from app.generation.base import AnswerProvider, confidence_words
from app.generation.llm_base import LLMError
from app.generation.prompt import cited_numbers, could_become_no_answer, is_no_answer
from app.generation.question_type import classify
from app.generation.rewrite import HeuristicRewriter, QueryRewriter, Turn
from app.schemas import Citation, ConversationDetail, ConversationOut, MessageOut, RetrievedChunk
from app.services.retrieval_service import RetrievalService, passes_gate

log = logging.getLogger("rag-chat.chat")

FALLBACK_MESSAGE = "I couldn't find this in your documents."
NO_READY_DOCS = "Upload a PDF to start. Questions are answered from your documents only."
SNIPPET_CHARS = 300
TITLE_CHARS = 60

Event = dict[str, Any]


@dataclass(frozen=True)
class Engine:
    """What will answer this request (built by the router in the API layer)."""

    mode: str
    provider: AnswerProvider | None
    rewriter: QueryRewriter
    error_code: str | None = None
    error_message: str | None = None


def to_citation(n: int, r: RetrievedChunk, highlight: list[int] | None = None) -> Citation:
    """Build the citation payload for source number `n`."""
    text = r.chunk.text
    snippet = text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"
    return Citation(
        n=n,
        doc_id=r.chunk.doc_id,
        filename=r.chunk.filename,
        page_start=r.chunk.page_start,
        page_end=r.chunk.page_end,
        snippet=snippet,
        text=text,
        highlight=highlight,
        vector_score=r.vector_score,
        bm25_score=r.bm25_score,
        rrf_score=r.rrf_score,
        rerank_score=r.rerank_score,
    )


def extract_citations(
    answer: str, sources: list[RetrievedChunk], span_source: int | None = None, span_range: list[int] | None = None
) -> list[Citation]:
    """Citations for the sources the answer actually cited; the span's source carries its highlight."""
    return [
        to_citation(n, sources[n - 1], span_range if n == span_source else None)
        for n in sorted(cited_numbers(answer))
        if 1 <= n <= len(sources)
    ]


def _event(name: str, data: dict[str, Any]) -> Event:
    return {"event": name, "data": data}


class ChatService:
    """Runs questions through rewrite → retrieval → gate → provider and stores the conversation."""

    def __init__(
        self, db: Database, retrieval: RetrievalService, min_rerank_score: float, history_turns: int = 3
    ) -> None:
        self.db = db
        self.retrieval = retrieval
        self.min_rerank_score = min_rerank_score
        self.history_turns = history_turns

    # ---- ask ----

    def ask(
        self,
        user: User,
        question: str,
        engine: Engine,
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
        history = self._history(user.id, conversation_id) if conversation_id else []
        return self._stream(user, question, engine, conversation_id, scope, history)

    def _history(self, user_id: str, conversation_id: str) -> list[Turn]:
        """The last few (question, answer) pairs of a conversation, oldest first."""
        with self.db.session() as s:
            conv = ConversationRepo(s).get(user_id, conversation_id)
            msgs = list(conv.messages) if conv else []
        turns = [
            Turn(q.content, a.content)
            for q, a in zip(msgs[::2], msgs[1::2], strict=False)
            if q.role == "user" and a.role == "assistant"
        ]
        return turns[-self.history_turns :]

    def _stream(
        self,
        user: User,
        question: str,
        engine: Engine,
        conversation_id: str | None,
        scope: list[str] | None,
        history: list[Turn],
    ) -> Iterator[Event]:
        provider = engine.provider
        label = provider.label if provider else ""
        yield _event("mode", {"mode": engine.mode, "label": label})
        rw = (engine.rewriter or HeuristicRewriter()).rewrite(question, history)
        if rw.rewritten:
            yield _event("rewrite", {"search_query": rw.search_query, "question": rw.question})
        sources = self.retrieval.search(user.id, rw.search_query, scope).chunks
        retrieved = [to_citation(i, s).model_dump() for i, s in enumerate(sources, start=1)]
        finish = {"user": user, "question": question, "conversation_id": conversation_id, "retrieved": retrieved}

        if not passes_gate(sources, self.min_rerank_score):  # same gate for every mode
            yield from self._finish(**finish, answer=FALLBACK_MESSAGE, citations=[], answerable=False, mode=engine.mode)
            return
        if provider is None:
            yield _event("error", {"code": engine.error_code, "message": engine.error_message, "mode": engine.mode})
            return

        full, emitted, meta = "", 0, {}
        started = time.perf_counter()
        try:
            for ev in provider.answer(rw.question, sources, classify(rw.question)):
                if ev.kind == "meta":
                    meta |= ev.data
                    continue
                full += ev.text
                if could_become_no_answer(full) or is_no_answer(full):
                    continue  # never let the sentinel reach the screen
                yield _event("token", {"text": full[emitted:]})
                emitted = len(full)
        except LLMError as exc:
            code = "ollama_error" if engine.mode == "ollama" else "llm_error"
            yield _event("error", {"code": code, "message": str(exc), "mode": engine.mode})
            return
        log.info(
            "answer",
            extra={
                "user_id": user.id,
                "mode": engine.mode,
                "rewritten": rw.rewritten,
                "latency_ms": round((time.perf_counter() - started) * 1000),
            },
        )

        if is_no_answer(full) or not full.strip():
            yield from self._finish(
                **finish, answer=FALLBACK_MESSAGE, citations=[], answerable=False, mode=engine.mode, label=label
            )
            return
        if emitted < len(full):
            yield _event("token", {"text": full[emitted:]})
        answer = full.strip()
        citations = extract_citations(answer, sources, meta.get("span_source"), meta.get("span_char_range"))
        yield from self._finish(
            **finish, answer=answer, citations=citations, answerable=True, mode=engine.mode, label=label, meta=meta
        )

    def _finish(
        self,
        user: User,
        question: str,
        conversation_id: str | None,
        retrieved: list[dict[str, Any]],
        answer: str,
        citations: list[Citation],
        answerable: bool,
        mode: str,
        label: str = "",
        meta: dict[str, Any] | None = None,
    ) -> Iterator[Event]:
        meta = meta or {}
        yield _event("citations", {"citations": [c.model_dump() for c in citations], "retrieved": retrieved})
        with self.db.session() as s:
            repo = ConversationRepo(s)
            conv = repo.get(user.id, conversation_id) if conversation_id else None
            if conv is None:
                conv = repo.add(Conversation(user_id=user.id, title=_title(question)))
            repo.add_message(Message(conversation_id=conv.id, role="user", content=question))
            msg = repo.add_message(
                Message(
                    conversation_id=conv.id,
                    role="assistant",
                    content=answer,
                    answerable=answerable,
                    citations_json=json.dumps([c.model_dump() for c in citations]),
                    mode=mode,
                    label=label or None,
                )
            )
            conv_id, msg_id = conv.id, msg.id
        yield _event(
            "done",
            {
                "answer": answer,
                "answerable": answerable,
                "conversation_id": conv_id,
                "message_id": msg_id,
                "mode": mode,
                "label": label,
                "span": meta.get("span"),
                "confidence": meta.get("confidence"),
                "confidence_label": confidence_words(meta.get("confidence")),
                "hint": meta.get("hint"),
            },
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
                    id=m.id,
                    role=m.role,  # type: ignore[arg-type]
                    content=m.content,
                    answerable=m.answerable,
                    citations=[Citation(**c) for c in json.loads(m.citations_json or "[]")],
                    mode=m.mode,
                    label=m.label,
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
