"""Answer orchestration: the no-answer gate, NO_ANSWER detection and citation extraction."""

import re
from collections.abc import Iterator
from typing import Any

from app.generation.llm import LLM, LLMError
from app.generation.prompt import NO_ANSWER_TOKEN, SYSTEM_PROMPT, build_user_prompt
from app.schemas import AskResponse, Citation, RetrievedChunk

FALLBACK_MESSAGE = (
    "I couldn't find this in the uploaded documents. Try rephrasing the question, "
    "or upload a document that covers it."
)
NO_DOCS_MESSAGE = "No documents are indexed yet. Upload a PDF to get started."
_CITE_GROUP = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
SNIPPET_CHARS = 300


def passes_gate(results: list[RetrievedChunk], min_score: float) -> bool:
    """True when the best reranked chunk is relevant enough to be worth sending to the LLM."""
    scores = [r.rerank_score for r in results if r.rerank_score is not None]
    return bool(scores) and max(scores) >= min_score


def is_no_answer(text: str) -> bool:
    """True when the model signalled that the sources are insufficient."""
    return text.strip().strip(".").upper().startswith(NO_ANSWER_TOKEN)


def cited_numbers(answer: str) -> list[int]:
    """All source numbers referenced as [n], [n][m] or [n, m], in order of first appearance."""
    seen: dict[int, None] = {}
    for group in _CITE_GROUP.findall(answer):
        for n in group.split(","):
            seen.setdefault(int(n), None)
    return list(seen)


def to_citation(n: int, r: RetrievedChunk) -> Citation:
    """Build the citation payload for source number `n`."""
    text = r.chunk.text
    snippet = text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"
    return Citation(
        n=n,
        filename=r.chunk.filename,
        page_start=r.chunk.page_start,
        page_end=r.chunk.page_end,
        snippet=snippet,
        text=text,
        vector_score=r.vector_score,
        bm25_score=r.bm25_score,
        rrf_score=r.rrf_score,
        rerank_score=r.rerank_score,
    )


def extract_citations(answer: str, sources: list[RetrievedChunk]) -> list[Citation]:
    """Citations for the sources the answer actually cited (out-of-range numbers are ignored)."""
    return [to_citation(n, sources[n - 1]) for n in sorted(cited_numbers(answer)) if 1 <= n <= len(sources)]


def _event(name: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"event": name, "data": data}


def answer_stream(
    question: str, sources: list[RetrievedChunk], llm: LLM | None, min_score: float
) -> Iterator[dict[str, Any]]:
    """Yield `token` events while the answer streams, then one `done` (or `error`) event.

    The `done` payload is an `AskResponse` plus `retrieved` (all sources with scores, for debugging).
    Output is held back until it can no longer be the start of NO_ANSWER, so the sentinel never
    reaches the user.
    """
    retrieved = [to_citation(i, s).model_dump() for i, s in enumerate(sources, start=1)]

    def done(answer: str, citations: list[Citation], answerable: bool) -> dict[str, Any]:
        payload = AskResponse(answer=answer, citations=citations, answerable=answerable).model_dump()
        return _event("done", payload | {"retrieved": retrieved})

    if not passes_gate(sources, min_score):
        yield done(FALLBACK_MESSAGE, [], False)
        return
    if llm is None:
        yield _event(
            "error",
            {"message": "No API key yet. Click “API key” in the sidebar and paste your Anthropic key."},
        )
        return

    full = ""
    emitted = 0
    try:
        for delta in llm.stream(SYSTEM_PROMPT, build_user_prompt(question, sources)):
            full += delta
            head = full.lstrip()
            if len(head) < len(NO_ANSWER_TOKEN) and NO_ANSWER_TOKEN.startswith(head.upper()):
                continue  # might still turn into NO_ANSWER
            if is_no_answer(full):
                continue
            yield _event("token", {"text": full[emitted:]})
            emitted = len(full)
    except LLMError as exc:
        yield _event("error", {"message": str(exc)})
        return

    if is_no_answer(full) or not full.strip():
        yield done(FALLBACK_MESSAGE, [], False)
        return
    if emitted < len(full):
        yield _event("token", {"text": full[emitted:]})
    yield done(full.strip(), extract_citations(full, sources), True)
