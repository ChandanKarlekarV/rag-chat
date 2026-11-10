"""Answerers turn (question, retrieved sources) into a streamed, cited answer or `NO_ANSWER`.

* `LLMAnswerer` — Claude (or any `LLMClient`) with the grounded prompt; needs an API key.
* `LocalAnswerer` — the on-device extractive reader; needs no key and quotes the PDF verbatim.
"""

import re
from collections.abc import Iterator
from typing import Protocol

from app.generation.llm_base import LLMClient
from app.generation.prompt import NO_ANSWER_TOKEN, SYSTEM_PROMPT, build_user_prompt
from app.generation.reader import Span, format_answer
from app.schemas import RetrievedChunk

_ASKS_PRICE = re.compile(
    r"\b(cost|costs|price|priced|pricing|how much (?:do|does|is|are|will) .*\b(?:pay|charge))\b", re.I
)
_MONEY = re.compile(
    r"[$€£¥₹]|\b(usd|eur|gbp|inr|dollars?|euros?|pounds?|rupees?|cents?|free of charge|no cost|at no cost)\b", re.I
)
_ASKS_COUNT = re.compile(r"\bhow many\b", re.I)
_NUMBER = re.compile(
    r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|hundred|thousand|million|single|both|no)\b",
    re.I,
)


class Answerer(Protocol):
    """Anything that can answer from retrieved sources."""

    provider: str  # "anthropic" | "local" | "fake"
    model: str

    def stream(self, question: str, sources: list[RetrievedChunk]) -> Iterator[str]:
        """Yield the answer text (with [n] citations) or exactly NO_ANSWER."""
        ...

    def check(self) -> None:
        """Raise LLMError if the answerer cannot be used."""
        ...


class SpanReader(Protocol):
    def best_spans(self, question: str, sources: list[RetrievedChunk]) -> list[Span]: ...


def answer_type_ok(question: str, answer_span: str) -> bool:
    """Reject spans of the wrong type: price questions need money, "how many" needs a number."""
    if _ASKS_PRICE.search(question) and not _MONEY.search(answer_span):
        return False
    return not (_ASKS_COUNT.search(question) and not _NUMBER.search(answer_span))


class LLMAnswerer:
    """Grounded generation with an LLM."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.provider = llm.provider
        self.model = llm.model

    def stream(self, question: str, sources: list[RetrievedChunk]) -> Iterator[str]:
        """Stream the LLM's answer to the numbered-sources prompt."""
        yield from self.llm.stream(SYSTEM_PROMPT, build_user_prompt(question, sources))

    def check(self) -> None:
        """Delegate to the LLM's credential check."""
        self.llm.check()


class LocalAnswerer:
    """Extractive answers from the documents, computed locally.

    Each passage's best span is scored as `reader margin + rerank_weight × (passage rerank score −
    best rerank score)`, so a confident span in a less relevant passage (e.g. the wrong manual)
    does not beat the right passage. The best span is used only if that score reaches `min_score`
    and the span has the type the question asks for.
    """

    provider = "local"

    def __init__(self, reader: SpanReader, model: str, min_score: float, rerank_weight: float = 1.0) -> None:
        self.reader = reader
        self.model = model
        self.min_score = min_score
        self.rerank_weight = rerank_weight

    def choose(self, question: str, sources: list[RetrievedChunk]) -> tuple[Span, float] | None:
        """The span that will be quoted (with its combined score), or None to refuse."""
        spans = self.reader.best_spans(question, sources)
        if not spans:
            return None
        reranks = [s.rerank_score if s.rerank_score is not None else 0.0 for s in sources]
        top = max(reranks)
        scored = sorted(
            ((sp.margin + self.rerank_weight * (reranks[sp.source - 1] - top), sp) for sp in spans),
            key=lambda x: x[0],
            reverse=True,
        )
        for score, sp in scored:
            if score < self.min_score:
                return None
            if answer_type_ok(question, sources[sp.source - 1].chunk.text[sp.start : sp.end]):
                return sp, score
        return None

    def stream(self, question: str, sources: list[RetrievedChunk]) -> Iterator[str]:
        """Quote the best supporting sentence, or NO_ANSWER if the reader isn't confident."""
        chosen = self.choose(question, sources)
        if chosen is None:
            yield NO_ANSWER_TOKEN
            return
        words = format_answer(chosen[0], sources, question).split(" ")
        for i, word in enumerate(words):
            yield word if i == 0 else " " + word

    def check(self) -> None:
        """The local model needs no credentials."""
