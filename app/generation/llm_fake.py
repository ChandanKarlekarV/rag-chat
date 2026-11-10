"""Deterministic `LLMClient` for tests and offline demos (`LLM_PROVIDER=fake`).

With a scripted `answer` it replays that text. Otherwise it is *extractive*: it answers with the
one or two sentences of source [1] that share the most words with the question, and cites [1],
so the full UI flow works without any API key. It never invents text.
"""

import re
from collections.abc import Iterator

from app.generation.llm_base import LLMError

_SOURCE_1 = re.compile(r"\[1\] \([^)]*\)\n(.+?)(?:\n\n\[2\]|\n</sources>)", re.S)
_QUESTION = re.compile(r"QUESTION: (.+)$", re.S)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    ]
)


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


class FakeLLM:
    """Offline stand-in for a real model."""

    provider = "fake"
    model = "extractive-fake"

    def __init__(
        self, answer: str | None = None, chunk_size: int = 4, valid: bool = True, completion: str | None = None
    ) -> None:
        self.answer = answer
        self.completion = completion
        self.chunk_size = chunk_size
        self.valid = valid
        self.calls: list[tuple[str, str]] = []

    def _extractive(self, user: str) -> str:
        source = _SOURCE_1.search(user)
        if not source:
            return "NO_ANSWER"
        q = _QUESTION.search(user)
        wanted = _words(q.group(1)) if q else set()
        sentences = [s for s in _SENTENCE.split(source.group(1).strip()) if s]
        ranked = sorted(range(len(sentences)), key=lambda i: (-len(wanted & _words(sentences[i])), i))
        best = sorted(ranked[:2])
        return " ".join(sentences[i] for i in best) + " [1]"

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Yield the scripted or extractive answer in small deltas."""
        self.calls.append((system, user))
        text = self.answer if self.answer is not None else self._extractive(user)
        for i in range(0, len(text), self.chunk_size):
            yield text[i : i + self.chunk_size]

    def complete(self, system: str, user: str, max_tokens: int = 512) -> str:
        """Return the scripted completion, or echo the latest question for rewrites."""
        self.calls.append((system, user))
        if self.completion is not None:
            return self.completion
        match = re.search(r"Latest question: (.+)$", user, re.S)
        return match.group(1).strip() if match else ""

    def check(self) -> None:
        """Succeeds unless constructed with `valid=False`."""
        if not self.valid:
            raise LLMError("This API key was rejected by Anthropic. Check that you copied it completely.")
