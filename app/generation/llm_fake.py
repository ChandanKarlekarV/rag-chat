"""Deterministic `LLMClient` for tests and offline demos (`LLM_PROVIDER=fake`).

With a scripted `answer` it replays that text. Otherwise it is *extractive*: it answers with the
first sentences of source [1] and cites it, so the full UI flow works without any API key.
"""

import re
from collections.abc import Iterator

from app.generation.llm_base import LLMError

_SOURCE_1 = re.compile(r"\[1\] \([^)]*\)\n(.+?)(?:\n\n\[2\]|\n</sources>)", re.S)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


class FakeLLM:
    """Offline stand-in for a real model."""

    provider = "fake"
    model = "extractive-fake"

    def __init__(self, answer: str | None = None, chunk_size: int = 4, valid: bool = True) -> None:
        self.answer = answer
        self.chunk_size = chunk_size
        self.valid = valid
        self.calls: list[tuple[str, str]] = []

    def _extractive(self, user: str) -> str:
        match = _SOURCE_1.search(user)
        if not match:
            return "NO_ANSWER"
        sentences = _SENTENCE.split(match.group(1).strip())
        return " ".join(sentences[:2]) + " [1]"

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Yield the scripted or extractive answer in small deltas."""
        self.calls.append((system, user))
        text = self.answer if self.answer is not None else self._extractive(user)
        for i in range(0, len(text), self.chunk_size):
            yield text[i : i + self.chunk_size]

    def check(self) -> None:
        """Succeeds unless constructed with `valid=False`."""
        if not self.valid:
            raise LLMError("This API key was rejected by Anthropic. Check that you copied it completely.")
