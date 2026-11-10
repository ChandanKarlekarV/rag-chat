"""Test doubles: `FakeLLM` (an `LLMClient`) and `FakeProvider` (an `AnswerProvider`)."""

import re
from collections.abc import Iterator

from app.generation.base import AnswerEvent, AnswerMode, QuestionType
from app.generation.generative import GenerativeProvider
from app.generation.llm_base import LLMError
from app.schemas import RetrievedChunk


class FakeLLM:
    """Replays a scripted answer in small deltas and records every call."""

    provider = "fake"
    model = "fake-model"

    def __init__(
        self, answer: str = "NO_ANSWER", chunk_size: int = 4, valid: bool = True, completion: str | None = None
    ) -> None:
        self.answer = answer
        self.completion = completion
        self.chunk_size = chunk_size
        self.valid = valid
        self.calls: list[tuple[str, str]] = []

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Yield the scripted answer in deltas."""
        self.calls.append((system, user))
        for i in range(0, len(self.answer), self.chunk_size):
            yield self.answer[i : i + self.chunk_size]

    def complete(self, system: str, user: str, max_tokens: int = 512) -> str:
        """Scripted completion, or echo the latest question (an identity rewrite)."""
        self.calls.append((system, user))
        if self.completion is not None:
            return self.completion
        match = re.search(r"Latest question: (.+)$", user, re.S)
        return match.group(1).strip() if match else ""

    def check(self) -> None:
        """Succeeds unless constructed with `valid=False`."""
        if not self.valid:
            raise LLMError("This API key was rejected by Anthropic. Check that you copied it completely.")


class FakeProvider(GenerativeProvider):
    """A generative provider over `FakeLLM`, for contract tests of any mode."""

    def __init__(self, llm: FakeLLM | None = None, mode: AnswerMode = AnswerMode.API) -> None:
        super().__init__(llm or FakeLLM(), mode)

    def answer(self, question: str, chunks: list[RetrievedChunk], qtype: QuestionType) -> Iterator[AnswerEvent]:
        """Same as the real generative providers."""
        yield from super().answer(question, chunks, qtype)
