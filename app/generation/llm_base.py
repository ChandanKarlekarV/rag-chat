"""Provider-agnostic LLM interface."""

from collections.abc import Iterator
from typing import Protocol


class LLMError(RuntimeError):
    """Raised when the LLM call fails. The message is safe to show to users."""


class LLMClient(Protocol):
    """Anything that can stream a grounded answer and make short completions."""

    provider: str
    model: str

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Yield text deltas as they are generated."""
        ...

    def complete(self, system: str, user: str, max_tokens: int = 512, json_mode: bool = False) -> str:
        """Return a non-streamed completion (query rewriting, judging, study material).

        `json_mode` asks for a JSON object where the backend can enforce it (Ollama); others rely
        on the prompt, and callers always parse defensively.
        """
        ...

    def check(self) -> None:
        """Raise LLMError if the credentials or model are unusable."""
        ...

    def list_models(self) -> list[str]:
        """Chat model ids the credentials can use (raises LLMError if they're rejected)."""
        ...
