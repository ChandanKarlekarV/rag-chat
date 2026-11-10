"""Provider-agnostic LLM interface."""

from collections.abc import Iterator
from typing import Protocol


class LLMError(RuntimeError):
    """Raised when the LLM call fails. The message is safe to show to users."""


class LLMClient(Protocol):
    """Anything that can stream a grounded answer."""

    provider: str
    model: str

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Yield text deltas as they are generated."""
        ...

    def check(self) -> None:
        """Raise LLMError if the credentials or model are unusable."""
        ...
