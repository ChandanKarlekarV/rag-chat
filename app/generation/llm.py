"""Provider-agnostic streaming LLM interface with an Anthropic implementation."""

from collections.abc import Iterator
from typing import Protocol

import anthropic


class LLMError(RuntimeError):
    """Raised when the LLM call fails. The message is safe to show to users."""


class LLM(Protocol):
    """Anything that can stream a completion for a system prompt + user prompt."""

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        """Yield text deltas as they are generated."""
        ...


# Models that accept the server-side refusal fallback (`fallbacks: "default"`).
_FALLBACK_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-fable-5-2"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLM:
    """Streams answers from the Anthropic Messages API."""

    def __init__(self, api_key: str, model: str, max_tokens: int = 2048, timeout_s: float = 60.0) -> None:
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s, max_retries=2)
        self.model = model
        self.max_tokens = max_tokens

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        """Stream text deltas. Low effort keeps latency down; grounded QA needs little reasoning."""
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": "low"},
        }
        if self.model in _FALLBACK_MODELS:
            kwargs |= {"betas": [_FALLBACK_BETA], "fallbacks": "default"}
        try:
            with self._client.beta.messages.stream(**kwargs) as stream:
                yield from stream.text_stream
                final = stream.get_final_message()
        except anthropic.APITimeoutError as exc:
            raise LLMError("The language model timed out. Please try again.") from exc
        except anthropic.AuthenticationError as exc:
            raise LLMError("The Anthropic API key is missing or invalid.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("The language model is rate-limited right now. Please retry shortly.") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the language model API.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"The language model returned an error ({exc.status_code}).") from exc
        if final.stop_reason == "refusal":
            raise LLMError("The language model declined to answer this question.")
