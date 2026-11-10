"""Anthropic implementation of `LLMClient`.

SDK usage verified against the installed `anthropic` 2.7.0 (signature of
`client.beta.messages.stream` includes `betas`, `fallbacks`, `output_config`) and the bundled
Claude API reference; the model id is from platform.claude.com/docs models overview.
"""

from collections.abc import Iterator
from typing import Any

import anthropic

from app.generation.llm_base import LLMError

# Models that accept the server-side refusal fallback (`fallbacks: "default"`).
_FALLBACK_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-fable-5-2"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicLLM:
    """Streams answers from the Anthropic Messages API."""

    provider = "anthropic"

    def __init__(self, api_key: str, model: str, max_tokens: int = 2048, timeout_s: float = 60.0) -> None:
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s, max_retries=2)
        self.model = model
        self.max_tokens = max_tokens

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Stream text deltas. Low effort keeps latency down; grounded QA needs little reasoning."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": "low"},
        }
        if self.model in _FALLBACK_MODELS:
            kwargs |= {"betas": [_FALLBACK_BETA], "fallbacks": "default"}
        try:
            with self._client.beta.messages.stream(**kwargs) as stream:
                yield from stream.text_stream
                final = stream.get_final_message()
        except anthropic.APITimeoutError as exc:
            raise LLMError("The language model took too long to answer. Please try again.") from exc
        except anthropic.AuthenticationError as exc:
            raise LLMError("Your Anthropic API key was rejected. Open Settings and check the key.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("The language model is busy right now. Please retry in a moment.") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the language model. Check your connection and retry.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError("The language model could not answer this time. Please retry.") from exc
        if final.stop_reason == "refusal":
            raise LLMError("The language model declined to answer this question.")

    def check(self) -> None:
        """Validate the key and model with the free Models API (no tokens are spent)."""
        try:
            self._client.models.retrieve(self.model)
        except anthropic.AuthenticationError as exc:
            raise LLMError("This API key was rejected by Anthropic. Check that you copied it completely.") from exc
        except anthropic.PermissionDeniedError as exc:
            raise LLMError("This API key does not have access to the configured model.") from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"The model '{self.model}' was not found. Check LLM_MODEL.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Anthropic is rate-limiting this key right now. Try again shortly.") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach Anthropic from the server.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError("Anthropic could not check this key right now. Try again shortly.") from exc
