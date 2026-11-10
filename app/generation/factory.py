"""Build the configured `LLMClient` for a given API key."""

from app.config import Settings
from app.generation.llm_anthropic import AnthropicLLM
from app.generation.llm_base import LLMClient
from app.generation.llm_fake import FakeLLM


def make_llm(settings: Settings, api_key: str) -> LLMClient:
    """Return an LLM for `LLM_PROVIDER`; `api_key` is ignored by the fake provider."""
    if settings.llm_provider == "fake":
        return FakeLLM()
    return AnthropicLLM(api_key, settings.llm_model, settings.llm_max_tokens, settings.llm_timeout_s)
