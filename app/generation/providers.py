"""Which AI provider an API key belongs to, and how to reach it.

Base URLs and model ids were checked against each provider's docs on 2026-11-11:
* OpenAI: Bearer auth, `GET /v1/models` → `{"data": [{"id": ...}]}` (developers.openai.com API reference).
* Google Gemini: OpenAI-compatible base `https://generativelanguage.googleapis.com/v1beta/openai/`
  with streaming and `/models` (aistudio.google.com/docs/openai).
* Groq: `https://api.groq.com/openai/v1` (console.groq.com/docs/openai, /docs/models).
* OpenRouter: `https://openrouter.ai/api/v1`, model ids are `provider/slug` (openrouter.ai/docs).
* Mistral: `https://api.mistral.ai/v1`, SSE stream ending in `data: [DONE]` (docs.mistral.ai/api).
* DeepSeek: `https://api.deepseek.com` + `/chat/completions` (api-docs.deepseek.com).
* xAI: `https://api.x.ai/v1/chat/completions`, SSE ending in `data: [DONE]` (docs.x.ai).

unverified: the key prefixes used for auto-detection are not stated in those docs; they are the
formats keys are issued in today. When detection fails, the user picks the provider in Settings.
unverified: OpenAI's own Chat Completions reference page could not be fetched; the request and
stream format used for it is the one Gemini, Mistral and xAI document as "OpenAI-compatible".
"""

from dataclasses import dataclass
from typing import Literal

from app.generation.llm_base import LLMError

Kind = Literal["anthropic", "openai"]


@dataclass(frozen=True)
class ProviderSpec:
    """One supported provider."""

    id: str
    name: str
    kind: Kind  # "anthropic" = Anthropic SDK; "openai" = OpenAI-compatible Chat Completions over HTTP
    base_url: str
    default_model: str  # used when the key's model list doesn't say otherwise
    prefixes: tuple[str, ...]  # key prefixes for auto-detection, most specific first
    key_url: str  # where users create a key


PROVIDERS: dict[str, ProviderSpec] = {
    p.id: p
    for p in (
        ProviderSpec(
            "anthropic",
            "Anthropic (Claude)",
            "anthropic",
            "",
            "claude-sonnet-5-5",
            ("sk-ant-",),
            "https://console.anthropic.com/settings/keys",
        ),
        ProviderSpec(
            "openrouter",
            "OpenRouter",
            "openai",
            "https://openrouter.ai/api/v1",
            "",
            ("sk-or-",),
            "https://openrouter.ai/settings/keys",
        ),
        ProviderSpec(
            "openai",
            "OpenAI",
            "openai",
            "https://api.openai.com/v1",
            "gpt-6.1-sol",
            ("sk-",),
            "https://platform.openai.com/api-keys",
        ),
        ProviderSpec(
            "gemini",
            "Google Gemini",
            "openai",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "gemini-3.8-flash",
            ("AIza",),
            "https://aistudio.google.com/apikey",
        ),
        ProviderSpec(
            "groq",
            "Groq",
            "openai",
            "https://api.groq.com/openai/v1",
            "llama-3.3-70b-versatile",
            ("gsk_",),
            "https://console.groq.com/keys",
        ),
        ProviderSpec(
            "xai",
            "xAI (Grok)",
            "openai",
            "https://api.x.ai/v1",
            "grok-4.7",
            ("xai-",),
            "https://console.x.ai",
        ),
        ProviderSpec(
            "mistral",
            "Mistral",
            "openai",
            "https://api.mistral.ai/v1",
            "mistral-large-latest",
            (),
            "https://console.mistral.ai/api-keys",
        ),
        ProviderSpec(
            "deepseek",
            "DeepSeek",
            "openai",
            "https://api.deepseek.com",
            "deepseek-flash",
            (),
            "https://platform.deepseek.com/api_keys",
        ),
    )
}

# Model ids that are not chat models (filtered out of the model picker).
_NOT_CHAT = (
    "embed",
    "whisper",
    "tts",
    "audio",
    "image",
    "moderation",
    "dall-e",
    "guard",
    "transcribe",
    "realtime",
    "search",
    "rerank",
    "ocr",
    "veo",
    "imagen",
    "aqa",
    "speech",
)


@dataclass(frozen=True)
class ApiCredentials:
    """A key plus the provider and model it should be used with. Never logged or stored."""

    key: str
    provider: str
    model: str = ""  # empty = the provider's default

    def __repr__(self) -> str:  # keep the key out of tracebacks and logs
        return f"ApiCredentials(provider={self.provider!r}, model={self.model!r}, key=***)"

    @property
    def spec(self) -> ProviderSpec:
        """The provider's spec."""
        return PROVIDERS[self.provider]


def detect_provider(key: str) -> str | None:
    """Guess the provider from the key's prefix, or None when it can't be told apart."""
    for spec in PROVIDERS.values():  # dict order puts "sk-ant-"/"sk-or-" before the generic "sk-"
        if any(key.startswith(p) for p in spec.prefixes):
            return spec.id
    return None


def resolve_credentials(key: str | None, provider: str | None = None, model: str | None = None) -> ApiCredentials:
    """Turn the pasted key and the optional provider/model choice into credentials.

    Raises LLMError (a message safe to show) when the key is missing or its provider is unknown.
    """
    key = (key or "").strip()
    if not key:
        raise LLMError("Paste your API key first.")
    chosen = (provider or "").strip().lower()
    if chosen in ("", "auto"):
        chosen = detect_provider(key) or ""
        if not chosen:
            raise LLMError("We couldn't tell which provider this key is for. Choose the provider in Settings.")
    if chosen not in PROVIDERS:
        raise LLMError("That provider isn't supported. Choose one from the list in Settings.")
    return ApiCredentials(key, chosen, (model or "").strip()[:200])


def chat_models(ids: list[str]) -> list[str]:
    """Keep chat-capable model ids (drop embedding, speech, image… models), sorted, without the
    `models/` prefix some providers add (Gemini's examples use the bare id)."""
    bare = (i.removeprefix("models/") for i in ids)
    return sorted({i for i in bare if not any(word in i.lower() for word in _NOT_CHAT)})


def default_model(spec: ProviderSpec, available: list[str]) -> str:
    """The provider's default if the key can use it, else the first chat model the key lists."""
    if spec.default_model and (not available or spec.default_model in available):
        return spec.default_model
    return available[0] if available else spec.default_model
