"""Pick the answer provider for a request (ANSWER_MODES.md sections 1 and 5).

Auto-selection: an API key is available (the user's from Settings, or the server's) → API;
else Ollama is reachable → Ollama; else Local. A mode the user chose explicitly (per request or
saved as their default) wins when it is available.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from app.generation.anthropic import ApiProvider
from app.generation.base import AnswerMode, AnswerProvider
from app.generation.llm_base import LLMClient, LLMError
from app.generation.ollama import NOT_RUNNING, OllamaLLM, OllamaProvider
from app.generation.providers import ApiCredentials, resolve_credentials
from app.generation.rewrite import HeuristicRewriter, LLMRewriter, QueryRewriter

DESCRIPTIONS = {
    AnswerMode.LOCAL: "Finds and quotes the answer directly from your PDF. Free, private, works offline.",
    AnswerMode.OLLAMA: "Writes an answer using a model running on your computer.",
    AnswerMode.API: (
        "Writes a fuller answer using your AI provider (Claude, OpenAI, Gemini, Groq and more). Needs an API key."
    ),
}
TITLES = {AnswerMode.LOCAL: "Local · no key", AnswerMode.OLLAMA: "Ollama", AnswerMode.API: "API key"}
NO_KEY = "Add an API key from any supported provider in Settings."
OLLAMA_CACHE_S = 15.0

LLMFactory = Callable[[ApiCredentials], LLMClient]
OllamaFactory = Callable[[str], OllamaLLM]


@dataclass(frozen=True)
class KeyInput:
    """What the browser sent for API mode: the pasted key and optional provider/model choices."""

    key: str | None = None
    provider: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class ModeInfo:
    """Availability of one mode, for the UI's mode control."""

    mode: AnswerMode
    title: str
    available: bool
    reason: str
    description: str


@dataclass(frozen=True)
class Choice:
    """The provider that will answer, or why none can."""

    mode: AnswerMode
    provider: AnswerProvider | None
    rewriter: QueryRewriter
    error_code: str | None = None
    error_message: str | None = None


@dataclass
class _OllamaStatus:
    checked_at: float = 0.0
    ok: bool = False
    models: list[str] = field(default_factory=list)
    reason: str = NOT_RUNNING


class AnswerRouter:
    """Knows which modes can run and builds the provider for a request."""

    def __init__(
        self,
        local: AnswerProvider | None,
        api_factory: LLMFactory,
        server_creds: ApiCredentials | None,
        ollama_factory: OllamaFactory | None,
        default_ollama_model: str,
    ) -> None:
        self.local = local
        self.api_factory = api_factory
        self.server_creds = server_creds
        self.ollama_factory = ollama_factory
        self.default_ollama_model = default_ollama_model
        self._ollama = _OllamaStatus()

    # ---- availability ----

    def ollama_status(self, force: bool = False) -> _OllamaStatus:
        """Reachability and installed models, cached for a few seconds."""
        if self.ollama_factory is None:
            return _OllamaStatus(time.monotonic(), False, [], "Ollama isn't configured on this server.")
        if force or time.monotonic() - self._ollama.checked_at > OLLAMA_CACHE_S:
            try:
                models = self.ollama_factory(self.default_ollama_model).list_models()
                ok = bool(models)
                reason = (
                    f"{len(models)} model(s) installed."
                    if ok
                    else "Ollama is running but has no models. Run: ollama pull llama3.2"
                )
                self._ollama = _OllamaStatus(time.monotonic(), ok, models, reason)
            except LLMError as exc:
                self._ollama = _OllamaStatus(time.monotonic(), False, [], str(exc))
        return self._ollama

    def credentials(self, key: KeyInput | None) -> ApiCredentials | None:
        """The user's key (with its provider) if they sent one, else the server's, else None.

        Raises LLMError when the user sent a key whose provider can't be determined.
        """
        if key is not None and (key.key or "").strip():
            return resolve_credentials(key.key, key.provider, key.model)
        return self.server_creds

    def _api_status(self, key: KeyInput | None) -> tuple[bool, str]:
        try:
            creds = self.credentials(key)
        except LLMError as exc:
            return False, str(exc)
        if creds is None:
            return False, NO_KEY
        return True, f"Key added ({creds.spec.name})."

    def modes(self, key: KeyInput | None) -> list[ModeInfo]:
        """Every mode with whether it can run right now and why."""
        api_ok, api_reason = self._api_status(key)
        ollama = self.ollama_status()
        local_ok = self.local is not None
        return [
            ModeInfo(
                AnswerMode.LOCAL,
                TITLES[AnswerMode.LOCAL],
                local_ok,
                "Runs on this server. No key needed." if local_ok else "The local model isn't loaded on this server.",
                DESCRIPTIONS[AnswerMode.LOCAL],
            ),
            ModeInfo(
                AnswerMode.OLLAMA, TITLES[AnswerMode.OLLAMA], ollama.ok, ollama.reason, DESCRIPTIONS[AnswerMode.OLLAMA]
            ),
            ModeInfo(
                AnswerMode.API,
                TITLES[AnswerMode.API],
                api_ok,
                api_reason,
                DESCRIPTIONS[AnswerMode.API],
            ),
        ]

    def recommended(self, key: KeyInput | None) -> AnswerMode:
        """Auto-selection: API key → API; Ollama reachable → Ollama; else Local."""
        available = {m.mode for m in self.modes(key) if m.available}
        for mode in (AnswerMode.API, AnswerMode.OLLAMA, AnswerMode.LOCAL):
            if mode in available:
                return mode
        return AnswerMode.LOCAL

    # ---- selection ----

    def pick(
        self,
        requested: AnswerMode | None,
        key: KeyInput | None,
        preferred: AnswerMode | None = None,
        ollama_model: str | None = None,
    ) -> Choice:
        """Provider for this request. An explicit request is honoured even if it then fails, so the
        user sees a friendly reason; a saved preference falls back to auto when unavailable."""
        if requested is None:
            available = {m.mode for m in self.modes(key) if m.available}
            requested = preferred if preferred in available else self.recommended(key)
        if requested == AnswerMode.API:
            try:
                creds = self.credentials(key)
            except LLMError as exc:
                return Choice(requested, None, HeuristicRewriter(), "bad_api_key", str(exc))
            if creds is None:
                return Choice(requested, None, HeuristicRewriter(), "no_api_key", f"{NO_KEY} Or switch to Local mode.")
            llm = self.api_factory(creds)
            return Choice(requested, ApiProvider(llm), LLMRewriter(llm))
        if requested == AnswerMode.OLLAMA:
            if self.ollama_factory is None:
                return Choice(requested, None, HeuristicRewriter(), "ollama_unreachable", NOT_RUNNING)
            ollama_llm = self.ollama_factory(ollama_model or self.default_ollama_model)
            return Choice(requested, OllamaProvider(ollama_llm), LLMRewriter(ollama_llm))
        if self.local is None:
            return Choice(
                AnswerMode.LOCAL,
                None,
                HeuristicRewriter(),
                "local_unavailable",
                "The local model isn't loaded on this server.",
            )
        return Choice(AnswerMode.LOCAL, self.local, HeuristicRewriter())
