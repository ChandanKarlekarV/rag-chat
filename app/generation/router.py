"""Pick the answer provider for a request (ANSWER_MODES.md sections 1 and 5).

Auto-selection: an API key is available (the user's from Settings, or the server's) → API;
else Ollama is reachable → Ollama; else Local. A mode the user chose explicitly (per request or
saved as their default) wins when it is available.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from app.generation.anthropic import AnthropicProvider
from app.generation.base import AnswerMode, AnswerProvider
from app.generation.llm_base import LLMClient, LLMError
from app.generation.ollama import NOT_RUNNING, OllamaLLM, OllamaProvider
from app.generation.rewrite import HeuristicRewriter, LLMRewriter, QueryRewriter

DESCRIPTIONS = {
    AnswerMode.LOCAL: "Finds and quotes the answer directly from your PDF. Free, private, works offline.",
    AnswerMode.OLLAMA: "Writes an answer using a model running on your computer.",
    AnswerMode.API: "Writes a fuller answer using Claude. Needs an API key.",
}
TITLES = {AnswerMode.LOCAL: "Local · no key", AnswerMode.OLLAMA: "Ollama", AnswerMode.API: "Claude API"}
OLLAMA_CACHE_S = 15.0

LLMFactory = Callable[[str], LLMClient]
OllamaFactory = Callable[[str], OllamaLLM]


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
        server_key: str,
        ollama_factory: OllamaFactory | None,
        default_ollama_model: str,
    ) -> None:
        self.local = local
        self.api_factory = api_factory
        self.server_key = server_key
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

    def modes(self, user_key: str | None) -> list[ModeInfo]:
        """Every mode with whether it can run right now and why."""
        has_key = bool((user_key or "").strip() or self.server_key)
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
                has_key,
                "Key added." if has_key else "Add your Anthropic API key in Settings.",
                DESCRIPTIONS[AnswerMode.API],
            ),
        ]

    def recommended(self, user_key: str | None) -> AnswerMode:
        """Auto-selection: API key → API; Ollama reachable → Ollama; else Local."""
        available = {m.mode for m in self.modes(user_key) if m.available}
        for mode in (AnswerMode.API, AnswerMode.OLLAMA, AnswerMode.LOCAL):
            if mode in available:
                return mode
        return AnswerMode.LOCAL

    # ---- selection ----

    def pick(
        self,
        requested: AnswerMode | None,
        user_key: str | None,
        preferred: AnswerMode | None = None,
        ollama_model: str | None = None,
    ) -> Choice:
        """Provider for this request. An explicit request is honoured even if it then fails, so the
        user sees a friendly reason; a saved preference falls back to auto when unavailable."""
        if requested is None:
            available = {m.mode for m in self.modes(user_key) if m.available}
            requested = preferred if preferred in available else self.recommended(user_key)
        key = (user_key or "").strip() or self.server_key
        if requested == AnswerMode.API:
            if not key:
                return Choice(
                    requested,
                    None,
                    HeuristicRewriter(),
                    "no_api_key",
                    "Add your Anthropic API key in Settings, or switch to Local mode.",
                )
            llm = self.api_factory(key)
            return Choice(requested, AnthropicProvider(llm), LLMRewriter(llm))
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
