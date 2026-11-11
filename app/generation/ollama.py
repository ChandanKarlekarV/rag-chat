"""Ollama provider: generative answers from a model running on the user's machine.

API verified against docs.ollama.com (api/chat, api/tags, faq):
* `POST /api/chat` with `model`, `messages` [{role, content}], `stream`; streaming replies are
  newline-delimited JSON (`application/x-ndjson`) whose lines carry `message.content` and `done`.
* `GET /api/tags` returns `{"models": [{"name": ...}, ...]}` (installed models).
* Default address `http://localhost:11434`.
* `/api/chat` accepts `format: "json"` (JSON output) and `options.num_ctx` / `options.num_predict`
  (docs.ollama.com/api/chat ChatRequest and ModelOptions schemas, checked 2026-11-11).
"""

import json
import time
from collections.abc import Iterator

import httpx

from app.generation.base import AnswerMode
from app.generation.generative import GenerativeProvider
from app.generation.llm_base import LLMError

NOT_RUNNING = "Ollama isn't running. Start it, or switch to Local mode."
TOO_SLOW = (
    "Ollama took too long to answer. If your laptop is on battery its GPU runs slowly: plug it in, "
    "try a smaller model, or use Local mode."
)


class OllamaLLM:
    """`LLMClient` over Ollama's local HTTP API."""

    provider = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        max_reply_s: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s  # per read: the longest wait for the next piece of the reply
        self.max_reply_s = max_reply_s  # the longest a whole non-streamed reply may take
        self._transport = transport  # tests inject httpx.MockTransport

    def _client(self, timeout: float) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=timeout, transport=self._transport)

    def list_models(self) -> list[str]:
        """Names of installed models; raises LLMError if Ollama is unreachable."""
        try:
            with self._client(2.0) as c:
                r = c.get("/api/tags")
                r.raise_for_status()
                return [m["name"] for m in r.json().get("models", [])]
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            raise LLMError(NOT_RUNNING) from exc

    def check(self) -> None:
        """Ollama must be reachable and the chosen model installed."""
        names = self.list_models()
        wanted = {self.model, f"{self.model}:latest"}
        if not wanted & set(names):
            raise LLMError(f"The Ollama model '{self.model}' isn't installed. Run: ollama pull {self.model}")

    def _messages(self, system: str, user: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _chat(self, body: dict, deadline_s: float | None = None) -> Iterator[str]:
        """Stream `message.content` deltas from `/api/chat` (NDJSON). The read timeout applies per
        line, so a slow but working model never times out; `deadline_s` caps the whole reply."""
        started = time.monotonic()
        try:
            with self._client(self.timeout_s) as c, c.stream("POST", "/api/chat", json=body) as r:
                if r.status_code == 404:
                    raise LLMError(f"The Ollama model '{self.model}' isn't installed. Run: ollama pull {self.model}")
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if event.get("error"):
                        raise LLMError(f"Ollama reported an error: {event['error']}")
                    text = (event.get("message") or {}).get("content", "")
                    if text:
                        yield text
                    if event.get("done"):
                        return
                    if deadline_s is not None and time.monotonic() - started > deadline_s:
                        raise LLMError(TOO_SLOW)
        except httpx.TimeoutException as exc:
            raise LLMError(TOO_SLOW) from exc
        except httpx.HTTPError as exc:
            raise LLMError(NOT_RUNNING) from exc
        except ValueError as exc:
            raise LLMError("Ollama sent a reply we couldn't read. Please retry.") from exc

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Stream the answer's text deltas."""
        body = {"model": self.model, "messages": self._messages(system, user), "stream": True}
        yield from self._chat(body)

    def complete(self, system: str, user: str, max_tokens: int = 512, json_mode: bool = False) -> str:
        """Whole reply as one string (follow-up rewriting, study material). Streamed underneath so
        long replies on a slow GPU don't hit the read timeout. `json_mode` sets Ollama's
        `format: "json"`, which constrains the model to valid JSON."""
        body: dict = {
            "model": self.model,
            "messages": self._messages(system, user),
            "stream": True,
            # num_ctx: room for the study-material excerpts (Ollama's default context is smaller).
            "options": {"num_predict": max_tokens, "num_ctx": 8192},
        }
        if json_mode:
            body["format"] = "json"
        return "".join(self._chat(body, self.max_reply_s))


class OllamaProvider(GenerativeProvider):
    """Generative answers from a local Ollama model."""

    def __init__(self, llm: OllamaLLM) -> None:
        super().__init__(llm, AnswerMode.OLLAMA)
        self.label = f"Generated by {llm.model} (Ollama, on this machine)"
