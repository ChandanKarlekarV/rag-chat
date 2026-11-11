"""`LLMClient` for any OpenAI-compatible Chat Completions API (OpenAI, Gemini, Groq, OpenRouter,
Mistral, DeepSeek, xAI). Plain httpx, no SDK; see providers.py for the docs each base URL came from.

Request: `POST {base}/chat/completions` with `model`, `messages` and `stream`; Bearer auth.
Stream: data-only server-sent events whose JSON carries `choices[0].delta.content`, ended by
`data: [DONE]` (as documented by Mistral, xAI and Gemini's compatibility layer). No token limit is
sent: the field name differs between providers (`max_tokens` is deprecated on some), and the
prompts already ask for short output.
"""

import json
from collections.abc import Iterator

import httpx

from app.generation.llm_base import LLMError
from app.generation.providers import ProviderSpec, chat_models

_MAX_ERROR_CHARS = 200


def _provider_message(response: httpx.Response) -> str:
    """The provider's own short error text, if the body has one (never contains the key)."""
    try:
        body = response.json()
    except ValueError:
        return ""
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        return str(err.get("message", ""))[:_MAX_ERROR_CHARS]
    if isinstance(err, str):
        return err[:_MAX_ERROR_CHARS]
    return ""


class OpenAICompatLLM:
    """Streams answers from an OpenAI-compatible endpoint."""

    def __init__(
        self,
        spec: ProviderSpec,
        api_key: str,
        model: str,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.spec = spec
        self.provider = spec.id
        self.model = model or spec.default_model
        self._key = api_key
        self.timeout_s = timeout_s
        self._transport = transport  # tests inject httpx.MockTransport

    def _client(self, timeout: float) -> httpx.Client:
        headers = {"Authorization": f"Bearer {self._key}"}
        return httpx.Client(base_url=self.spec.base_url, headers=headers, timeout=timeout, transport=self._transport)

    def _raise_for(self, response: httpx.Response) -> None:
        """Map an error status to a friendly LLMError."""
        name = self.spec.name
        detail = _provider_message(response)
        status = response.status_code
        if status in (401, 403):
            raise LLMError(f"This API key was rejected by {name}. Check that you copied it completely.")
        if status == 404:
            raise LLMError(
                f"{name} doesn't offer the model '{self.model}' to this key. Pick another model in Settings."
            )
        if status == 429:
            raise LLMError(
                f"{name} is rate-limiting this key right now (or its credit has run out). Try again shortly."
            )
        if status == 400 and "key" in detail.lower():
            raise LLMError(f"This API key was rejected by {name}. Check that you copied it completely.")
        suffix = f" ({detail})" if detail else ""
        raise LLMError(f"{name} could not answer this time{suffix}. Please retry.")

    def list_models(self) -> list[str]:
        """Chat model ids this key can use (`GET /models` → `data[].id`)."""
        try:
            with self._client(10.0) as c:
                r = c.get("/models")
                if r.status_code >= 400:
                    self._raise_for(r)
                data = r.json().get("data", [])
        except httpx.HTTPError as exc:
            raise LLMError(f"Could not reach {self.spec.name}. Check your connection and retry.") from exc
        except ValueError as exc:
            raise LLMError(f"{self.spec.name} sent a reply we couldn't read.") from exc
        return chat_models([str(m.get("id", "")) for m in data if isinstance(m, dict) and m.get("id")])

    def check(self) -> None:
        """Validate the key with the free model listing (no tokens are spent)."""
        self.list_models()

    def _body(self, system: str, user: str, stream: bool) -> dict:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return {"model": self.model, "messages": messages, "stream": stream}

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Yield `choices[0].delta.content` pieces from the SSE stream."""
        try:
            with (
                self._client(self.timeout_s) as c,
                c.stream("POST", "/chat/completions", json=self._body(system, user, True)) as r,
            ):
                if r.status_code >= 400:
                    r.read()
                    self._raise_for(r)
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        return
                    event = json.loads(payload)
                    if event.get("error"):
                        raise LLMError(f"{self.spec.name} reported an error while answering. Please retry.")
                    for choice in event.get("choices") or []:
                        text = (choice.get("delta") or {}).get("content")
                        if text:
                            yield text
        except httpx.TimeoutException as exc:
            raise LLMError(f"{self.spec.name} took too long to answer. Please try again.") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Could not reach {self.spec.name}. Check your connection and retry.") from exc
        except ValueError as exc:
            raise LLMError(f"{self.spec.name} sent a reply we couldn't read.") from exc

    def complete(self, system: str, user: str, max_tokens: int = 512, json_mode: bool = False) -> str:
        """Non-streamed completion: `choices[0].message.content`. `json_mode` is left to the prompt:
        a JSON response-format flag isn't documented the same way by every provider (unverified)."""
        try:
            with self._client(max(self.timeout_s, 120.0)) as c:
                r = c.post("/chat/completions", json=self._body(system, user, False))
                if r.status_code >= 400:
                    self._raise_for(r)
                choices = r.json().get("choices") or []
        except httpx.TimeoutException as exc:
            raise LLMError(f"{self.spec.name} took too long to answer. Please try again.") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Could not reach {self.spec.name}. Check your connection and retry.") from exc
        except ValueError as exc:
            raise LLMError(f"{self.spec.name} sent a reply we couldn't read.") from exc
        if not choices:
            return ""
        return str((choices[0].get("message") or {}).get("content") or "")
