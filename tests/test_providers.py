"""Bring-any-key support: provider detection and the OpenAI-compatible client (mocked HTTP)."""

import json

import httpx
import pytest

from app.config import Settings
from app.generation.llm_base import LLMError
from app.generation.llm_openai import OpenAICompatLLM
from app.generation.providers import PROVIDERS, chat_models, default_model, detect_provider, resolve_credentials
from app.services.container import make_api_llm


@pytest.mark.parametrize(
    ("key", "provider"),
    [
        ("sk-ant-api03-abc", "anthropic"),
        ("sk-or-v1-abc", "openrouter"),
        ("sk-proj-abc", "openai"),
        ("sk-abc", "openai"),
        ("AIzaSyAbc", "gemini"),
        ("gsk_abc", "groq"),
        ("xai-abc", "xai"),
        ("abc123", None),
    ],
)
def test_detect_provider(key, provider):
    assert detect_provider(key) == provider


def test_resolve_credentials_choices_and_errors():
    assert resolve_credentials("abc123", "mistral", "mistral-small-latest").provider == "mistral"
    assert resolve_credentials("sk-abc", "deepseek").provider == "deepseek"  # explicit choice beats the prefix
    assert resolve_credentials("gsk_x", "auto").provider == "groq"
    with pytest.raises(LLMError, match="Choose the provider"):
        resolve_credentials("abc123")
    with pytest.raises(LLMError, match="isn't supported"):
        resolve_credentials("sk-abc", "acme")
    with pytest.raises(LLMError, match="Paste"):
        resolve_credentials("  ")
    assert "key=***" in repr(resolve_credentials("sk-secret"))  # never printed


def test_model_list_filtering_and_default():
    ids = ["models/gemini-3.8-flash", "gemini-embedding-001", "whisper-large-v3", "llama-3.3-70b-versatile"]
    assert chat_models(ids) == ["gemini-3.8-flash", "llama-3.3-70b-versatile"]
    assert default_model(PROVIDERS["gemini"], ["gemini-3.8-flash", "x"]) == "gemini-3.8-flash"
    assert default_model(PROVIDERS["openrouter"], ["a/b", "c/d"]) == "a/b"  # no default → first listed


def test_factory_builds_the_right_client():
    s = Settings(_env_file=None)
    assert type(make_api_llm(s, resolve_credentials("gsk_x"))).__name__ == "OpenAICompatLLM"
    assert type(make_api_llm(s, resolve_credentials("sk-ant-x"))).__name__ == "AnthropicLLM"


def _server(seen: list[httpx.Request], status: int = 200) -> httpx.MockTransport:
    """A fake OpenAI-compatible server: /models and streamed /chat/completions."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if status != 200:
            return httpx.Response(status, json={"error": {"message": "Invalid API key provided"}})
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": "m-chat"}, {"id": "m-embed"}]})
        body = json.loads(request.content)
        if not body["stream"]:
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "rewritten"}}]})
        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"role": "assistant"}}]}),
            "data: " + json.dumps({"choices": [{"delta": {"content": "Hello "}}]}),
            ": keep-alive comment",
            "data: " + json.dumps({"choices": [{"delta": {"content": "[1]"}}]}),
            "data: [DONE]",
        ]
        return httpx.Response(200, content="\n\n".join(lines).encode(), headers={"content-type": "text/event-stream"})

    return httpx.MockTransport(handler)


def test_openai_compatible_stream_complete_and_models():
    seen: list[httpx.Request] = []
    llm = OpenAICompatLLM(PROVIDERS["groq"], "gsk_secret", "m-chat", transport=_server(seen))
    assert "".join(llm.stream("sys", "user")) == "Hello [1]"
    assert llm.complete("sys", "user") == "rewritten"
    assert llm.list_models() == ["m-chat"]
    first = seen[0]
    assert str(first.url) == "https://api.groq.com/openai/v1/chat/completions"
    assert first.headers["authorization"] == "Bearer gsk_secret"
    body = json.loads(first.content)
    assert body["model"] == "m-chat" and body["messages"][0] == {"role": "system", "content": "sys"}


@pytest.mark.parametrize(("status", "words"), [(401, "rejected"), (404, "doesn't offer"), (429, "rate-limiting")])
def test_openai_compatible_errors_are_friendly(status, words):
    llm = OpenAICompatLLM(PROVIDERS["openai"], "sk-bad", "m", transport=_server([], status))
    with pytest.raises(LLMError, match=words):
        list(llm.stream("s", "u"))
    with pytest.raises(LLMError):
        llm.check()


def test_slow_ollama_says_too_slow_not_unreachable():
    from app.generation.ollama import TOO_SLOW, OllamaLLM

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    llm = OllamaLLM("http://ollama.test", "llama3.2", transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError, match="took too long") as info:
        llm.complete("s", "u")
    assert str(info.value) == TOO_SLOW


def test_gemini_base_url_joins_correctly():
    seen: list[httpx.Request] = []
    OpenAICompatLLM(PROVIDERS["gemini"], "AIza", "g", transport=_server(seen)).list_models()
    assert str(seen[0].url) == "https://generativelanguage.googleapis.com/v1beta/openai/models"
