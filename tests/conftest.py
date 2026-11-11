"""Shared fixtures: an app wired with real SQLite/Chroma/BM25 and fake models, LLM and Ollama."""

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.generation.fake import FakeLLM
from app.generation.ollama import OllamaLLM
from app.generation.providers import ApiCredentials
from app.main import create_app
from app.services.container import build_container
from scripts.make_sample_pdf import build
from tests.fakes import FakeEmbedder, FakeReader, FakeReranker

GOOD_KEY = "sk-ant-test-good"
CSRF = {"X-Requested-With": "fetch"}
OLLAMA_ANSWER = "The battery stores 11.2 kWh [1]."


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> Path:
    return build(tmp_path_factory.mktemp("pdf") / "halcyon.pdf")


def ollama_transport(running: bool = True, models: tuple[str, ...] = ("llama3.2:latest",)) -> httpx.MockTransport:
    """A fake Ollama server speaking the documented /api/tags and NDJSON /api/chat formats."""

    def handler(request: httpx.Request) -> httpx.Response:
        if not running:
            raise httpx.ConnectError("connection refused", request=request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": m} for m in models]})
        if request.url.path == "/api/chat":
            body = json.loads(request.content)
            if not body.get("stream", True):
                return httpx.Response(
                    200, json={"message": {"role": "assistant", "content": "rewritten"}, "done": True}
                )
            words = OLLAMA_ANSWER.split(" ")
            lines = [json.dumps({"message": {"role": "assistant", "content": w + " "}, "done": False}) for w in words]
            lines.append(json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}))
            return httpx.Response(
                200, content="\n".join(lines).encode(), headers={"content-type": "application/x-ndjson"}
            )
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


class Harness:
    """Test client plus handles to the fakes."""

    def __init__(self, client: TestClient, made: list[FakeLLM]) -> None:
        self.client = client
        self.made = made

    def new_client(self) -> TestClient:
        """A second browser (separate cookie jar) against the same app."""
        return TestClient(self.client.app)


def make_harness(
    tmp_path: Path,
    sample_pdf: Path,
    server_key: str = "",
    redis_client: object = None,
    ollama: bool | None = None,
    local: bool = True,
    **overrides,
) -> Iterator[Harness]:
    """`ollama`: None = not configured, True = running, False = configured but unreachable."""
    overrides.setdefault("min_qa_score", 0.0)
    settings = Settings(
        data_dir=tmp_path,
        anthropic_api_key=server_key,
        min_rerank_score=0.0,
        database_url="",
        ollama_enabled=ollama is not None,
        _env_file=None,
        **overrides,
    )
    made: list[FakeLLM] = []

    def factory(creds: ApiCredentials) -> FakeLLM:
        llm = FakeLLM("The battery stores 11.2 kWh of usable energy [1].", valid=creds.key in (GOOD_KEY, "server-key"))
        made.append(llm)
        return llm

    ollama_factory = None
    if ollama is not None:
        transport = ollama_transport(running=ollama)
        ollama_factory = lambda model: OllamaLLM("http://ollama.test", model, transport=transport)  # noqa: E731

    container = build_container(
        settings,
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        llm_factory=factory,
        reader=FakeReader() if local else None,
        sample_pdf=sample_pdf,
        redis_client=redis_client,
        ollama_factory=ollama_factory,
    )
    with TestClient(create_app(container)) as client:
        yield Harness(client, made)


@pytest.fixture
def h(tmp_path, sample_pdf) -> Iterator[Harness]:
    """App with Local mode available, no server key, no Ollama."""
    yield from make_harness(tmp_path, sample_pdf)


def register(client: TestClient, email: str = "ada@example.com", password: str = "correct-horse") -> dict:
    r = client.post("/auth/register", json={"email": email, "password": password}, headers=CSRF)
    assert r.status_code == 201, r.text
    return r.json()


def upload(client: TestClient, path: Path, name: str | None = None):
    with path.open("rb") as fh:
        return client.post("/documents", files={"file": (name or path.name, fh, "application/pdf")}, headers=CSRF)


def sse(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def ask(client: TestClient, question: str, key: str | None = GOOD_KEY, **body) -> list[tuple[str, dict]]:
    """Ask via the API. With a key and no explicit mode, auto-selection picks API mode."""
    headers = CSRF | ({"X-LLM-Key": key} if key else {})
    r = client.post("/chat/ask", json={"question": question, **body}, headers=headers)
    assert r.status_code == 200, r.text
    return sse(r.text)
