"""Shared fixtures: an app wired with real SQLite/Chroma/BM25 and fake models + FakeLLM."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.generation.llm_fake import FakeLLM
from app.main import create_app
from app.services.container import build_container
from scripts.make_sample_pdf import build
from tests.fakes import FakeEmbedder, FakeReranker

GOOD_KEY = "sk-ant-test-good"
CSRF = {"X-Requested-With": "fetch"}


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory) -> Path:
    return build(tmp_path_factory.mktemp("pdf") / "halcyon.pdf")


class Harness:
    """Test client plus handles to the fakes."""

    def __init__(self, client: TestClient, made: list[FakeLLM]) -> None:
        self.client = client
        self.made = made

    def new_client(self) -> TestClient:
        """A second browser (separate cookie jar) against the same app."""
        return TestClient(self.client.app)


def make_harness(tmp_path: Path, sample_pdf: Path, server_key: str = "", **overrides) -> Iterator[Harness]:
    settings = Settings(
        data_dir=tmp_path, anthropic_api_key=server_key, min_rerank_score=0.0, llm_provider="anthropic", **overrides
    )
    made: list[FakeLLM] = []

    def factory(key: str) -> FakeLLM:
        llm = FakeLLM("The battery stores 11.2 kWh of usable energy [1].", valid=key in (GOOD_KEY, "server-key"))
        made.append(llm)
        return llm

    container = build_container(
        settings, embedder=FakeEmbedder(), reranker=FakeReranker(), llm_factory=factory, sample_pdf=sample_pdf
    )
    with TestClient(create_app(container)) as client:
        yield Harness(client, made)


@pytest.fixture
def h(tmp_path, sample_pdf) -> Iterator[Harness]:
    """App without a server key (users bring their own)."""
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
    headers = CSRF | ({"X-Anthropic-Api-Key": key} if key else {})
    r = client.post("/chat/ask", json={"question": question, **body}, headers=headers)
    assert r.status_code == 200, r.text
    return sse(r.text)
