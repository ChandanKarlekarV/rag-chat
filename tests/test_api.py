"""End-to-end API tests with real Chroma/SQLite/BM25 and fake models + fake LLM."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import DocumentDB
from app.index import DocumentIndex
from app.main import Services, create_app
from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector_store import VectorStore
from scripts.make_sample_pdf import build
from tests.fakes import FakeEmbedder, FakeLLM, FakeReranker


@pytest.fixture(scope="module")
def sample_pdf(tmp_path_factory) -> Path:
    return build(tmp_path_factory.mktemp("pdf") / "halcyon.pdf")


@pytest.fixture
def client(tmp_path):
    settings = Settings(data_dir=tmp_path, min_rerank_score=0.0, anthropic_api_key="")
    embedder, reranker = FakeEmbedder(), FakeReranker()
    vectors, bm25 = VectorStore(settings.chroma_dir), BM25Store(settings.bm25_path)
    index = DocumentIndex(DocumentDB(settings.sqlite_path), vectors, bm25, embedder, settings.uploads_dir)
    llm = FakeLLM("The battery stores 11.2 kWh [1].")
    services = Services(settings, index, HybridRetriever(embedder, vectors, bm25, reranker), llm)
    with TestClient(create_app(services)) as c:
        c.llm = llm
        yield c


def _sse(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def _upload(client, path: Path, name: str | None = None):
    with path.open("rb") as fh:
        return client.post("/upload", files={"file": (name or path.name, fh, "application/pdf")})


def test_health_and_index_page(client):
    assert client.get("/health").json() == {"status": "ok", "documents": 0, "llm_configured": True}
    assert "rag-chat" in client.get("/").text


def test_upload_list_dedupe_delete(client, sample_pdf):
    r = _upload(client, sample_pdf)
    assert r.status_code == 200, r.text
    doc = r.json()["document"]
    assert doc["pages"] == 10 and doc["chunks"] > 10
    assert _upload(client, sample_pdf, "copy.pdf").json()["duplicate"] is True
    assert [d["doc_id"] for d in client.get("/documents").json()] == [doc["doc_id"]]
    assert client.delete(f"/documents/{doc['doc_id']}").status_code == 200
    assert client.get("/documents").json() == []
    assert client.delete(f"/documents/{doc['doc_id']}").status_code == 404


def test_upload_rejects_non_pdf_and_textless_pdf(client, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello")
    r = _upload(client, bad)
    assert r.status_code == 400 and "not a PDF" in r.json()["detail"]

    import pymupdf

    blank = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(blank)
    r = _upload(client, blank)
    assert r.status_code == 400 and "no extractable text" in r.json()["detail"]


def test_ask_streams_answer_with_citations(client, sample_pdf):
    _upload(client, sample_pdf)
    r = client.post("/ask", json={"question": "How much usable energy capacity does the battery have?"})
    events = _sse(r.text)
    assert r.headers["content-type"].startswith("text/event-stream")
    assert any(e == "token" for e, _ in events)
    name, done = events[-1]
    assert name == "done" and done["answerable"] is True
    assert done["citations"][0]["n"] == 1 and done["citations"][0]["filename"] == "halcyon.pdf"


def test_ask_unrelated_question_falls_back(client, sample_pdf):
    _upload(client, sample_pdf)
    r = client.post("/ask", json={"question": "chocolate cookie recipe"})
    name, done = _sse(r.text)[-1]
    assert name == "done" and done["answerable"] is False
    assert client.llm.calls == []


def test_ask_with_no_documents_returns_error(client):
    name, data = _sse(client.post("/ask", json={"question": "anything?"}).text)[-1]
    assert name == "error" and "Upload" in data["message"]
