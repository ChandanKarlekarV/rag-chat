"""Answer modes (ANSWER_MODES.md section 9): router, endpoints, extractive provider, BYOK, SSE contract, Ollama."""

import logging
import os

import pytest

from app.generation.base import AnswerMode
from app.generation.extractive import LocalExtractiveProvider, answer_type_ok, split_sentences
from app.generation.fake import FakeLLM
from app.generation.prompt import NO_ANSWER_TOKEN
from app.generation.question_type import classify
from app.generation.reader import Span
from app.generation.router import AnswerRouter
from app.schemas import Chunk, RetrievedChunk
from tests.conftest import CSRF, GOOD_KEY, OLLAMA_ANSWER, ask, make_harness, register, upload

QUESTION = "How much usable energy capacity does the battery have?"


def _src(text: str, rerank: float = 1.0, page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=f"d:{page}",
            doc_id="d",
            filename="f.pdf",
            page_start=page,
            page_end=page,
            index=page,
            text=text,
            token_count=1,
        ),
        rerank_score=rerank,
    )


class StubReader:
    def __init__(self, spans):
        self.spans = spans

    def best_spans(self, question, sources):
        return self.spans


def overlap_scorer(question: str, sentences: list[str]) -> list[float]:
    q = set(question.lower().split())
    return [float(len(q & set(s.lower().split()))) - 1.0 for s in sentences]


# ---------- question type ----------


@pytest.mark.parametrize(
    ("q", "expected"),
    [
        ("Summarize this document", "summary"),
        ("Give me an overview", "summary"),
        ("What is this manual about?", "summary"),
        ("Why does charging stop below -5 degrees?", "explain"),
        ("What is the difference between Self-Powered and Backup Only?", "explain"),
        ("Compare the two warranties", "explain"),
        ("How heavy is the battery?", "factoid"),
        ("What does error E-214 mean?", "factoid"),
    ],
)
def test_classify(q, expected):
    assert classify(q) == expected


# ---------- extractive provider ----------

TEXT = "The enclosure is rated IP56. It weighs 118 kg and is 1150 mm high. Up to four units can be combined."


def _local(spans, min_qa=0.05):
    return LocalExtractiveProvider(StubReader(spans), "stub", overlap_scorer, min_qa_score=min_qa)


def test_extractive_answer_is_span_plus_quoted_sentences():
    start = TEXT.index("118 kg")
    provider = _local([Span(1, start, start + 6, 5.0)])
    events = list(provider.answer("How heavy is it in kg?", [_src(TEXT)], "factoid"))
    meta = next(e.data for e in events if e.kind == "meta")
    text = "".join(e.text for e in events if e.kind == "token")
    assert text.startswith("**Answer:** 118 kg [1]")
    assert "**From the document:**\n" in text and "\n> It weighs 118 kg and is 1150 mm high. [1]" in text
    assert sum(line.startswith("> ") for line in text.splitlines()) <= 3  # MAX_SENTENCES
    assert meta["span"] == "118 kg" and meta["span_char_range"] == [start, start + 6] and meta["confidence"] > 0.9
    for line in text.splitlines():  # every quoted sentence is verbatim from the chunk
        if line.startswith("> "):
            assert line[2:].rsplit(" [", 1)[0] in TEXT


def test_extractive_refuses_when_not_confident_or_wrong_type():
    start = TEXT.index("118 kg")
    unsure = _local([Span(1, start, start + 6, -6.0)])  # sigmoid(-6) ≈ 0.002 < 0.05
    assert "".join(e.text for e in unsure.answer("How heavy?", [_src(TEXT)], "factoid")) == NO_ANSWER_TOKEN
    wrong_type = _local([Span(1, start, start + 6, 5.0)])
    assert (
        "".join(e.text for e in wrong_type.answer("How much does it cost?", [_src(TEXT)], "factoid")) == NO_ANSWER_TOKEN
    )
    assert "".join(e.text for e in _local([]).answer("q", [], "factoid")) == NO_ANSWER_TOKEN


def test_extractive_prefers_the_better_ranked_chunk():
    a, b = "The H2 weighs 118 kg.", "The W7 weighs 6.8 kg."
    spans = [Span(2, b.index("6.8"), b.index("6.8") + 6, 6.0), Span(1, a.index("118"), a.index("118") + 6, 5.5)]
    provider = _local(spans)
    text = "".join(
        e.text for e in provider.answer("How heavy is the H2?", [_src(a, 8.0), _src(b, -3.0, page=2)], "factoid")
    )
    assert text.startswith("**Answer:** 118 kg [1]")


def test_summary_and_explain_routing_quote_and_hint():
    chunks = [_src("First idea here. Second idea. Third.", page=1), _src("Another page starts. More.", page=2)]
    summary = list(_local([]).answer("Summarize this", chunks, "summary"))
    text = "".join(e.text for e in summary if e.kind == "token")
    assert (
        text.startswith("**Excerpt summary**")
        and "> First idea here. [1]" in text
        and "> Another page starts. [2]" in text
    )
    assert any(e.kind == "meta" and "switch to Claude API or Ollama" in e.data["hint"] for e in summary)
    explain = list(_local([]).answer("why another page more", chunks, "explain"))
    assert any(e.kind == "meta" and "hint" in e.data for e in explain)


def test_helpers():
    assert answer_type_ok("How much does it cost?", "$4,990") and not answer_type_ok("What is the price?", "118 kg")
    assert not answer_type_ok("How many units?", "a battery") and answer_type_ok("How many units?", "four")
    sents = split_sentences("  A b. C d!  Ef", 3)
    assert [s.text for s in sents] == ["A b.", "C d!", "Ef"] and sents[1].source == 3
    assert "  A b. C d!  Ef"[sents[1].start : sents[1].end] == "C d!"


@pytest.mark.skipif(os.environ.get("SKIP_MODEL_TESTS") == "1", reason="model download disabled")
def test_extractive_on_fixture_pdf_with_a_real_qa_model(sample_pdf):
    """Real SQuAD 2.0 model on the fixture PDF: exact span + page, and no answer for an unrelated question."""
    pytest.importorskip("transformers")
    from app.generation.reader import ExtractiveReader
    from app.ingest.chunker import chunk_pages
    from app.ingest.cleaner import clean_pages
    from app.ingest.loader import load_pdf

    try:
        reader = ExtractiveReader(os.environ.get("QA_TEST_MODEL", "deepset/roberta-base-squad2"))
    except OSError as exc:  # offline and not cached
        pytest.skip(f"QA model not available: {exc}")
    chunks = chunk_pages(clean_pages(load_pdf(sample_pdf.read_bytes(), "h.pdf")), "d", "h.pdf")
    weight = next(c for c in chunks if "118 kg" in c.text)
    provider = LocalExtractiveProvider(reader, "real", overlap_scorer)
    events = list(
        provider.answer(
            "How heavy is the battery enclosure?", [RetrievedChunk(chunk=weight, rerank_score=5.0)], "factoid"
        )
    )
    meta = next(e.data for e in events if e.kind == "meta")
    assert meta["span"] == "118 kg" and weight.page_start == 1
    unrelated = provider.answer(
        "Who won the 1998 football world cup?", [RetrievedChunk(chunk=weight, rerank_score=5.0)], "factoid"
    )
    assert "".join(e.text for e in unrelated if e.kind == "token") == NO_ANSWER_TOKEN


# ---------- router ----------


class _Ollama:
    def __init__(self, models):
        self.models = models

    def list_models(self):
        from app.generation.llm_base import LLMError

        if self.models is None:
            raise LLMError("Ollama isn't running. Start it, or switch to Local mode.")
        return self.models


def _router(server_key="", ollama_models=None, local=True):
    local_p = _local([]) if local else None
    ollama_factory = (lambda m: _Ollama(ollama_models)) if ollama_models is not False else None
    return AnswerRouter(local_p, lambda k: FakeLLM(), server_key, ollama_factory, "llama3.2")


def test_router_auto_selection():
    assert _router().recommended(None) == AnswerMode.LOCAL  # neither key nor Ollama
    assert _router(ollama_models=["llama3.2"]).recommended(None) == AnswerMode.OLLAMA
    assert _router(ollama_models=["llama3.2"]).recommended("sk-ant-x") == AnswerMode.API  # key wins
    assert _router(server_key="srv").recommended(None) == AnswerMode.API


def test_router_respects_preference_and_falls_back():
    r = _router(ollama_models=["llama3.2"])
    assert r.pick(None, "sk-ant-x", preferred=AnswerMode.LOCAL).mode == AnswerMode.LOCAL
    assert _router().pick(None, None, preferred=AnswerMode.API).mode == AnswerMode.LOCAL  # unavailable → auto
    explicit = _router().pick(AnswerMode.API, None)
    assert explicit.provider is None and explicit.error_code == "no_api_key"
    modes = {m.mode: m for m in _router().modes(None)}
    assert modes[AnswerMode.LOCAL].available and not modes[AnswerMode.API].available
    assert "Settings" in modes[AnswerMode.API].reason and "isn't running" in modes[AnswerMode.OLLAMA].reason


# ---------- endpoints ----------


def test_modes_endpoint_and_preferences(tmp_path, sample_pdf):
    for harness in make_harness(tmp_path, sample_pdf, ollama=True):
        c = harness.client
        register(c)
        body = c.get("/chat/modes").json()
        assert body["recommended"] == "ollama" and body["ollama_models"] == ["llama3.2:latest"]
        assert {m["mode"]: m["available"] for m in body["modes"]} == {"local": True, "ollama": True, "api": False}
        assert c.get("/chat/modes", headers={"X-LLM-Key": GOOD_KEY}).json()["recommended"] == "api"
        r = c.put("/me/preferences", json={"default_mode": "local", "ollama_model": "llama3.2"}, headers=CSRF)
        assert r.status_code == 200 and r.json()["preferred_mode"] == "local"
        assert c.get("/auth/me").json()["preferred_mode"] == "local"
        assert c.put("/me/preferences", json={"default_mode": "nonsense"}, headers=CSRF).status_code == 422


def test_saved_default_mode_is_used(tmp_path, sample_pdf):
    for harness in make_harness(tmp_path, sample_pdf, ollama=True):
        c = harness.client
        register(c)
        upload(c, sample_pdf)
        c.put("/me/preferences", json={"default_mode": "local"}, headers=CSRF)
        events = ask(c, QUESTION, key=GOOD_KEY)  # key present, but the user chose Local
        assert (events[0][1]["mode"] == "local" and harness.made[-1:] == []) or all(not m.calls for m in harness.made)


def test_test_key_endpoint(h):
    c = h.client
    register(c)
    assert c.post("/chat/test-key", headers=CSRF).json() == {"ok": False, "message": "Paste your API key first."}
    assert c.post("/chat/test-key", headers=CSRF | {"X-LLM-Key": GOOD_KEY}).json()["ok"] is True
    bad = c.post("/chat/test-key", headers=CSRF | {"X-LLM-Key": "sk-ant-wrong"}).json()
    assert bad["ok"] is False and "rejected" in bad["message"]


def test_mode_and_label_persisted_on_messages(h, sample_pdf):
    c = h.client
    register(c)
    upload(c, sample_pdf)
    done = ask(c, QUESTION, key=None, mode="local")[-1][1]
    assert done["mode"] == "local" and done["label"].startswith("Extractive")
    assert done["confidence_label"] in ("High match", "Medium match", "Low match")
    msgs = c.get(f"/conversations/{done['conversation_id']}").json()["messages"]
    assert msgs[1]["mode"] == "local" and msgs[1]["label"].startswith("Extractive")


def test_local_answer_highlights_the_span_in_its_citation(h, sample_pdf):
    c = h.client
    register(c)
    upload(c, sample_pdf)
    events = ask(c, QUESTION, key=None, mode="local")
    done, cites = events[-1][1], events[-2][1]["citations"]
    lit = [x for x in cites if x["highlight"]]
    assert lit and lit[0]["text"][lit[0]["highlight"][0] : lit[0]["highlight"][1]] == done["span"]


# ---------- BYOK ----------


def test_pasted_key_is_never_persisted_or_logged(h, sample_pdf, caplog):
    secret = "sk-ant-test-good"
    c = h.client
    register(c)
    upload(c, sample_pdf)
    with caplog.at_level(logging.DEBUG):
        ask(c, QUESTION, key=secret, mode="api")
        c.post("/chat/test-key", headers=CSRF | {"X-LLM-Key": secret})
        c.get("/chat/modes", headers={"X-LLM-Key": secret})
    assert secret not in caplog.text
    data_dir = c.app.state.container.settings.data_dir
    for path in data_dir.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes(), f"key found in {path}"


# ---------- SSE contract & Ollama ----------


def _shape(events):
    names = [e for e, _ in events if e != "token"]
    return names, sorted(events[-1][1].keys())


def test_same_sse_schema_for_every_mode(tmp_path, sample_pdf):
    for harness in make_harness(tmp_path, sample_pdf, ollama=True):
        c = harness.client
        register(c)
        upload(c, sample_pdf)
        shapes = {mode: _shape(ask(c, QUESTION, mode=mode)) for mode in ("local", "ollama", "api")}
        assert shapes["local"] == shapes["ollama"] == shapes["api"]
        assert shapes["local"][0] == ["mode", "citations", "done"]


def test_ollama_mode_streams_the_model_answer(tmp_path, sample_pdf):
    for harness in make_harness(tmp_path, sample_pdf, ollama=True):
        c = harness.client
        register(c)
        upload(c, sample_pdf)
        events = ask(c, QUESTION, key=None, mode="ollama")
        assert events[0][1]["mode"] == "ollama" and "Ollama" in events[0][1]["label"]
        assert sum(e == "token" for e, _ in events) > 1  # streamed
        assert events[-1][1]["answer"] == OLLAMA_ANSWER


def test_ollama_unreachable_gives_a_friendly_error(tmp_path, sample_pdf):
    for harness in make_harness(tmp_path, sample_pdf, ollama=False):
        c = harness.client
        register(c)
        upload(c, sample_pdf)
        name, data = ask(c, QUESTION, key=None, mode="ollama")[-1]
        assert name == "error" and data["code"] == "ollama_error" and "isn't running" in data["message"]
        modes = {m["mode"]: m for m in c.get("/chat/modes").json()["modes"]}
        assert modes["ollama"]["available"] is False and "Local mode" in modes["ollama"]["reason"]
