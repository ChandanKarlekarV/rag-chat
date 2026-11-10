"""Unit tests for RRF fusion, the no-answer gate, NO_ANSWER handling and citation parsing."""

import pytest

from app.generation.answer import (
    FALLBACK_MESSAGE,
    answer_stream,
    cited_numbers,
    extract_citations,
    passes_gate,
)
from app.retrieval.bm25_store import BM25Store, tokenize
from app.retrieval.hybrid import rrf_fuse
from app.schemas import Chunk, RetrievedChunk
from tests.fakes import FakeLLM


def _rc(i: int, rerank: float | None = None, page: int = 1) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"d:{i}", doc_id="d", filename="f.pdf", page_start=page, page_end=page,
        index=i, text=f"chunk text {i}", token_count=3,
    )
    return RetrievedChunk(chunk=chunk, rerank_score=rerank)


# ---------- RRF ----------

def test_rrf_scores_match_formula():
    fused = dict(rrf_fuse([["a", "b", "c"], ["b", "d"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)
    assert fused["d"] == pytest.approx(1 / 62)


def test_rrf_rewards_agreement_between_rankings():
    order = [doc for doc, _ in rrf_fuse([["a", "b"], ["c", "b"]])]
    assert order[0] == "b"  # rank 2 in both lists beats rank 1 in only one
    assert set(order) == {"a", "b", "c"}


def test_rrf_is_deterministic_on_ties_and_handles_empty():
    assert [d for d, _ in rrf_fuse([["x", "y"], ["y", "x"]])] == ["x", "y"]
    assert rrf_fuse([[], []]) == []


# ---------- BM25 ----------

def test_bm25_search_and_doc_filter(tmp_path):
    store = BM25Store(tmp_path / "bm25.pkl")
    chunks = [
        Chunk(chunk_id="a:0", doc_id="a", filename="a.pdf", page_start=1, page_end=1, index=0,
              text="The warranty lasts twelve years.", token_count=5),
        Chunk(chunk_id="b:0", doc_id="b", filename="b.pdf", page_start=1, page_end=1, index=0,
              text="Firmware updates install at night.", token_count=5),
        Chunk(chunk_id="b:1", doc_id="b", filename="b.pdf", page_start=2, page_end=2, index=1,
              text="Clean the vents every six months.", token_count=6),
    ]
    store.rebuild(chunks)
    assert store.search("how long is the warranty", 5)[0][0] == "a:0"
    assert store.search("warranty", 5, doc_ids=["b"]) == []
    assert store.search("anything", 5, doc_ids=[]) == []
    reloaded = BM25Store(tmp_path / "bm25.pkl")
    assert reloaded.search("firmware", 5)[0][0] == "b:0"


def test_tokenize_drops_stopwords():
    assert tokenize("What is THE Warranty?") == ["warranty"]


# ---------- fallback gate ----------

def test_gate_passes_only_above_threshold():
    assert passes_gate([_rc(0, -5.0), _rc(1, 1.5)], min_score=0.0)
    assert not passes_gate([_rc(0, -5.0), _rc(1, -0.1)], min_score=0.0)
    assert passes_gate([_rc(0, 0.0)], min_score=0.0)  # boundary is inclusive


def test_gate_fails_without_results_or_scores():
    assert not passes_gate([], min_score=-100)
    assert not passes_gate([_rc(0, None)], min_score=-100)


def test_low_score_returns_fallback_without_calling_llm():
    llm = FakeLLM("should never be used")
    events = list(answer_stream("q?", [_rc(0, -9.0)], llm, min_score=0.0))
    assert llm.calls == []
    assert [e["event"] for e in events] == ["done"]
    done = events[0]["data"]
    assert done["answerable"] is False
    assert done["answer"] == FALLBACK_MESSAGE
    assert done["citations"] == []


def test_llm_no_answer_maps_to_unanswerable_and_is_never_streamed():
    llm = FakeLLM("NO_ANSWER", chunk_size=2)
    events = list(answer_stream("q?", [_rc(0, 3.0)], llm, min_score=0.0))
    assert len(llm.calls) == 1
    assert all(e["event"] != "token" for e in events)
    assert events[-1]["data"]["answerable"] is False


def test_answer_streams_tokens_and_returns_only_cited_sources():
    sources = [_rc(0, 5.0, page=1), _rc(1, 4.0, page=2), _rc(2, 3.0, page=7)]
    llm = FakeLLM("No, it lasts 12 years [3] and is transferable [1][3].")
    events = list(answer_stream("q?", sources, llm, min_score=0.0))
    streamed = "".join(e["data"]["text"] for e in events if e["event"] == "token")
    assert streamed == llm.answer
    done = events[-1]["data"]
    assert done["answerable"] is True
    assert [c["n"] for c in done["citations"]] == [1, 3]
    assert done["citations"][1]["page_start"] == 7
    assert len(done["retrieved"]) == 3
    assert "[3] (f.pdf, p.7)" in llm.calls[0][1]


def test_no_llm_configured_yields_error_event():
    events = list(answer_stream("q?", [_rc(0, 5.0)], None, min_score=0.0))
    assert events[-1]["event"] == "error"


# ---------- citation parsing ----------

def test_cited_numbers_handles_all_formats():
    assert cited_numbers("a [2] b [1][3] c [2, 4] d [10]") == [2, 1, 3, 4, 10]


def test_extract_citations_ignores_out_of_range():
    sources = [_rc(0, 1.0), _rc(1, 1.0)]
    assert [c.n for c in extract_citations("x [2] y [5] z [0]", sources)] == [2]
