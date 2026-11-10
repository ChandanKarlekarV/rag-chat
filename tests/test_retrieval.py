"""Unit tests: RRF fusion, per-user BM25, the relevance gate, NO_ANSWER handling, citation parsing."""

import pytest

from app.generation.llm_fake import FakeLLM
from app.generation.prompt import build_user_prompt, cited_numbers, could_become_no_answer, is_no_answer
from app.retrieval.bm25_store import BM25Store, tokenize
from app.retrieval.hybrid import rrf_fuse
from app.schemas import Chunk, RetrievedChunk
from app.services.chat_service import extract_citations
from app.services.retrieval_service import passes_gate


def _rc(i: int, rerank: float | None = None, page: int = 1) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"d:{i}", doc_id="d", filename="f.pdf", page_start=page, page_end=page,
        index=i, text=f"chunk text {i}. Second sentence here.", token_count=3,
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

def _chunk(cid: str, doc: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, doc_id=doc, filename=f"{doc}.pdf", page_start=1, page_end=1, index=0, text=text, token_count=5)


def test_bm25_is_per_user_and_filters_docs(tmp_path):
    store = BM25Store(tmp_path)
    store.rebuild("alice", [
        _chunk("a:0", "a", "The warranty lasts twelve years."),
        _chunk("a:1", "a2", "Vents need cleaning."),
        _chunk("a:2", "a2", "The status light is green."),
    ])
    store.rebuild("bob", [_chunk("b:0", "b", "Firmware updates install at night.")])
    assert store.search("alice", "how long is the warranty", 5)[0][0] == "a:0"
    assert store.search("bob", "warranty", 5) == []  # Bob never sees Alice's corpus
    assert store.search("alice", "warranty", 5, doc_ids=["a2"]) == []
    assert store.search("alice", "anything", 5, doc_ids=[]) == []
    assert BM25Store(tmp_path).search("bob", "firmware", 5)[0][0] == "b:0"  # persisted
    store.delete_user("alice")
    assert store.search("alice", "warranty", 5) == []


def test_bm25_rejects_unsafe_user_ids(tmp_path):
    with pytest.raises(ValueError):
        BM25Store(tmp_path).rebuild("../evil", [])


def test_tokenize_drops_stopwords():
    assert tokenize("What is THE Warranty?") == ["warranty"]


# ---------- gate ----------

def test_gate_passes_only_above_threshold():
    assert passes_gate([_rc(0, -5.0), _rc(1, 1.5)], min_score=0.0)
    assert not passes_gate([_rc(0, -5.0), _rc(1, -0.1)], min_score=0.0)
    assert passes_gate([_rc(0, 0.0)], min_score=0.0)  # boundary is inclusive


def test_gate_fails_without_results_or_scores():
    assert not passes_gate([], min_score=-100)
    assert not passes_gate([_rc(0, None)], min_score=-100)


# ---------- NO_ANSWER ----------

@pytest.mark.parametrize("text", ["NO_ANSWER", " no_answer.", "NO_ANSWER\n"])
def test_is_no_answer(text):
    assert is_no_answer(text)


def test_prefix_holdback():
    assert could_become_no_answer("NO_")
    assert could_become_no_answer("  n")
    assert not could_become_no_answer("Nope, ")
    assert not could_become_no_answer("The answer")


# ---------- citations ----------

def test_cited_numbers_handles_all_formats():
    assert cited_numbers("a [2] b [1][3] c [2, 4] d [10]") == [2, 1, 3, 4, 10]


def test_extract_citations_ignores_out_of_range():
    sources = [_rc(0, 1.0), _rc(1, 1.0, page=7)]
    cites = extract_citations("x [2] y [5] z [0]", sources)
    assert [c.n for c in cites] == [2] and cites[0].page_start == 7


def test_prompt_marks_sources_as_untrusted_data():
    prompt = build_user_prompt("q?", [_rc(0, 1.0, page=3)])
    assert "untrusted" in prompt and "<sources>" in prompt and "[1] (f.pdf, p.3)" in prompt


def test_fake_llm_is_extractive_and_cites_source_one():
    prompt = build_user_prompt("q?", [_rc(0, 1.0), _rc(1, 1.0)])
    answer = "".join(FakeLLM().stream("sys", prompt))
    assert answer.endswith("[1]") and "chunk text 0" in answer
