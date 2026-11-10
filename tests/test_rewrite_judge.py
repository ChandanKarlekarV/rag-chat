"""Unit tests: follow-up rewriting, the local answerer's formatting, and the judge's parsing/metrics."""

import pytest

from app.generation.fake import FakeLLM
from app.generation.llm_base import LLMError
from app.generation.rewrite import HeuristicRewriter, LLMRewriter, Turn, needs_context
from app.schemas import Chunk, RetrievedChunk
from scripts.judge import Verdict, judge_prompt, parse_verdict, summarize

HISTORY = [Turn("How much usable energy can one H2 store?", "11.2 kWh [1]")]


def _src(text: str, rerank: float = 1.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id="d:0", doc_id="d", filename="f.pdf", page_start=1, page_end=1, index=0, text=text, token_count=1
        ),
        rerank_score=rerank,
    )


# ---------- rewriting ----------


@pytest.mark.parametrize("q", ["And how heavy is it?", "What about error E-101?", "Can it be transferred?", "Price?"])
def test_needs_context_detects_follow_ups(q):
    assert needs_context(q)


@pytest.mark.parametrize("q", ["How heavy is the battery enclosure?", "Which port does the Modbus interface use?"])
def test_needs_context_ignores_standalone_questions(q):
    assert not needs_context(q)


def test_heuristic_rewriter_expands_search_but_keeps_question():
    rw = HeuristicRewriter().rewrite("And how heavy is it?", HISTORY)
    assert rw.rewritten and rw.question == "And how heavy is it?"
    assert "H2" in rw.search_query and "usable" in rw.search_query
    assert rw.search_query.startswith("And how heavy is it?")


def test_heuristic_rewriter_no_history_or_standalone_is_unchanged():
    r = HeuristicRewriter()
    assert not r.rewrite("And how heavy is it?", []).rewritten
    assert not r.rewrite("How heavy is the battery enclosure?", HISTORY).rewritten


def test_llm_rewriter_uses_model_output():
    llm = FakeLLM(completion="How heavy is the Halcyon H2 battery?")
    rw = LLMRewriter(llm).rewrite("And how heavy is it?", HISTORY)
    assert rw.rewritten and rw.question == rw.search_query == "How heavy is the Halcyon H2 battery?"
    assert "<conversation>" in llm.calls[0][1] and "11.2 kWh" in llm.calls[0][1]


def test_llm_rewriter_falls_back_on_error_or_garbage():
    class Broken(FakeLLM):
        def complete(self, system, user, max_tokens=512):
            raise LLMError("down")

    assert HeuristicRewriter().rewrite("And how heavy is it?", HISTORY) == LLMRewriter(Broken()).rewrite(
        "And how heavy is it?", HISTORY
    )
    multi = LLMRewriter(FakeLLM(completion="line one\nline two")).rewrite("And how heavy is it?", HISTORY)
    assert multi.question == "And how heavy is it?"


# ---------- judge ----------


def test_parse_verdict_tolerates_fences():
    v = parse_verdict('```json\n{"correct": true, "faithful": true, "refused": false, "reason": "ok"}\n```')
    assert v == Verdict(True, True, False, "ok")
    with pytest.raises(ValueError):
        parse_verdict("no json here")


def test_judge_prompt_and_summary():
    p = judge_prompt("q?", ["12 years|twelve years"], [_src("passage")], "answer [1]")
    assert "12 years or twelve years" in p and "[1] passage" in p
    rows = [
        ({"answerable": True}, Verdict(True, True, False, "")),
        ({"answerable": True}, Verdict(False, False, False, "")),
        ({"answerable": False}, Verdict(False, True, True, "")),
        ({"answerable": False}, Verdict(False, False, False, "")),
    ]
    m = summarize(rows)
    assert m == {"correct": 0.5, "faithful": 0.5, "refusal_accuracy": 0.75, "unanswerable_refused": 0.5}
