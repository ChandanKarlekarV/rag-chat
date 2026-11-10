"""Unit tests: follow-up rewriting, the local answerer's formatting, and the judge's parsing/metrics."""

import pytest

from app.generation.answerer import LocalAnswerer
from app.generation.llm_base import LLMError
from app.generation.llm_fake import FakeLLM
from app.generation.reader import Span, format_answer, sentence_around
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


# ---------- local answerer ----------

TEXT = "The enclosure is rated IP56. It weighs 118 kg and is 1150 mm high. Up to four units can be combined."


class StubReader:
    def __init__(self, spans):
        self.spans = spans

    def best_spans(self, question, sources):
        return self.spans


def test_sentence_around_and_format_answer_quote_verbatim():
    start = TEXT.index("118 kg")
    sentence, a, b = sentence_around(TEXT, start, start + 6)
    assert sentence == "It weighs 118 kg and is 1150 mm high." and sentence[a:b] == "118 kg"
    out = format_answer(Span(1, start, start + 6, 5.0), [_src(TEXT)])
    assert out == "It weighs **118 kg** and is 1150 mm high. [1]"


def test_format_answer_keeps_space_before_bold_span():
    text = "At the end of its life, return it to the take-back programme."
    start = text.index(" return")  # span starting with whitespace, as tokenizers sometimes report
    out = format_answer(Span(1, start, text.index(" programme") + 10, 3.0), [_src(text)])
    assert "life, **return it to the take-back programme**" in out


def test_yes_no_questions_include_the_qualifying_sentence():
    text = "Mount it on concrete or brick. Plasterboard walls are not suitable. Allow four hours."
    start = text.index("concrete")
    out = format_answer(Span(1, start, start + 8, 3.0), [_src(text)], "Can I mount it on plasterboard?")
    assert out == "Mount it on **concrete** or brick. Plasterboard walls are not suitable. [1]"
    plain = format_answer(Span(1, start, start + 8, 3.0), [_src(text)], "Where do I mount it?")
    assert plain == "Mount it on **concrete** or brick. [1]"


def test_answer_type_check_rejects_wrong_kind_of_answer():
    from app.generation.answerer import answer_type_ok

    assert not answer_type_ok("How much does the H2 cost?", "118 kg")
    assert answer_type_ok("What is the price?", "$4,990")
    assert not answer_type_ok("How many employees does it have?", "a home battery")
    assert answer_type_ok("How many units can be combined?", "Up to four")
    assert answer_type_ok("How heavy is it?", "118 kg")


def test_local_answerer_prefers_the_better_ranked_passage():
    a, b = "The H2 weighs 118 kg.", "The W7 weighs 6.8 kg."
    spans = [Span(2, b.index("6.8"), b.index("6.8") + 6, 6.0), Span(1, a.index("118"), a.index("118") + 6, 5.0)]
    local = LocalAnswerer(StubReader(spans), "m", min_score=0.0, rerank_weight=1.0)
    out = "".join(local.stream("How heavy is the H2?", [_src(a, rerank=8.0), _src(b, rerank=2.0)]))
    assert "118 kg" in out and out.endswith("[1]")


def test_local_answerer_streams_answer_or_no_answer():
    start = TEXT.index("118 kg")
    good = LocalAnswerer(StubReader([Span(1, start, start + 6, 5.0)]), "m", min_score=0.0)
    assert "".join(good.stream("How heavy?", [_src(TEXT)])) == "It weighs **118 kg** and is 1150 mm high. [1]"
    unsure = LocalAnswerer(StubReader([Span(1, start, start + 6, -1.0)]), "m", min_score=0.0)
    assert "".join(unsure.stream("How heavy?", [_src(TEXT)])) == "NO_ANSWER"
    assert "".join(LocalAnswerer(StubReader([]), "m", 0.0).stream("q", [])) == "NO_ANSWER"


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
