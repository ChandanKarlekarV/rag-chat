"""Evaluate retrieval, the local answer model, and follow-up rewriting on tests/eval_set.json.

Prints Markdown tables:

1. Retrieval (answerable questions): Hit@1, Hit@5, MRR for vector-only, hybrid, hybrid + rerank;
   fallback accuracy of the relevance gate (all questions, rerank config only).
2. End-to-end answers with the local extractive model (no API key): a question counts as
   answered correctly when the reply contains the expected fact and cites the expected page;
   unanswerable questions count as correct when the reply is "I couldn't find this".
3. Follow-up questions: retrieval and answers with and without query rewriting.

Nothing here calls a paid API, so it is free and deterministic. For an LLM-judged evaluation of
Claude's written answers see scripts/judge.py.

Usage: python scripts/evaluate.py [--sweep]
"""

import argparse
import json
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.db.models import User
from app.db.repositories import UserRepo
from app.generation.answerer import LocalAnswerer
from app.generation.prompt import cited_numbers
from app.generation.reader import format_answer
from app.generation.rewrite import HeuristicRewriter, Turn
from app.retrieval.hybrid import Mode
from app.schemas import RetrievedChunk
from app.services.container import Container, build_container
from app.services.retrieval_service import passes_gate

CONFIGS: list[tuple[str, Mode]] = [("Vector only", "vector"), ("Hybrid (RRF)", "hybrid"), ("Hybrid + Rerank", "rerank")]


def first_hit_rank(results: list[RetrievedChunk], filename: str, page: int) -> int | None:
    """1-based rank of the first result from `filename` whose page range contains `page`."""
    for rank, r in enumerate(results, start=1):
        if r.chunk.filename == filename and r.chunk.page_start <= page <= r.chunk.page_end:
            return rank
    return None


def best_rerank(results: list[RetrievedChunk]) -> float:
    """Best rerank score among results (-inf if none)."""
    return max((r.rerank_score for r in results if r.rerank_score is not None), default=float("-inf"))


def norm(text: str) -> str:
    """Lowercase, drop Markdown bold, collapse whitespace."""
    return re.sub(r"\s+", " ", text.replace("**", "")).strip().lower()


def contains_facts(answer: str, facts: list[str]) -> bool:
    """Every fact must appear; a fact may list alternatives separated by '|'."""
    a = norm(answer)
    return all(any(norm(alt) in a for alt in fact.split("|")) for fact in facts)


@dataclass
class Outcome:
    """What the full pipeline did for one question."""

    refused: bool
    answer: str
    cited_ok: bool
    margin: float | None


def is_correct(o: Outcome, q: dict) -> bool:
    """Answerable: fact present and right page cited. Unanswerable: refused."""
    if not q.get("answer"):
        return o.refused
    return not o.refused and contains_facts(o.answer, q["answer"]) and o.cited_ok


def run_local(
    c: Container, local: LocalAnswerer, user_id: str, question: str, search: str, gate: float, q: dict | None
) -> Outcome:
    """Retrieve → gate → local reader, exactly like the app (without persisting)."""
    sources = c.retrieval.search(user_id, search).chunks
    if not passes_gate(sources, gate):
        return Outcome(True, "", False, None)
    # Decide with no threshold first so sweeps can re-apply any threshold to the same candidate.
    unbounded = LocalAnswerer(local.reader, local.model, float("-inf"), local.rerank_weight)
    chosen = unbounded.choose(question, sources)
    margin = chosen[1] if chosen else None
    if chosen is None:
        return Outcome(True, "", False, None)
    answer = format_answer(chosen[0], sources, question)
    cited_ok = False
    if q and q.get("expected_doc"):
        for n in cited_numbers(answer):
            if 1 <= n <= len(sources):
                ch = sources[n - 1].chunk
                cited_ok |= ch.filename == q["expected_doc"] and ch.page_start <= q["expected_page"] <= ch.page_end
    return Outcome(margin < local.min_score, answer, cited_ok, margin)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-set", type=Path, default=ROOT / "tests" / "eval_set.json")
    parser.add_argument("--sweep", action="store_true", help="print per-question scores and threshold sweeps")
    args = parser.parse_args()

    spec = json.loads(args.eval_set.read_text(encoding="utf-8"))
    pdfs = [ROOT / p for p in spec["documents"]]
    if missing := [p for p in pdfs if not p.exists()]:
        sys.exit(f"{missing[0]} not found. Run: python scripts/make_sample_pdf.py")
    questions = spec["questions"]
    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]
    followups = spec.get("followups", [])

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        settings = Settings(data_dir=Path(tmp), database_url="", llm_provider="local", _env_file=None)
        c = build_container(settings)
        local = c.local
        assert local is not None
        with c.db.session() as s:
            user = UserRepo(s).add(User(email="eval@example.com", password_hash="-"))
        total_chunks = 0
        for pdf in pdfs:
            doc = c.ingest.ingest_now(user, pdf.read_bytes(), pdf.name)
            total_chunks += c.ingest.status(user.id, doc.id).chunks
        print(
            f"Corpus: {len(pdfs)} PDFs, {total_chunks} chunks. Eval set: {len(answerable)} answerable + "
            f"{len(unanswerable)} unanswerable questions, {len(followups)} follow-ups.\n"
        )

        # ---------- 1. retrieval ----------
        rows, rerank_runs = [], []
        for label, mode in CONFIGS:
            hits = hits1 = 0
            rr = 0.0
            start = time.perf_counter()
            for q in questions:
                results = c.retrieval.search(user.id, q["question"], mode=mode).chunks
                if mode == "rerank":
                    rerank_runs.append((q, results))
                if q["answerable"]:
                    rank = first_hit_rank(results, q["expected_doc"], q["expected_page"])
                    hits += rank is not None
                    hits1 += rank == 1
                    rr += 1.0 / rank if rank else 0.0
            ms = (time.perf_counter() - start) * 1000 / len(questions)
            fb = "n/a"
            if mode == "rerank":
                ok = sum(passes_gate(res, settings.min_rerank_score) == q["answerable"] for q, res in rerank_runs)
                fb = f"{ok / len(questions):.2f}"
            n = len(answerable)
            rows.append((label, hits1 / n, hits / n, rr / n, fb, ms))
        print(f"### Retrieval (MIN_RERANK_SCORE={settings.min_rerank_score})\n")
        print("| Config | Hit@1 | Hit@5 | MRR | Fallback accuracy | Latency / query (CPU) |\n|---|---|---|---|---|---|")
        for label, h1, h5, mrr, fb, ms in rows:
            print(f"| {label} | {h1:.2f} | {h5:.2f} | {mrr:.2f} | {fb} | {ms:.0f} ms |")

        # ---------- 2. end-to-end local answers ----------
        gate = settings.min_rerank_score
        started = time.perf_counter()
        outcomes = {
            q["question"]: run_local(c, local, user.id, q["question"], q["question"], gate, q) for q in questions
        }
        e2e_ms = (time.perf_counter() - started) * 1000 / len(questions)
        correct = sum(is_correct(outcomes[q["question"]], q) for q in answerable)
        false_refusals = sum(outcomes[q["question"]].refused for q in answerable)
        wrong = len(answerable) - correct - false_refusals
        refused_ok = sum(outcomes[q["question"]].refused for q in unanswerable)
        overall = (correct + refused_ok) / len(questions)
        print(
            f"\n### End-to-end answers: local model `{settings.reader_model}`, no API key "
            f"(READER_MIN_SCORE={settings.reader_min_score})\n"
        )
        print(
            "| Answerable: correct fact + page | Answered wrongly | Refused but answerable "
            "| Unanswerable: correctly refused | Overall accuracy | Latency / question (CPU) |"
        )
        print("|---|---|---|---|---|---|")
        print(
            f"| {correct}/{len(answerable)} ({correct / len(answerable):.0%}) | {wrong} | {false_refusals} "
            f"| {refused_ok}/{len(unanswerable)} | {overall:.0%} | {e2e_ms:.0f} ms |"
        )

        # ---------- 3. follow-ups ----------
        if followups:
            rewriter = HeuristicRewriter()
            table = []
            for label, use_rewrite in (("Follow-up as typed", False), ("With query rewriting", True)):
                hits = answered = 0
                for f in followups:
                    first_sources = c.retrieval.search(user.id, f["first"]).chunks
                    history = [Turn(f["first"], "".join(local.stream(f["first"], first_sources)))]
                    search = rewriter.rewrite(f["followup"], history).search_query if use_rewrite else f["followup"]
                    res = c.retrieval.search(user.id, search).chunks
                    hits += first_hit_rank(res, f["expected_doc"], f["expected_page"]) is not None
                    answered += is_correct(run_local(c, local, user.id, f["followup"], search, gate, f), f)
                table.append((label, hits / len(followups), answered / len(followups)))
            print("\n### Follow-up questions (heuristic rewriting, no API key)\n")
            print("| Config | Hit@5 | Answer correct (fact + page) |\n|---|---|---|")
            for label, h5, acc in table:
                print(f"| {label} | {h5:.2f} | {acc:.2f} |")

        # ---------- sweeps ----------
        if args.sweep:
            print("\nPer question: rerank score, reader margin, outcome")
            for q, res in rerank_runs:
                o = outcomes[q["question"]]
                m = "—" if o.margin is None else f"{o.margin:.2f}"
                tag = "A" if q["answerable"] else "U"
                ok = is_correct(o, q)
                print(f"  {tag} {best_rerank(res):7.2f} {m:>7}  {'ok  ' if ok else 'MISS'} {q['question']}")
                if not ok:
                    print(f"        -> {o.answer or '(refused)'}")
            print(
                "\n| MIN_RERANK_SCORE | Unanswerable refused | Answerable passed | Gate accuracy |\n|---|---|---|---|"
            )
            for t in [-6, -5, -4, -3, -2, -1, 0]:
                refused = sum(not passes_gate(r, t) for q, r in rerank_runs if not q["answerable"])
                passed = sum(passes_gate(r, t) for q, r in rerank_runs if q["answerable"])
                acc = (refused + passed) / len(questions)
                print(f"| {t} | {refused}/{len(unanswerable)} | {passed}/{len(answerable)} | {acc:.2f} |")
            print("\n| READER_MIN_SCORE | Answerable correct | Unanswerable refused | Overall |\n|---|---|---|---|")
            for t in [-6, -5, -4, -3, -2, -1, 0, 1, 2, 4]:
                ok_a = ok_u = 0
                for q in questions:
                    o = outcomes[q["question"]]
                    answered = o.margin is not None and o.margin >= t
                    if q["answerable"]:
                        ok_a += answered and contains_facts(o.answer, q["answer"]) and o.cited_ok
                    else:
                        ok_u += not answered
                print(
                    f"| {t} | {ok_a}/{len(answerable)} | {ok_u}/{len(unanswerable)} "
                    f"| {(ok_a + ok_u) / len(questions):.0%} |"
                )


if __name__ == "__main__":
    main()
