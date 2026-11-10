"""Evaluate retrieval, every answer mode, and follow-up rewriting on tests/eval_set.json.

Prints Markdown tables (ANSWER_MODES.md section 8):

1. Retrieval: Hit@1, Hit@5, MRR for vector-only, hybrid, hybrid + rerank; gate fallback accuracy.
2. Answers per mode (Local always; Ollama when reachable; API when ANTHROPIC_API_KEY is set,
   otherwise "skipped"):
   - Correct: the reply contains the expected fact AND cites the expected page.
   - Contains gold: the short gold answer appears in the reply.
   - F1 / EM: SQuAD-style token F1 and exact match of the prediction against `gold_answer`
     (prediction = the extracted span in Local mode, the whole reply otherwise).
   - Citation acc.: the first cited source is on the expected page.
   - Fallback acc.: unanswerable questions correctly refused.
   - Median latency per question (CPU).
3. Follow-ups (Local mode): retrieval and answers with and without query rewriting.

Usage: python scripts/evaluate.py [--sweep] [--eval-set tests/heldout_set.json]
"""

import argparse
import json
import os
import re
import statistics
import string
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.db.models import User
from app.db.repositories import UserRepo
from app.generation.base import AnswerMode, AnswerProvider
from app.generation.extractive import LocalExtractiveProvider
from app.generation.prompt import cited_numbers, is_no_answer
from app.generation.question_type import classify
from app.generation.rewrite import HeuristicRewriter, Turn
from app.retrieval.hybrid import Mode
from app.schemas import RetrievedChunk
from app.services.container import Container, build_container
from app.services.retrieval_service import passes_gate

CONFIGS: list[tuple[str, Mode]] = [("Vector only", "vector"), ("Hybrid (RRF)", "hybrid"), ("Hybrid + Rerank", "rerank")]


# ---------- metrics ----------


def first_hit_rank(results: list[RetrievedChunk], filename: str, page: int) -> int | None:
    """1-based rank of the first result from `filename` whose page range contains `page`."""
    for rank, r in enumerate(results, start=1):
        if r.chunk.filename == filename and r.chunk.page_start <= page <= r.chunk.page_end:
            return rank
    return None


def norm(text: str) -> str:
    """Lowercase, drop Markdown, collapse whitespace (for substring checks)."""
    return re.sub(r"\s+", " ", text.replace("**", "").replace("> ", " ")).strip().lower()


def contains_facts(answer: str, facts: list[str]) -> bool:
    """Every fact must appear; a fact may list alternatives separated by '|'."""
    a = norm(answer)
    return all(any(norm(alt) in a for alt in fact.split("|")) for fact in facts)


def squad_normalize(text: str) -> str:
    """SQuAD answer normalisation: lowercase, strip punctuation, articles and extra spaces."""
    text = re.sub(r"\[\d+\]", " ", text.replace("**", ""))
    text = "".join(ch for ch in text.lower() if ch not in set(string.punctuation) - {"-"})
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> float:
    """1.0 if normalised strings are identical."""
    return float(squad_normalize(pred) == squad_normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    """SQuAD token-overlap F1."""
    p, g = squad_normalize(pred).split(), squad_normalize(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if not p or not g or not common:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


@dataclass
class Outcome:
    """What the full pipeline did for one question in one mode."""

    refused: bool
    answer: str
    prediction: str
    cited_ok: bool
    first_cite_ok: bool
    qa: float | None
    latency_ms: float


def run(
    c: Container, provider: AnswerProvider, user_id: str, question: str, search: str, gate: float, q: dict
) -> Outcome:
    """Retrieve → shared gate → provider, exactly like the app (without persisting)."""
    t = time.perf_counter()
    sources = c.retrieval.search(user_id, search).chunks
    if not passes_gate(sources, gate):
        return Outcome(True, "", "", False, False, None, (time.perf_counter() - t) * 1000)
    text, meta = "", {}
    for ev in provider.answer(question, sources, classify(question)):
        if ev.kind == "meta":
            meta |= ev.data
        else:
            text += ev.text
    ms = (time.perf_counter() - t) * 1000
    qa = meta.get("confidence")
    if is_no_answer(text):
        return Outcome(True, "", "", False, False, qa, ms)
    cites = [n for n in cited_numbers(text) if 1 <= n <= len(sources)]

    def on_page(n: int) -> bool:
        ch = sources[n - 1].chunk
        return (
            bool(q.get("expected_doc"))
            and ch.filename == q["expected_doc"]
            and ch.page_start <= q["expected_page"] <= ch.page_end
        )

    prediction = meta.get("span") or re.sub(r"\[\d+\]", "", text)
    return Outcome(False, text, prediction, any(on_page(n) for n in cites), bool(cites) and on_page(cites[0]), qa, ms)


def is_correct(o: Outcome, q: dict) -> bool:
    """Answerable: fact present and right page cited. Unanswerable: refused."""
    if not q["answerable"]:
        return o.refused
    return not o.refused and contains_facts(o.answer, q["answer"]) and o.cited_ok


# ---------- main ----------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-set", type=Path, default=ROOT / "tests" / "eval_set.json")
    parser.add_argument("--sweep", action="store_true", help="print per-question outcomes and a MIN_QA_SCORE sweep")
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
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        settings = Settings(data_dir=Path(tmp), database_url="", anthropic_api_key=key, redis_url="", chroma_url="")
        c = build_container(settings)
        with c.db.session() as s:
            user = UserRepo(s).add(User(email="eval@example.com", password_hash="-"))
        total_chunks = 0
        for pdf in pdfs:
            doc = c.ingest.ingest_now(user, pdf.read_bytes(), pdf.name)
            total_chunks += c.ingest.status(user.id, doc.id).chunks
        gate = settings.min_rerank_score
        print(
            f"Corpus: {len(pdfs)} PDFs, {total_chunks} chunks. Eval set: {len(answerable)} answerable + "
            f"{len(unanswerable)} unanswerable questions, {len(followups)} follow-ups.\n"
        )

        # ---------- 1. retrieval ----------
        print(f"### Retrieval (MIN_RERANK_SCORE={gate})\n")
        print(
            "| Config | Hit@1 | Hit@5 | MRR | Gate fallback acc. | Latency / query (CPU) |\n|---|---|---|---|---|---|"
        )
        for label, mode in CONFIGS:
            hits = hits1 = 0
            rr = 0.0
            gate_ok = 0
            start = time.perf_counter()
            for q in questions:
                results = c.retrieval.search(user.id, q["question"], mode=mode).chunks
                gate_ok += passes_gate(results, gate) == q["answerable"]
                if q["answerable"]:
                    rank = first_hit_rank(results, q["expected_doc"], q["expected_page"])
                    hits += rank is not None
                    hits1 += rank == 1
                    rr += 1.0 / rank if rank else 0.0
            ms = (time.perf_counter() - start) * 1000 / len(questions)
            n = len(answerable)
            fb = f"{gate_ok / len(questions):.2f}" if mode == "rerank" else "n/a"
            print(f"| {label} | {hits1 / n:.2f} | {hits / n:.2f} | {rr / n:.2f} | {fb} | {ms:.0f} ms |")

        # ---------- 2. answers per mode ----------
        print(f"\n### Answers per mode (MIN_QA_SCORE={settings.min_qa_score}, QA model `{settings.qa_model}`)\n")
        print(
            "| Mode | Correct (fact + page) | Contains gold | F1 | EM | Citation acc. "
            "| Fallback acc. | Median latency |"
        )
        print("|---|---|---|---|---|---|---|---|")
        local_outcomes: dict[str, Outcome] = {}
        for mode in (AnswerMode.LOCAL, AnswerMode.OLLAMA, AnswerMode.API):
            available = {m.mode: m for m in c.router.modes(None)}[mode]
            if not available.available:
                print(f"| {mode.value} | skipped: {available.reason} | | | | | | |")
                continue
            provider = c.router.pick(mode, None).provider
            outcomes = {
                q["question"]: run(c, provider, user.id, q["question"], q["question"], gate, q) for q in questions
            }
            if mode == AnswerMode.LOCAL:
                local_outcomes = outcomes
            ans = [(q, outcomes[q["question"]]) for q in answerable]
            correct = sum(is_correct(o, q) for q, o in ans)
            gold = sum(not o.refused and norm(q["gold_answer"]) in norm(o.answer) for q, o in ans)
            f1 = statistics.mean(token_f1(o.prediction, q["gold_answer"]) if not o.refused else 0.0 for q, o in ans)
            em = statistics.mean(exact_match(o.prediction, q["gold_answer"]) if not o.refused else 0.0 for q, o in ans)
            cite = sum(o.first_cite_ok for _, o in ans) / len(ans)
            refused = sum(outcomes[q["question"]].refused for q in unanswerable)
            latency = statistics.median(o.latency_ms for o in outcomes.values())
            print(
                f"| {mode.value} | {correct}/{len(ans)} ({correct / len(ans):.0%}) | {gold}/{len(ans)} | {f1:.2f} "
                f"| {em:.2f} | {cite:.2f} | {refused}/{len(unanswerable)} | {latency:.0f} ms |"
            )

        # ---------- 3. follow-ups ----------
        local = c.router.local
        if followups and local is not None:
            rewriter = HeuristicRewriter()
            print("\n### Follow-up questions (Local mode, heuristic rewriting)\n")
            print("| Config | Hit@5 | Answer correct (fact + page) |\n|---|---|---|")
            for label, use_rewrite in (("Follow-up as typed", False), ("With query rewriting", True)):
                hits = answered = 0
                for f in followups:
                    f = f | {"answerable": True}
                    first = run(c, local, user.id, f["first"], f["first"], gate, {})
                    history = [Turn(f["first"], first.answer)]
                    search = rewriter.rewrite(f["followup"], history).search_query if use_rewrite else f["followup"]
                    res = c.retrieval.search(user.id, search).chunks
                    hits += first_hit_rank(res, f["expected_doc"], f["expected_page"]) is not None
                    answered += is_correct(run(c, local, user.id, f["followup"], search, gate, f), f)
                print(f"| {label} | {hits / len(followups):.2f} | {answered / len(followups):.2f} |")

        # ---------- sweep ----------
        if args.sweep and isinstance(local, LocalExtractiveProvider):
            unbounded = LocalExtractiveProvider(
                local.reader, local.name, local.scorer, 0.0, local.qa_weight, local.max_sentences
            )
            raw = {q["question"]: run(c, unbounded, user.id, q["question"], q["question"], gate, q) for q in questions}
            print("\nPer question (Local mode): qa_score, outcome")
            for q in questions:
                o = local_outcomes.get(q["question"], raw[q["question"]])
                qa = "—" if raw[q["question"]].qa is None else f"{raw[q['question']].qa:.3f}"
                tag = "A" if q["answerable"] else "U"
                print(f"  {tag} {qa:>6} {'ok  ' if is_correct(o, q) else 'MISS'} {q['question']}")
                if not is_correct(o, q):
                    print(f"         -> {(o.answer or '(refused)')[:200]!r}")
            print("\n| MIN_QA_SCORE | Answerable correct | Unanswerable refused | Overall |\n|---|---|---|---|")
            for t in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
                ok_a = ok_u = 0
                for q in questions:
                    o = raw[q["question"]]
                    answered = not o.refused and o.qa is not None and o.qa >= t
                    if q["answerable"]:
                        ok_a += answered and contains_facts(o.answer, q["answer"]) and o.cited_ok
                    else:
                        ok_u += not answered
                overall = (ok_a + ok_u) / len(questions)
                print(f"| {t} | {ok_a}/{len(answerable)} | {ok_u}/{len(unanswerable)} | {overall:.0%} |")


if __name__ == "__main__":
    main()
