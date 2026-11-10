"""Compare retrieval configs on `tests/eval_set.json` and print a Markdown table.

Metrics (answerable questions):
  Hit@5  - share of questions whose expected page lies within a top-5 chunk's page range.
  MRR    - mean of 1/rank of the first such chunk (0 if none in the top 5).
Fallback (all questions, hybrid+rerank only, since the gate thresholds the rerank score):
  accuracy of the gate decision: unanswerable -> refused, answerable -> passed to the LLM.

The evaluation is retrieval-only and never calls the LLM, so it is free and deterministic.

Usage: python scripts/evaluate.py [--sweep]
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.db.models import User
from app.db.repositories import UserRepo
from app.retrieval.hybrid import Mode
from app.schemas import RetrievedChunk
from app.services.container import build_container
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-set", type=Path, default=ROOT / "tests" / "eval_set.json")
    parser.add_argument("--sweep", action="store_true", help="also print gate accuracy for several thresholds")
    args = parser.parse_args()

    spec = json.loads(args.eval_set.read_text(encoding="utf-8"))
    pdfs = [ROOT / p for p in spec["documents"]]
    if missing := [p for p in pdfs if not p.exists()]:
        sys.exit(f"{missing[0]} not found. Run: python scripts/make_sample_pdf.py")
    questions = spec["questions"]
    answerable = [q for q in questions if q["answerable"]]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        settings = Settings(data_dir=Path(tmp))
        services = build_container(settings)
        with services.db.session() as s:
            user = UserRepo(s).add(User(email="eval@example.com", password_hash="-"))
        total_chunks = 0
        for pdf in pdfs:
            doc = services.ingest.ingest_now(user, pdf.read_bytes(), pdf.name)
            status = services.ingest.status(user.id, doc.id)
            total_chunks += status.chunks
            print(f"Indexed {doc.filename}: {status.pages} pages, {status.chunks} chunks.", file=sys.stderr)

        rows = []
        rerank_runs: list[tuple[dict, list[RetrievedChunk]]] = []
        for label, mode in CONFIGS:
            hits, hits1, rr = 0, 0, 0.0
            start = time.perf_counter()
            for q in questions:
                results = services.retrieval.search(user.id, q["question"], mode=mode).chunks
                if mode == "rerank":
                    rerank_runs.append((q, results))
                if q["answerable"]:
                    rank = first_hit_rank(results, q["expected_doc"], q["expected_page"])
                    hits += rank is not None
                    hits1 += rank == 1
                    rr += 1.0 / rank if rank else 0.0
            ms = (time.perf_counter() - start) * 1000 / len(questions)
            fallback = "n/a"
            if mode == "rerank":
                correct = sum(passes_gate(res, settings.min_rerank_score) == q["answerable"] for q, res in rerank_runs)
                fallback = f"{correct / len(questions):.2f}"
            n = len(answerable)
            rows.append((label, hits1 / n, hits / n, rr / n, fallback, ms))

    print(
        f"Corpus: {len(pdfs)} PDFs, {total_chunks} chunks. Eval set: {len(answerable)} answerable + "
        f"{len(questions) - len(answerable)} unanswerable questions. MIN_RERANK_SCORE={settings.min_rerank_score}\n"
    )
    print("| Config | Hit@1 | Hit@5 | MRR | Fallback accuracy | Latency / query (CPU) |")
    print("|---|---|---|---|---|---|")
    for label, hit1, hit, mrr, fb, ms in rows:
        print(f"| {label} | {hit1:.2f} | {hit:.2f} | {mrr:.2f} | {fb} | {ms:.0f} ms |")

    if args.sweep:
        print("\nBest rerank score per question:")
        for q, res in rerank_runs:
            tag = "A" if q["answerable"] else "U"
            rank = first_hit_rank(res, q["expected_doc"], q["expected_page"]) if q["answerable"] else None
            hit = "" if not q["answerable"] else f"  (hit@{rank})" if rank else "  (MISS)"
            print(f"  {tag} {best_rerank(res):7.2f}  {q['question']}{hit}")
        print("\n| MIN_RERANK_SCORE | Unanswerable refused | Answerable passed | Accuracy |")
        print("|---|---|---|---|")
        unans = [r for q, r in rerank_runs if not q["answerable"]]
        ans = [r for q, r in rerank_runs if q["answerable"]]
        for t in [-6, -5, -4, -3, -2, -1, 0, 1, 2]:
            refused = sum(not passes_gate(r, t) for r in unans)
            passed = sum(passes_gate(r, t) for r in ans)
            print(f"| {t} | {refused}/{len(unans)} | {passed}/{len(ans)} | {(refused + passed) / len(questions):.2f} |")


if __name__ == "__main__":
    main()
