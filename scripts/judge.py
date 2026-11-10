"""LLM-as-judge evaluation of full answers (retrieval + gate + answerer) on tests/eval_set.json.

For every question the pipeline produces an answer exactly as the app would. A judge model then
grades it against the reference facts and the cited passages:

* correct   - answers the question and agrees with the reference facts
* faithful  - every claim is supported by the cited passages (no outside knowledge)
* refusal   - for unanswerable questions, did the system say it couldn't find the answer?

Requires an Anthropic API key (it makes paid calls): ANTHROPIC_API_KEY=... python scripts/judge.py
Options: --answerer local|ollama|api  (which mode answers; the judge is always Claude)
         --judge-model MODEL       (default claude-opus-5-5)
         --limit N                 (only the first N questions, to try it cheaply)
"""

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.db.models import User
from app.db.repositories import UserRepo
from app.generation.base import AnswerMode, AnswerProvider
from app.generation.llm_anthropic import AnthropicLLM
from app.generation.llm_base import LLMClient, LLMError
from app.generation.prompt import cited_numbers, is_no_answer
from app.generation.question_type import classify
from app.schemas import RetrievedChunk
from app.services.container import build_container
from app.services.retrieval_service import passes_gate

JUDGE_SYSTEM = """You grade answers produced by a document question-answering system.
You are given the question, reference facts written by a human (may be empty if the question is
not answerable from the documents), the passages the answer cited, and the answer.
Return ONLY a JSON object with these keys:
  "correct":  true if the answer correctly answers the question and agrees with the reference facts
              (false if it is wrong, incomplete on the key fact, or a refusal of an answerable question),
  "faithful": true if every claim in the answer is supported by the cited passages, else false
              (a refusal counts as faithful),
  "refused":  true if the answer says the information could not be found,
  "reason":   one short sentence.
Judge only what is written. Do not reward answers that use knowledge outside the passages."""


@dataclass(frozen=True)
class Verdict:
    """The judge's grades for one answer."""

    correct: bool
    faithful: bool
    refused: bool
    reason: str


def parse_verdict(text: str) -> Verdict:
    """Parse the judge's JSON (tolerating code fences or surrounding prose)."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"judge returned no JSON: {text[:200]!r}")
    data = json.loads(match.group(0))
    return Verdict(bool(data["correct"]), bool(data["faithful"]), bool(data["refused"]), str(data.get("reason", "")))


def judge_prompt(question: str, facts: list[str], cited: list[RetrievedChunk], answer: str) -> str:
    """Render the material the judge sees."""
    passages = "\n\n".join(f"[{i}] {c.chunk.text}" for i, c in enumerate(cited, start=1)) or "(none)"
    reference = "; ".join(f.replace("|", " or ") for f in facts) or "(not answerable from the documents)"
    return (
        f"QUESTION: {question}\n\nREFERENCE FACTS: {reference}\n\n<cited_passages>\n{passages}\n</cited_passages>"
        f"\n\nANSWER: {answer}"
    )


def summarize(rows: list[tuple[dict, Verdict]]) -> dict[str, float]:
    """Aggregate metrics: correctness on answerable, faithfulness overall, refusal accuracy."""
    ans = [v for q, v in rows if q["answerable"]]
    una = [v for q, v in rows if not q["answerable"]]
    refusal_ok = sum(not v.refused for v in ans) + sum(v.refused for v in una)
    return {
        "correct": sum(v.correct for v in ans) / max(len(ans), 1),
        "faithful": sum(v.faithful for _, v in rows) / max(len(rows), 1),
        "refusal_accuracy": refusal_ok / max(len(rows), 1),
        "unanswerable_refused": sum(v.refused for v in una) / max(len(una), 1),
    }


def run_pipeline(
    c, provider: AnswerProvider, user_id: str, question: str, gate: float
) -> tuple[str, list[RetrievedChunk]]:
    """Answer one question like the app does; returns (answer text, cited passages)."""
    sources = c.retrieval.search(user_id, question).chunks
    if not passes_gate(sources, gate):
        return "I couldn't find this in your documents.", []
    events = provider.answer(question, sources, classify(question))
    text = "".join(e.text for e in events if e.kind == "token").strip()
    if is_no_answer(text):
        return "I couldn't find this in your documents.", []
    cited = [sources[n - 1] for n in cited_numbers(text) if 1 <= n <= len(sources)]
    return text, cited


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answerer", choices=["local", "ollama", "api"], default="local")
    parser.add_argument("--judge-model", default="claude-opus-5-5")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        print("Set ANTHROPIC_API_KEY to run the LLM judge (it makes paid API calls).", file=sys.stderr)
        return 2
    spec = json.loads((ROOT / "tests" / "eval_set.json").read_text(encoding="utf-8"))
    questions = spec["questions"][: args.limit or None]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        settings = Settings(data_dir=Path(tmp), database_url="", anthropic_api_key=key)
        c = build_container(settings)
        choice = c.router.pick(AnswerMode(args.answerer), key)
        if choice.provider is None:
            print(choice.error_message, file=sys.stderr)
            return 2
        answerer: AnswerProvider = choice.provider
        judge: LLMClient = AnthropicLLM(key, args.judge_model)
        with c.db.session() as s:
            user = UserRepo(s).add(User(email="judge@example.com", password_hash="-"))
        for p in spec["documents"]:
            c.ingest.ingest_now(user, (ROOT / p).read_bytes(), Path(p).name)

        rows: list[tuple[dict, Verdict]] = []
        for q in questions:
            answer, cited = run_pipeline(c, answerer, user.id, q["question"], settings.min_rerank_score)
            try:
                verdict = parse_verdict(
                    judge.complete(JUDGE_SYSTEM, judge_prompt(q["question"], q.get("answer", []), cited, answer))
                )
            except (LLMError, ValueError) as exc:
                print(f"  judge failed on {q['question']!r}: {exc}", file=sys.stderr)
                continue
            rows.append((q, verdict))
            flag = "ok " if (verdict.correct if q["answerable"] else verdict.refused) else "BAD"
            print(f"  {flag} {q['question']}\n      {answer[:160]}\n      judge: {verdict.reason}")

    m = summarize(rows)
    print(f"\n### LLM-judged answers (answerer: {args.answerer}, judge: {args.judge_model}, {len(rows)} questions)\n")
    print("| Correct (answerable) | Faithful to cited passages | Refusal accuracy | Unanswerable refused |")
    print("|---|---|---|---|")
    print(
        f"| {m['correct']:.0%} | {m['faithful']:.0%} | {m['refusal_accuracy']:.0%} | {m['unanswerable_refused']:.0%} |"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
