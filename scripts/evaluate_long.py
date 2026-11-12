"""Scale test: a long PDF through the real app (real models, in-process HTTP, no rate limits).

Default: builds `samples/long/meridian_almanac.pdf` (scripts/make_long_pdf.py, 520 pages) and its
questions. Or bring your own: `--pdf book.pdf --answers key.json`, where the key is a JSON list of
`{"question": ..., "answer": ..., "page": ...}` (extra fields are ignored; `"answerable": false`
marks a question the book can't answer).

It then:
1. times ingest (read → split → embed → index),
2. asks every question and scores it: the reply must contain every key fact of the expected answer
   (its numbers, amounts, percentages and capitalised names) and cite the expected page; questions
   about things not in the book must be refused,
3. builds the study material and reports its size (note points/words, mind-map branches, diagrams).

Usage:
  python scripts/evaluate_long.py [--mode local|ollama]
  python scripts/evaluate_long.py --pdf my_book.pdf --answers my_key.json [--misses]
"""

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.services.container import build_container  # noqa: E402
from scripts.make_long_pdf import build  # noqa: E402

H = {"X-Requested-With": "fetch"}
KINDS = ["notes", "sticky", "mindmap", "flow", "glossary", "flashcards"]
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_NAME = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}\b")
_NOT_FACTS = {"The", "This", "Project", "Chapter", "Page"}


def key_facts(answer: str, question: str) -> list[str]:
    """What a correct reply must contain: every number in the expected answer, plus capitalised
    names that the question doesn't already mention ("Pune" for "Where is the Falcon hub?")."""
    facts = [n.replace(",", "") for n in _NUMBER.findall(answer)]
    asked = set(re.findall(r"[A-Za-z]+", question))
    facts += [w for w in _NAME.findall(answer) if w not in _NOT_FACTS and w not in asked]
    return facts or [answer]


def has_fact(reply: str, fact: str) -> bool:
    """Whole-number / whole-word match, commas ignored ("5" must not match "15" or "5.2")."""
    text = reply.replace(",", "")
    return (
        re.search(rf"(?<![\d.]){re.escape(fact)}(?![\d]|\.\d)", text) is not None
        if fact[0].isdigit()
        else (re.search(rf"\b{re.escape(fact)}\b", text, re.I) is not None)
    )


def ask(client: TestClient, question: str, mode: str) -> tuple[str, list[dict], dict]:
    r = client.post("/chat/ask", json={"question": question, "mode": mode}, headers=H)
    text, citations, done = "", [], {}
    for block in r.text.strip().split("\n\n"):
        head, data = block.split("\n", 1)
        event, payload = head.split(": ", 1)[1], json.loads(data.split("data: ", 1)[1])
        if event == "token":
            text += payload["text"]
        elif event == "citations":
            citations = payload["citations"]
        elif event in ("done", "error"):
            done = payload
    return text, citations, done


def load_questions(path: Path) -> list[dict]:
    """Normalise an answer key to {question, gold_answer|None, page|None, answerable}."""
    out = []
    for q in json.loads(path.read_text(encoding="utf-8")):
        answerable = q.get("answerable", True)
        gold = q.get("gold_answer", q.get("answer"))
        out.append(
            {
                "question": q["question"],
                "gold": gold,
                "page": q.get("page"),
                "answerable": answerable,
                "id": q.get("id", ""),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="local", choices=["local", "ollama"])
    parser.add_argument("--pdf", type=Path, help="your PDF (default: the generated 520-page almanac)")
    parser.add_argument("--answers", type=Path, help="answer key JSON for --pdf")
    parser.add_argument("--unanswerable", nargs="*", default=[], help="extra questions the book can't answer")
    parser.add_argument("--misses", action="store_true", help="print every question that wasn't fully right")
    parser.add_argument("--no-study", action="store_true", help="skip the study-material step")
    args = parser.parse_args()
    if args.pdf:
        pdf, qfile = args.pdf, args.answers
        if qfile is None:
            sys.exit("--pdf needs --answers (a JSON list of {question, answer, page}).")
    else:
        folder = ROOT / "samples" / "long"
        pdf, qfile = folder / "meridian_almanac.pdf", folder / "questions.json"
        if not pdf.exists() or not qfile.exists():
            build(folder)
    questions = load_questions(qfile)
    questions += [
        {"question": q, "gold": None, "page": None, "answerable": False, "id": "extra"} for q in args.unanswerable
    ]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        settings = Settings(
            data_dir=Path(tmp),
            database_url="",
            redis_url="",
            chroma_url="",
            ask_rate_per_min=100000,
            max_upload_mb=200,
        )
        with TestClient(create_app(build_container(settings))) as c:
            c.post("/auth/register", json={"email": "scale@example.com", "password": "scale-test-pass"}, headers=H)
            start = time.perf_counter()
            with pdf.open("rb") as fh:
                doc = c.post("/documents", files={"file": (pdf.name, fh, "application/pdf")}, headers=H).json()
            if "id" not in doc:
                sys.exit(f"Upload failed: {doc}")
            status = c.get(f"/documents/{doc['id']}/status").json()  # TestClient runs the ingest task inline
            if status["status"] != "ready":
                sys.exit(f"Ingest failed: {status}")
            print(
                f"Ingest: {status['pages']} pages, {status['chunks']} chunks in {time.perf_counter() - start:.1f} s\n"
            )

            ok = cited = both = refused = 0
            latencies: list[float] = []
            misses: list[str] = []
            for q in questions:
                t = time.perf_counter()
                text, cites, done = ask(c, q["question"], args.mode)
                latencies.append(time.perf_counter() - t)
                if q["answerable"]:
                    facts = key_facts(q["gold"], q["question"])
                    fact_ok = all(has_fact(text, f) for f in facts)
                    page_ok = q["page"] is None or any(x["page_start"] <= q["page"] <= x["page_end"] for x in cites)
                    ok += fact_ok
                    cited += page_ok
                    both += fact_ok and page_ok
                    if not (fact_ok and page_ok):
                        pages = sorted({x["page_start"] for x in cites})
                        misses.append(
                            f"{q['id']:<6} p{q['page']} {q['question']}\n       expected {facts}; "
                            f"fact={'ok' if fact_ok else 'MISSING'} page={'ok' if page_ok else 'WRONG'} "
                            f"cited pages {pages}\n       reply: {(done.get('span') or text).strip()[:160]!r}"
                        )
                else:
                    good = done.get("answerable") is False
                    refused += good
                    if not good:
                        misses.append(f"{q['id']:<6} (no answer) {q['question']}\n       reply: {text.strip()[:160]!r}")
            n = sum(q["answerable"] for q in questions)
            latencies.sort()
            print("| Mode | Correct fact + page | Correct fact | Cited the right page | Refused (no answer) | Median |")
            print("|---|---|---|---|---|---|")
            print(
                f"| {args.mode} | {both}/{n} | {ok}/{n} | {cited}/{n} | {refused}/{len(questions) - n} | "
                f"{latencies[len(latencies) // 2] * 1000:.0f} ms |\n"
            )
            if args.misses and misses:
                print("Misses:\n" + "\n".join(misses) + "\n")

            if args.no_study:
                return
            t = time.perf_counter()
            pack = c.post("/study", json={"doc_id": doc["id"], "kinds": KINDS, "mode": args.mode}, headers=H).json()
            points = [p for s in pack["notes"] for p in s["points"]]
            words = sum(len(p["text"].split()) for p in points)
            print(
                f"Study material ({pack['label']}) in {time.perf_counter() - t:.1f} s: title {pack['title']!r}, "
                f"{len(pack['notes'])} sections, {len(points)} note points ({words} words), "
                f"{len(pack['sticky'])} sticky notes, "
                f"{len(pack['mindmap']['children']) if pack['mindmap'] else 0} mind-map branches, "
                f"{len(pack['flows'])} diagrams, {len(pack['glossary'])} key terms, "
                f"{len(pack['flashcards'])} flashcards"
            )
            if args.misses:
                Path(tmp, "pack.json").write_text(json.dumps(pack, indent=1), encoding="utf-8")
                out = ROOT / "data" / "last_study_pack.json"
                out.parent.mkdir(exist_ok=True)
                out.write_text(json.dumps(pack, indent=1), encoding="utf-8")
                print(f"Full study output saved to {out}")


if __name__ == "__main__":
    main()
