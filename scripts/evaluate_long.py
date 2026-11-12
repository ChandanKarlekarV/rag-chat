"""Scale test: a 520-page book through the real app (real models, in-process HTTP).

Builds `samples/long/meridian_almanac.pdf` if needed (scripts/make_long_pdf.py), uploads it, then:
1. times ingest (read → split → embed → index),
2. asks every question in `samples/long/questions.json` (answers spread from page 14 to page 519,
   plus questions the book can't answer) and scores fact + cited page, and refusals,
3. builds the study material and reports its size (notes points/words, mind-map branches, diagrams).

Usage: python scripts/evaluate_long.py [--mode local|ollama]
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


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="local", choices=["local", "ollama"])
    args = parser.parse_args()
    folder = ROOT / "samples" / "long"
    pdf, qfile = folder / "meridian_almanac.pdf", folder / "questions.json"
    if not pdf.exists() or not qfile.exists():
        build(folder)
    questions = json.loads(qfile.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        settings = Settings(data_dir=Path(tmp), database_url="", redis_url="", chroma_url="", ask_rate_per_min=100000)
        with TestClient(create_app(build_container(settings))) as c:
            c.post("/auth/register", json={"email": "scale@example.com", "password": "scale-test-pass"}, headers=H)
            start = time.perf_counter()
            with pdf.open("rb") as fh:
                doc = c.post("/documents", files={"file": (pdf.name, fh, "application/pdf")}, headers=H).json()
            status = c.get(f"/documents/{doc['id']}/status").json()  # TestClient runs the ingest task inline
            print(
                f"Ingest: {status['pages']} pages, {status['chunks']} chunks in {time.perf_counter() - start:.1f} s\n"
            )

            ok = cited = refused = 0
            latencies: list[float] = []
            for q in questions:
                t = time.perf_counter()
                text, cites, done = ask(c, q["question"], args.mode)
                latencies.append(time.perf_counter() - t)
                if q["answerable"]:
                    ok += norm(q["gold_answer"]) in norm(text)
                    cited += any(x["page_start"] <= q["page"] <= x["page_end"] for x in cites)
                else:
                    refused += done.get("answerable") is False
            n = sum(q["answerable"] for q in questions)
            latencies.sort()
            print("| Mode | Correct fact | Cited the right page | Refused (no answer in book) | Median latency |")
            print("|---|---|---|---|---|")
            print(
                f"| {args.mode} | {ok}/{n} | {cited}/{n} | {refused}/{len(questions) - n} | "
                f"{latencies[len(latencies) // 2] * 1000:.0f} ms |\n"
            )

            t = time.perf_counter()
            pack = c.post("/study", json={"doc_id": doc["id"], "kinds": KINDS, "mode": args.mode}, headers=H).json()
            points = [p for s in pack["notes"] for p in s["points"]]
            words = sum(len(p["text"].split()) for p in points)
            print(
                f"Study material ({pack['label']}) in {time.perf_counter() - t:.1f} s: {len(pack['notes'])} sections, "
                f"{len(points)} note points ({words} words), {len(pack['sticky'])} sticky notes, "
                f"{len(pack['mindmap']['children']) if pack['mindmap'] else 0} mind-map branches, "
                f"{len(pack['flows'])} diagrams, {len(pack['glossary'])} key terms, "
                f"{len(pack['flashcards'])} flashcards"
            )


if __name__ == "__main__":
    main()
