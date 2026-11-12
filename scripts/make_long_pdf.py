"""Generate a long (520-page) test book with known answers, for testing at scale.

`samples/long/meridian_almanac.pdf` is a fictional almanac of an invented country: 26 chapters x 20
pages. Every page describes three named places or institutions with their own facts (province,
founding year, founder, staff, a chapter-specific measurement), so every page holds unique,
checkable information and many pages look alike (as in real reference books, which makes retrieval
non-trivial). Each chapter also has a definition, a one-sentence procedure, "X means Y" rules and a
numbered list, to exercise the study-material diagrams.

The script also writes `samples/long/questions.json`: answerable questions spread across the whole
book (with gold answers and pages) plus unanswerable ones. All text is generated here (no copyright).

Usage: python scripts/make_long_pdf.py [output_dir]
"""

import json
import random
import sys
from pathlib import Path

import pymupdf

TITLE = "The Meridian Almanac"
PAGES_PER_CHAPTER = 20
SEED = 7

# (chapter title, kind of thing, measurement phrase, unit, low, high)
CHAPTERS: list[tuple[str, str, str, str, int, int]] = [
    ("Bridges", "Bridge", "spans", "metres", 80, 2400),
    ("Dams and Reservoirs", "Reservoir", "holds", "million cubic metres", 12, 900),
    ("Railway Stations", "Station", "handles", "passengers a day", 900, 90000),
    ("Universities", "University", "enrols", "students", 1200, 48000),
    ("Hospitals", "Hospital", "has", "beds", 60, 1900),
    ("Seaports", "Port", "moves", "thousand containers a year", 20, 4800),
    ("Airports", "Airport", "has a main runway of", "metres", 1400, 4200),
    ("Museums", "Museum", "holds", "objects", 3000, 900000),
    ("Libraries", "Library", "lends", "books a year", 8000, 2000000),
    ("Power Stations", "Power Station", "generates", "megawatts", 40, 3800),
    ("Lighthouses", "Lighthouse", "stands", "metres tall", 12, 76),
    ("Vineyards", "Vineyard", "covers", "hectares", 4, 640),
    ("Forests", "Forest", "covers", "square kilometres", 30, 9000),
    ("Lakes", "Lake", "reaches a depth of", "metres", 6, 410),
    ("Mountains", "Peak", "rises to", "metres", 900, 5600),
    ("Canals", "Canal", "runs for", "kilometres", 3, 310),
    ("Stadiums", "Stadium", "seats", "spectators", 4000, 98000),
    ("Theatres", "Theatre", "seats", "people", 120, 2600),
    ("Observatories", "Observatory", "uses a telescope mirror of", "centimetres", 40, 820),
    ("Markets", "Market", "has", "stalls", 30, 2400),
    ("Shipyards", "Shipyard", "has a dry dock of", "metres", 90, 520),
    ("Mines", "Mine", "reaches a depth of", "metres", 60, 3900),
    ("Wind Farms", "Wind Farm", "has", "turbines", 6, 340),
    ("Botanical Gardens", "Garden", "grows", "plant species", 400, 31000),
    ("Tunnels", "Tunnel", "is", "metres long", 300, 57000),
    ("Zoos", "Zoo", "keeps", "animals", 150, 16000),
]
PROVINCES = [
    "Arvelle",
    "Brannock",
    "Castmoor",
    "Dellwyn",
    "Eskar",
    "Fallowmere",
    "Greystrand",
    "Hollin",
    "Ithmar",
    "Juniper Coast",
    "Kestrel Fells",
    "Lowmarch",
]
FIRST = [
    "Ada",
    "Bram",
    "Celia",
    "Dorin",
    "Elsa",
    "Fenwick",
    "Greta",
    "Hugo",
    "Ilse",
    "Jory",
    "Katrin",
    "Lucan",
    "Mira",
    "Nils",
    "Orla",
    "Pavel",
    "Quinn",
    "Rosa",
    "Silas",
    "Tamsin",
    "Ulric",
    "Vera",
    "Wendel",
    "Yara",
]
LAST = [
    "Ashdown",
    "Blackwood",
    "Corran",
    "Duvall",
    "Ember",
    "Faulkner",
    "Garrow",
    "Hale",
    "Ingram",
    "Jessop",
    "Kerrow",
    "Lindqvist",
    "Marlow",
    "Norcott",
    "Oakes",
    "Pellam",
    "Quarles",
    "Rendell",
    "Strand",
    "Tolver",
]
SYL_A = [
    "Kor",
    "El",
    "Var",
    "Thal",
    "Bri",
    "Mor",
    "Sel",
    "Dun",
    "Ast",
    "Pell",
    "Quen",
    "Ros",
    "Tav",
    "Wyn",
    "Hal",
    "Orm",
    "Cael",
    "Fen",
    "Gild",
    "Lun",
]
SYL_B = [
    "vath",
    "mere",
    "dale",
    "wick",
    "ford",
    "stone",
    "holm",
    "brook",
    "gate",
    "hurst",
    "moor",
    "well",
    "crest",
    "field",
    "haven",
    "ridge",
    "shaw",
    "thorne",
    "by",
    "wyn",
]


def names(rng: random.Random, n: int) -> list[str]:
    """`n` distinct invented place names."""
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < n:
        name = rng.choice(SYL_A) + rng.choice(SYL_B)
        if rng.random() < 0.35:
            name += " " + rng.choice(["Upper", "Lower", "North", "South", "East", "West", "Old", "New"])
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def chapter_extras(title: str, kind: str, unit: str) -> dict[str, str]:
    """Definition, procedure, rules and a numbered list for one chapter (for the diagrams)."""
    k = kind.lower()
    return {
        "definition": (
            f"A heritage {k} is a {k} listed in the national register because of its age, design or "
            f"historical role; listed sites receive state funding for repairs."
        ),
        "procedure": (
            f"To request a guided visit to a heritage {k}, fill in the visitor form on the national portal, "
            f"wait for the confirmation email, then show the confirmation code at the entrance."
        ),
        "rules": (
            f"A green plaque on a {k} means the site is open to the public all year. "
            f"A blue plaque means the site is open only in summer. "
            f"A red plaque indicates that the site is closed for restoration."
        ),
        "list": "\n".join(
            [
                f"How a new {k} is approved:",
                "1. The province submits a feasibility study",
                "2. The national planning board holds a public hearing",
                "3. The environment office issues an impact permit",
                "4. Parliament approves the budget",
            ]
        ),
    }


def build(out_dir: Path) -> tuple[Path, Path]:
    """Write the PDF and the question file; return their paths."""
    rng = random.Random(SEED)
    doc = pymupdf.open()
    toc: list[list] = []
    facts: list[dict] = []
    all_names = names(rng, len(CHAPTERS) * PAGES_PER_CHAPTER * 3)
    page_no = 0
    for ci, (chapter, kind, verb, unit, lo, hi) in enumerate(CHAPTERS):
        extras = chapter_extras(chapter, kind, unit)
        for pi in range(PAGES_PER_CHAPTER):
            page_no += 1
            page = doc.new_page(width=595, height=842)
            page.insert_text((50, 40), TITLE, fontsize=9, fontname="helv", color=(0.4, 0.4, 0.4))
            y = 70
            if pi == 0:
                heading = f"Chapter {ci + 1}. {chapter}"
                page.insert_text((50, 95), heading, fontsize=18, fontname="hebo")
                toc.append([1, heading, page_no])
                y = 115
            paras: list[str] = []
            if pi == 0:
                paras.append(
                    f"This chapter describes the {chapter.lower()} of the Meridian Republic, province by province. "
                    f"{extras['definition']}"
                )
            if pi == 1:
                paras.append(extras["procedure"] + " Visits are free for children under twelve.")
            if pi == 2:
                paras.append(extras["rules"])
            if pi == 3:
                paras.append(extras["list"])
            for e in range(3):
                name = all_names[(ci * PAGES_PER_CHAPTER + pi) * 3 + e]
                full = f"{name} {kind}"
                province = rng.choice(PROVINCES)
                year = rng.randint(1712, 2021)
                founder = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
                staff = rng.randint(12, 4800)
                measure = rng.randint(lo, hi)
                km = rng.randint(3, 240)
                town = rng.choice(all_names)
                paras.append(
                    f"The {full} is located in {province} province, {km} kilometres from {town}. "
                    f"It was founded in {year} by {founder}. "
                    f"Today the {full} {verb} {measure:,} {unit} and employs {staff:,} people. "
                    f"Its management reports to the {province} provincial council, which reviews its accounts every "
                    f"{rng.choice(['two', 'three', 'four', 'five'])} years."
                )
                facts.append(
                    {
                        "name": full,
                        "page": page_no,
                        "province": province,
                        "year": year,
                        "founder": founder,
                        "staff": staff,
                        "measure": measure,
                        "unit": unit,
                        "verb": verb,
                        "chapter": chapter,
                    }
                )
            body = "\n\n".join(paras)
            rect = pymupdf.Rect(50, y + 5, 545, 790)
            if page.insert_textbox(rect, body, fontsize=10.5, fontname="helv", lineheight=1.35) < 0:
                raise RuntimeError(f"page {page_no} overflows")
            page.insert_text((50, 815), f"Page {page_no} - {TITLE}", fontsize=9, fontname="helv", color=(0.4, 0.4, 0.4))
    doc.set_toc(toc)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "meridian_almanac.pdf"
    doc.save(pdf, garbage=3, deflate=True)

    q_rng = random.Random(SEED + 1)
    picks = q_rng.sample(facts, 40)
    questions = []
    for i, f in enumerate(sorted(picks, key=lambda f: f["page"])):
        style = i % 4
        if style == 0:
            q, gold = f"In which province is the {f['name']}?", f["province"]
        elif style == 1:
            q, gold = f"Who founded the {f['name']}?", f["founder"]
        elif style == 2:
            q, gold = f"How many people does the {f['name']} employ?", f"{f['staff']:,}"
        else:
            q, gold = f"In what year was the {f['name']} founded?", str(f["year"])
        questions.append({"question": q, "gold_answer": gold, "page": f["page"], "answerable": True})
    for kind in ("Bridge", "Hospital", "Airport", "Museum", "Mine", "Zoo"):
        fake = f"Zanzibarra {kind}"
        questions.append(
            {"question": f"Who founded the {fake}?", "gold_answer": None, "page": None, "answerable": False}
        )
    qfile = out_dir / "questions.json"
    qfile.write_text(json.dumps(questions, indent=1), encoding="utf-8")
    return pdf, qfile


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("samples/long")
    for path in build(out):
        print(path)
