"""Study material: local builder pieces, LLM JSON handling, and the /study endpoint."""

import json

from app.generation.fake import FakeLLM
from app.generation.study import (
    LLMStudyBuilder,
    LocalStudyBuilder,
    build_sections,
    find_flows,
    parse_json_object,
    shorten,
)
from app.ingest.cleaner import clean_pages
from app.ingest.loader import load_pdf
from app.ingest.outline import document_title, extract_outline
from app.schemas import Heading, PageText
from tests.conftest import CSRF, GOOD_KEY, register, upload
from tests.fakes import FakeEmbedder

ALL = ["notes", "sticky", "mindmap", "flow", "glossary", "flashcards"]


def _sample(sample_pdf):
    data = sample_pdf.read_bytes()
    raw = load_pdf(data, sample_pdf.name)
    meta, heads = extract_outline(data)
    return document_title(meta, sample_pdf.name), clean_pages(raw), raw, heads


def test_outline_finds_headings_and_title(sample_pdf):
    title, _, _, heads = _sample(sample_pdf)
    assert len(heads) >= 5 and all(h.level == 1 for h in heads)
    assert heads[0].page == 1 and "Welcome" in heads[0].title
    assert title  # from the running header or the file name


def test_sections_split_at_headings_and_quote_the_document(sample_pdf):
    _, pages, _, heads = _sample(sample_pdf)
    sections = build_sections(pages, heads)
    assert [s.title for s in sections][:2] == ["Welcome and Product Overview", "Safety Information"]
    full = " ".join(" ".join(p.text.split()) for p in pages)
    for sec in sections:
        for s in sec.sentences:
            assert s.text in full  # verbatim, nothing invented
            assert not s.text.startswith(("1.", "2. Safety"))  # headings are not sentences


def test_local_builder_makes_every_kind_with_pages(sample_pdf):
    title, pages, raw, heads = _sample(sample_pdf)
    out = LocalStudyBuilder(FakeEmbedder()).build(title, pages, raw, heads, ALL)
    assert out.notes and all(p.page for n in out.notes for p in n.points)
    assert 6 <= len(out.sticky) <= 12 and all(1 <= s.tone <= 8 for s in out.sticky)
    assert out.mindmap and len(out.mindmap.children) >= 5
    assert any(g.term == "Storm Watch" for g in out.glossary)
    assert any("_____" in c.front for c in out.flashcards)
    titles = {f.title for f in out.flows}
    assert "How to restart the system" in titles and "How to pair the gateway" in titles  # prose procedures
    rules = next(f for f in out.flows if f.kind == "conditions" and "what it means" in f.title)
    assert any(s.condition == "A solid red light" and s.text.startswith("Thermal shutdown") for s in rules.steps)
    assert [t.title for t in out.topics][:2] == ["Welcome and Product Overview", "Safety Information"]


def test_long_books_get_short_deduplicated_notes():
    """60 chapters that repeat the same sentences with a noun swapped: notes stay within the budget,
    a repeated point appears once, pronoun-led sentences are avoided, diagrams are merged."""
    from app.generation.study import NOTES_BUDGET, Section, Sentence, dedupe_flows, find_procedures

    sections = []
    for i in range(60):
        noun = f"widget{i}"
        sections.append(
            Section(
                f"Chapter {i}",
                i + 1,
                [
                    Sentence(f"A heritage {noun} is a {noun} listed in the register because of its age.", i + 1, i),
                    Sentence(f"It was built in {1900 + i} by the {noun} guild of the northern towns.", i + 1, i),
                    Sentence(f"The {noun} hall holds {100 + i} meetings each year for local members.", i + 1, i),
                    Sentence(
                        f"To book a {noun} tour, fill in the form, wait for a reply, then pay at the desk.", i + 1, i
                    ),
                    Sentence(f"Members of the {noun} society meet in spring to plan the summer markets.", i + 1, i),
                ],
            )
        )
    from app.generation import study

    vecs = study.score_sentences(sections, FakeEmbedder())
    sents = study.all_sentences(sections)
    scores = __import__("numpy").array([s.score for s in sents])
    picks = LocalStudyBuilder._pick_notes(sections, sents, scores, vecs, {id(s): i for i, s in enumerate(sents)})
    points = [s for chosen in picks.values() for s in chosen]
    assert len(points) <= 2 * len(sections) and len(points) <= max(NOTES_BUDGET, 2 * len(sections))
    assert sum("heritage" in s.text for s in points) <= 1  # the same point once, not 60 times
    assert not any(s.text.startswith("It ") for s in points)
    flows = dedupe_flows(find_procedures(sections))
    assert len(flows) == 1 and "the same in 59 other places" in flows[0].title


def test_prose_procedure_and_when_rules():
    from app.generation.study import Section, Sentence, find_procedures, find_rules

    sec = Section(
        "Help",
        2,
        [
            Sentence("To reset the router, unplug it, wait ten seconds, then plug it back in.", 2, 0),
            Sentence("If the light stays red, call support.", 2, 0),
            Sentence("When the power fails, the router keeps running on its battery.", 2, 0),
            Sentence("When power returns, the battery recharges.", 3, 0),
        ],
    )
    flow = find_procedures([sec])[0]
    assert flow.title == "How to reset the router"
    assert [s.text for s in flow.steps] == [
        "Unplug it",
        "Wait ten seconds",
        "Plug it back in",
        "If the light stays red, call support",
    ]
    rules = find_rules([sec])[0]
    assert rules.kind == "conditions"
    assert [s.condition for s in rules.steps] == [
        "If the light stays red",
        "When the power fails",
        "When power returns",
    ]
    assert rules.steps[2].page == 3


def test_numbered_steps_become_a_flow():
    raw = [PageText(page=3, text="Pairing:\n1. Open the app\n2. Tap Add device\nand wait\n3. Scan the code\nNext para")]
    flows = find_flows(raw, [Heading(page=1, level=1, title="1. Intro")])
    assert len(flows) == 1 and flows[0].title == "Pairing"
    assert [s.text for s in flows[0].steps] == ["Open the app", "Tap Add device and wait", "Scan the code"]


def test_shorten_keeps_meaning_and_limits_length():
    assert shorten("However, the cell heater warms the pack.", 100) == "The cell heater warms the pack."
    long = "A " + "very " * 80 + "long sentence."
    assert len(shorten(long, 120)) <= 120


def test_parse_json_object_tolerates_fences():
    assert parse_json_object('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}


def test_llm_builder_uses_model_json_and_falls_back(sample_pdf):
    title, pages, raw, heads = _sample(sample_pdf)
    reply = {
        "notes": [{"title": "Safety", "page": 2, "points": [{"text": "Keep 90 cm from gas meters.", "page": 2}]}],
        "glossary": [
            {"term": "Storm Watch", "definition": "Charges ahead of storms.", "page": 999},
            {"term": "LFP", "definition": "Lithium iron phosphate cells.", "page": 1},
            {"term": "Backup reserve", "definition": "Energy kept for outages.", "page": 5},
            {
                "term": "Emergency Stop",
                "definition": "The red button on the right-hand side of the enclosure.",
                "page": 2,
            },
            {"term": "Warranty", "definition": "Covers the battery for 25 years.", "page": 9},  # wrong number
        ],
        "sticky": "not a list",
    }
    llm = FakeLLM(completion=json.dumps(reply))
    out = LLMStudyBuilder(llm, LocalStudyBuilder(FakeEmbedder()), 4000).build(
        title, pages, raw, heads, ["notes", "glossary", "sticky"]
    )
    safety = next(n for n in out.notes if n.title == "Safety Information")  # mapped to the real section
    assert safety.points[0].text == "Keep 90 cm from gas meters."
    assert len(out.notes) == 10  # every section covered once: the others keep their local notes
    terms = {g.term for g in out.glossary}
    assert terms == {"LFP", "Backup reserve", "Emergency Stop"}  # supported by the cited page
    assert "Storm Watch" not in terms  # its page (999) doesn't exist, so it can't be checked
    assert "Warranty" not in terms  # "25 years" isn't on page 9 (it says 12 years)
    assert any("left out" in w for w in out.warnings)
    assert out.sticky and any("made locally" in w for w in out.warnings)  # unreadable part â†’ local
    system, user = llm.calls[0]
    assert "untrusted" in system and "[p. 2]" in user


def test_study_endpoint_local_and_api(h, sample_pdf):
    c = h.client
    register(c)
    doc = upload(c, sample_pdf).json()
    r = c.post("/study", json={"doc_id": doc["id"], "kinds": ["notes", "mindmap"], "mode": "local"}, headers=CSRF)
    assert r.status_code == 200, r.text
    pack = r.json()
    assert pack["mode"] == "local" and pack["notes"] and pack["mindmap"] and pack["sticky"] == []
    api = c.post(
        "/study", json={"doc_id": doc["id"], "kinds": ["notes"], "mode": "api"}, headers=CSRF | {"X-LLM-Key": GOOD_KEY}
    )
    assert api.status_code == 200 and api.json()["mode"] == "api"  # FakeLLM's reply isn't JSON â†’ local fallback
    assert any("made locally" in w for w in api.json()["warnings"])


def test_study_endpoint_errors(h, sample_pdf):
    c = h.client
    register(c)
    doc = upload(c, sample_pdf).json()
    assert c.post("/study", json={"doc_id": "nope", "kinds": ["notes"]}, headers=CSRF).status_code == 404
    assert c.post("/study", json={"doc_id": doc["id"], "kinds": []}, headers=CSRF).status_code == 422
    no_key = c.post("/study", json={"doc_id": doc["id"], "kinds": ["notes"], "mode": "api"}, headers=CSRF)
    assert no_key.status_code == 400 and no_key.json()["error"]["code"] == "no_api_key"
    other = h.new_client()
    register(other, "eve@example.com")
    assert other.post("/study", json={"doc_id": doc["id"], "kinds": ["notes"]}, headers=CSRF).status_code == 404
