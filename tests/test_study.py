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
    assert 6 <= len(out.sticky) <= 12 and all(1 <= s.tone <= 4 for s in out.sticky)
    assert out.mindmap and len(out.mindmap.children) >= 5
    assert any(g.term == "Storm Watch" for g in out.glossary)
    assert any("_____" in c.front for c in out.flashcards)
    assert out.flows == [] and out.warnings  # the sample has no numbered procedure


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
        "glossary": [{"term": "Storm Watch", "definition": "Charges ahead of storms.", "page": 999}],
        "sticky": "not a list",
    }
    llm = FakeLLM(completion=json.dumps(reply))
    out = LLMStudyBuilder(llm, LocalStudyBuilder(FakeEmbedder()), 4000).build(
        title, pages, raw, heads, ["notes", "glossary", "sticky"]
    )
    assert out.notes[0].points[0].text == "Keep 90 cm from gas meters."
    assert out.glossary[0].page is None  # invented page numbers are dropped
    assert out.sticky and any("made locally" in w for w in out.warnings)  # unreadable part → local
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
    assert api.status_code == 200 and api.json()["mode"] == "api"  # FakeLLM's reply isn't JSON → local fallback
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
