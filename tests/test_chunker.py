"""Unit tests for cleaning and chunking."""

import itertools
import re

import pytest

from app.ingest.chunker import chunk_pages, split_sentences
from app.ingest.cleaner import clean_pages, clean_text
from app.schemas import PageText


def _sentence(i: int) -> str:
    return f"Sentence number {i} talks about topic {i % 7} in some detail here."


@pytest.fixture
def pages() -> list[PageText]:
    return [PageText(page=p, text=" ".join(_sentence(p * 100 + i) for i in range(25))) for p in range(1, 5)]


def test_no_empty_chunks_and_size_limit(pages):
    chunks = chunk_pages(pages, "doc", "a.pdf", size=800, overlap=150)
    assert len(chunks) > 4
    for c in chunks:
        assert c.text.strip()
        assert len(c.text) <= 800
        assert c.token_count > 0


def test_ids_and_indexes_are_sequential(pages):
    chunks = chunk_pages(pages, "doc", "a.pdf")
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.chunk_id == f"doc:{c.index}" for c in chunks)


def test_overlap_is_whole_sentences_within_budget(pages):
    chunks = chunk_pages(pages, "doc", "a.pdf", size=800, overlap=150)
    for prev, nxt in itertools.pairwise(chunks):
        prev_sents = split_sentences(prev.text)
        nxt_sents = split_sentences(nxt.text)
        shared = next(
            (k for k in range(min(len(prev_sents), len(nxt_sents)), 0, -1) if prev_sents[-k:] == nxt_sents[:k]),
            0,
        )
        assert shared >= 1, "consecutive chunks should share at least one sentence"
        overlap_text = " ".join(nxt_sents[:shared])
        assert prev.text.endswith(overlap_text)
        assert len(overlap_text) <= 150


def test_page_ranges(pages):
    chunks = chunk_pages(pages, "doc", "a.pdf")
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 4
    for c in chunks:
        # Every sentence encodes its page as number // 100.
        pages_in_text = [int(n) // 100 for n in re.findall(r"Sentence number (\d+)", c.text)]
        assert c.page_start == min(pages_in_text)
        assert c.page_end == max(pages_in_text)
    assert any(c.page_start != c.page_end for c in chunks), "some chunk should cross a page"


def test_long_sentence_is_hard_split():
    page = PageText(page=1, text="word " * 600)
    chunks = chunk_pages([page], "d", "x.pdf", size=800, overlap=150)
    assert len(chunks) >= 4
    assert all(0 < len(c.text) <= 800 for c in chunks)


def test_empty_pages_produce_no_chunks():
    assert chunk_pages([PageText(page=1, text="   ")], "d", "x.pdf") == []


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_pages([], "d", "x.pdf", size=100, overlap=100)


def test_cleaner_joins_hyphenation_and_whitespace():
    assert clean_text("an exam-\nple   of\ntext") == "an example of text"


def test_cleaner_removes_repeated_headers_and_page_numbers():
    words = ["apples", "bridges", "comets", "dunes"]
    pages = [
        PageText(page=i, text=f"ACME Annual Report\nAll about {w}.\nMore on {w} here.\nPage {i}")
        for i, w in enumerate(words, start=1)
    ]
    cleaned = clean_pages(pages)
    for p, w in zip(cleaned, words, strict=True):
        assert p.text == f"All about {w}. More on {w} here."


def test_cleaner_keeps_body_lines_that_only_differ_by_numbers():
    pages = [PageText(page=i, text=f"Header\nIntro.\nTable row {i}.\nBody.\nMore body.\nFooter") for i in range(1, 5)]
    cleaned = clean_pages(pages)
    assert all(f"Table row {i}." in p.text for i, p in enumerate(cleaned, start=1))
    assert all("Header" not in p.text and "Footer" not in p.text for p in cleaned)
