"""Sentence-aware sliding-window chunker that remembers which pages each chunk spans."""

import re
from dataclasses import dataclass

from app.schemas import Chunk, PageText

_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])")
_TOKEN = re.compile(r"\w+|[^\w\s]")


@dataclass(frozen=True)
class _Sentence:
    text: str
    page: int


def count_tokens(text: str) -> int:
    """Cheap, model-agnostic token estimate (words and punctuation marks)."""
    return len(_TOKEN.findall(text))


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, first on paragraph breaks and then on sentence punctuation."""
    out: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if para:
            out.extend(s.strip() for s in _SENTENCE_END.split(para) if s.strip())
    return out


def _hard_split(sentence: str, size: int) -> list[str]:
    """Split an over-long sentence on word boundaries into pieces no longer than `size`."""
    pieces: list[str] = []
    current = ""
    for word in sentence.split():
        while len(word) > size:  # pathological unbroken string
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:size])
            word = word[size:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) > size:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _joined_len(sentences: list[_Sentence]) -> int:
    return sum(len(s.text) for s in sentences) + max(len(sentences) - 1, 0)


def chunk_pages(pages: list[PageText], doc_id: str, filename: str, size: int = 800, overlap: int = 150) -> list[Chunk]:
    """Pack sentences into chunks of at most `size` chars.

    Consecutive chunks share whole trailing sentences totalling at most `overlap` chars.
    Chunks may cross page boundaries; `page_start`/`page_end` record the span.
    """
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    sentences = [
        _Sentence(piece, p.page)
        for p in pages
        for sent in split_sentences(p.text)
        for piece in (_hard_split(sent, size) if len(sent) > size else [sent])
    ]

    windows: list[list[_Sentence]] = []
    current: list[_Sentence] = []
    for sent in sentences:
        if current and _joined_len([*current, sent]) > size:
            windows.append(current)
            tail: list[_Sentence] = []
            for prev in reversed(current):
                if _joined_len([prev, *tail]) > overlap:
                    break
                tail.insert(0, prev)
            if _joined_len([*tail, sent]) > size:
                tail = []
            current = tail
        current.append(sent)
    if current:
        windows.append(current)

    chunks: list[Chunk] = []
    for index, window in enumerate(windows):
        text = " ".join(s.text for s in window)
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}:{index}",
                doc_id=doc_id,
                filename=filename,
                page_start=min(s.page for s in window),
                page_end=max(s.page for s in window),
                index=index,
                text=text,
                token_count=count_tokens(text),
            )
        )
    return chunks
