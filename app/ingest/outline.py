"""Find a PDF's structure (title and headings) for study material.

PyMuPDF APIs verified against the installed version (2.1): `Document.get_toc()` returns
`[level, title, page]` rows; `Page.get_text("dict")` gives blocks → lines → spans with `text`,
`size` and `flags` (bit 16 = bold, `pymupdf.TEXT_FONT_BOLD`).
"""

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import PurePath

import pymupdf

from app.schemas import Heading

_BOLD = 16
_HAS_LETTER = re.compile(r"[A-Za-z]")
_MAX_HEADING_CHARS = 90


@dataclass(frozen=True)
class _Line:
    page: int
    text: str
    size: float
    bold: bool


def _lines(doc: pymupdf.Document) -> list[_Line]:
    out: list[_Line] = []
    for number, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                spans = [s for s in line["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = " ".join(" ".join(s["text"].split()) for s in spans).strip()
                size = max(s["size"] for s in spans)
                bold = all(s["flags"] & _BOLD for s in spans)
                out.append(_Line(number, text, round(size * 2) / 2, bold))
    return out


def _body_size(lines: list[_Line]) -> float:
    """The font size used for most characters."""
    weights: Counter[float] = Counter()
    for ln in lines:
        weights[ln.size] += len(ln.text)
    return weights.most_common(1)[0][0] if weights else 0.0


def _font_headings(lines: list[_Line], pages: int) -> list[Heading]:
    """Headings from typography: larger than body text, or bold at body size, short, not a sentence."""
    body = _body_size(lines)
    if not body:
        return []
    repeats = Counter(re.sub(r"\d+", "#", ln.text.lower()) for ln in lines)
    found: list[_Line] = []
    for ln in lines:
        text = ln.text
        if not (3 <= len(text) <= _MAX_HEADING_CHARS) or not _HAS_LETTER.search(text) or text.endswith((".", ",", ";")):
            continue
        if pages >= 3 and repeats[re.sub(r"\d+", "#", text.lower())] > pages / 2:
            continue  # running header or footer
        bigger = ln.size >= body * 1.15
        if bigger or (ln.bold and ln.size >= body and len(text) <= 60):
            found.append(ln)
    if not found:
        return []
    sizes = sorted({ln.size for ln in found}, reverse=True)
    level = {size: min(i + 1, 3) for i, size in enumerate(sizes)}
    return [Heading(page=ln.page, level=level[ln.size], title=ln.text) for ln in found]


def extract_outline(data: bytes) -> tuple[str, list[Heading]]:
    """(document title or "", headings in reading order). Uses the PDF bookmarks when present."""
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        meta_title = (doc.metadata or {}).get("title", "") or ""
        toc = [
            Heading(page=int(page), level=int(lvl), title=" ".join(str(title).split()))
            for lvl, title, page, *_ in doc.get_toc()
        ]
        toc = [h for h in toc if h.title and h.page >= 1]
        if len(toc) >= 2:
            return meta_title.strip(), toc
        lines = _lines(doc)
        title = meta_title.strip() or _title_guess(lines, doc.page_count)
        return title, _font_headings(lines, doc.page_count)


def _title_guess(lines: list[_Line], pages: int) -> str:
    """A running header repeated on most pages is usually the title; else page 1's largest line."""
    if pages >= 3:
        firsts = Counter(ln.text for ln in lines if ln.text and not ln.text[0].isdigit())
        for text, n in firsts.most_common(5):
            looks_like_title = 8 <= len(text) <= _MAX_HEADING_CHARS and _HAS_LETTER.search(text)
            if n > pages / 2 and looks_like_title and not re.search(r"\bpage\b|\d+\s*(of|/)\s*\d+", text, re.I):
                return text
    first = [ln for ln in lines if ln.page == 1 and 8 <= len(ln.text) <= _MAX_HEADING_CHARS]
    biggest = max(first, key=lambda ln: ln.size, default=None)
    return biggest.text if biggest and biggest.size >= _body_size(lines) * 1.4 else ""


def document_title(meta_title: str, filename: str) -> str:
    """A readable title: PDF metadata, else the file name without its extension."""
    if meta_title and len(meta_title) > 3 and not meta_title.lower().startswith(("microsoft word", "untitled")):
        return meta_title
    stem = PurePath(filename).stem.replace("_", " ").replace("-", " ").strip()
    return " ".join(stem.split()) or "Your document"
