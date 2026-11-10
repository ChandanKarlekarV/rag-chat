"""Normalise raw PDF text: fix hyphenation, strip repeated headers/footers, collapse whitespace."""

import re
from collections import Counter

from app.schemas import PageText

_HYPHEN_BREAK = re.compile(r"(\w)-\n\s*([a-z])")
_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"[ \t ]+")
_MIN_PAGES_FOR_BOILERPLATE = 3
_MAX_BOILERPLATE_LEN = 100
_EDGE_LINES = 2


def _line_key(line: str) -> str:
    """Normalise a line so 'Page 3' and 'Page 4' compare equal."""
    return _DIGITS.sub("#", line.strip().lower())


def _edge_lines(text: str) -> list[tuple[int, str]]:
    """Return (position, line) for the first and last few non-empty lines of a page."""
    lines = [(i, ln) for i, ln in enumerate(text.splitlines()) if ln.strip()]
    if len(lines) <= 2 * _EDGE_LINES:
        return lines
    return lines[:_EDGE_LINES] + lines[-_EDGE_LINES:]


def find_boilerplate(pages: list[str]) -> set[str]:
    """Return normalised short header/footer lines that appear on more than half of the pages.

    Only the first and last `_EDGE_LINES` lines of each page are candidates, so body lines that
    merely look alike (e.g. table rows differing by a number) are never removed.
    """
    if len(pages) < _MIN_PAGES_FOR_BOILERPLATE:
        return set()
    counts: Counter[str] = Counter()
    for text in pages:
        counts.update({_line_key(ln) for _, ln in _edge_lines(text) if len(ln.strip()) <= _MAX_BOILERPLATE_LEN})
    return {key for key, n in counts.items() if n > len(pages) / 2}


def clean_text(text: str, boilerplate: set[str] | None = None) -> str:
    """Clean one page. Paragraphs are kept separated by a blank line; lines inside are joined."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if boilerplate:
        edges = {i for i, ln in _edge_lines(text) if _line_key(ln) in boilerplate}
        text = "\n".join(ln for i, ln in enumerate(text.split("\n")) if i not in edges)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    lines = text.split("\n")
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        stripped = _SPACES.sub(" ", line).strip()
        if stripped:
            current.append(stripped)
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def clean_pages(pages: list[PageText]) -> list[PageText]:
    """Clean every page, removing lines that repeat on most pages (running headers/footers)."""
    boilerplate = find_boilerplate([p.text for p in pages])
    return [PageText(page=p.page, text=clean_text(p.text, boilerplate)) for p in pages]
