"""Local extractive provider: answers by quoting the PDF, never by generating text.

Pipeline after the shared retrieval and gate (ANSWER_MODES.md section 2):

1. An extractive QA model (`ExtractiveReader`, SQuAD 2.0) proposes the best answer span per chunk
   together with its "no answer" margin; `qa_score = sigmoid(margin)` is the probability-like
   confidence that the span beats "no answer".
2. Candidates are ranked by `QA_WEIGHT * qa_score + (1 - QA_WEIGHT) * sigmoid(rerank_score)`, must
   have the type the question asks for (price → money, "how many" → a number), and the winner must
   reach `MIN_QA_SCORE`; otherwise the provider answers NO_ANSWER.
3. Supporting sentences come from the winning chunk and the next two best chunks, scored by the
   cross-encoder against the question; the top `MAX_SENTENCES` are shown in document order.
4. The response is assembled from templates. Every sentence shown is an exact quote with a citation.

Summary questions get an excerpt summary (opening sentences of the best-spread sections) and
explain/compare questions get supporting sentences without a single span, each with a hint that a
generative mode writes a fuller answer.
"""

import math
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol

from app.generation.base import EXTRACTIVE_LABEL, AnswerEvent, AnswerMode, QuestionType
from app.generation.prompt import NO_ANSWER_TOKEN
from app.generation.reader import Span
from app.schemas import RetrievedChunk

_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9])")
_ASKS_PRICE = re.compile(r"\b(cost|costs|price|priced|pricing)\b", re.I)
_MONEY = re.compile(
    r"[$€£¥₹]|\b(usd|eur|gbp|inr|dollars?|euros?|pounds?|rupees?|cents?|free of charge|no cost)\b", re.I
)
_ASKS_COUNT = re.compile(r"\bhow many\b", re.I)
_NUMBER = re.compile(
    r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozen|hundred|thousand|single|both|no)\b",
    re.I,
)
SUMMARY_HINT = "For a written summary, switch to Claude API or Ollama mode."
EXPLAIN_HINT = "For a written explanation, switch to Claude API or Ollama mode."
MIN_SENTENCE_SCORE = -4.0  # cross-encoder logit below which a sentence is too unrelated to show


class SpanReader(Protocol):
    def best_spans(self, question: str, sources: list[RetrievedChunk]) -> list[Span]: ...


SentenceScorer = Callable[[str, list[str]], list[float]]


@dataclass(frozen=True)
class Sentence:
    """A sentence inside a retrieved chunk."""

    source: int  # 1-based source number
    start: int
    end: int
    text: str


def sigmoid(x: float) -> float:
    """Map a logit to (0, 1)."""
    return 1.0 / (1.0 + math.exp(-x))


def answer_type_ok(question: str, span_text: str) -> bool:
    """Reject spans of the wrong type: price questions need money, "how many" needs a number."""
    if _ASKS_PRICE.search(question) and not _MONEY.search(span_text):
        return False
    return not (_ASKS_COUNT.search(question) and not _NUMBER.search(span_text))


def split_sentences(text: str, source: int) -> list[Sentence]:
    """Sentences of a chunk with character offsets (verbatim, whitespace trimmed).

    A sentence ends at . ! or ? followed by whitespace and a capital letter or digit, so decimals
    ("37.8 MWh"), versions and abbreviations inside a sentence are never cut.
    """
    bounds = [0, *(m.end() for m in _SENTENCE_END.finditer(text)), len(text)]
    out = []
    for a, b in pairwise(bounds):
        raw = text[a:b]
        lead = len(raw) - len(raw.lstrip())
        sentence = raw.strip()
        if len(sentence) > 1:
            start = a + lead
            out.append(Sentence(source, start, start + len(sentence), sentence))
    return out


def _key(text: str) -> str:
    return " ".join(text.lower().split())


def _quote(s: Sentence) -> str:
    return f"> {s.text} [{s.source}]"


class LocalExtractiveProvider:
    """Extractive answers computed on this machine; needs no key or network."""

    mode = AnswerMode.LOCAL
    label = EXTRACTIVE_LABEL

    def __init__(
        self,
        reader: SpanReader,
        model: str,
        scorer: SentenceScorer,
        min_qa_score: float = 0.05,
        qa_weight: float = 0.6,
        max_sentences: int = 3,
    ) -> None:
        self.reader = reader
        self.name = model
        self.scorer = scorer
        self.min_qa_score = min_qa_score
        self.qa_weight = qa_weight
        self.max_sentences = max_sentences

    def is_available(self) -> tuple[bool, str]:
        """Always available once the model is loaded."""
        return True, "Runs on this server. No key needed."

    # ---- span selection ----

    def choose(self, question: str, chunks: list[RetrievedChunk]) -> tuple[Span, float] | None:
        """Best type-consistent span by combined score, with its qa_score (no threshold applied)."""
        spans = self.reader.best_spans(question, chunks)
        ranked = []
        for sp in spans:
            qa = sigmoid(sp.margin)
            rr = sigmoid(chunks[sp.source - 1].rerank_score or 0.0)
            ranked.append((self.qa_weight * qa + (1 - self.qa_weight) * rr, qa, sp))
        ranked.sort(key=lambda x: x[0], reverse=True)
        for _, qa, sp in ranked:
            if answer_type_ok(question, chunks[sp.source - 1].chunk.text[sp.start : sp.end]):
                return sp, qa
        return None

    def _supporting(
        self, question: str, chunks: list[RetrievedChunk], first: int, must_include: Sentence | None = None
    ) -> list[Sentence]:
        """Top sentences (by cross-encoder) from chunk `first` and the next two best chunks, in document order."""
        order = [first, *[i for i in range(1, len(chunks) + 1) if i != first][:2]]
        candidates = [s for i in order for s in split_sentences(chunks[i - 1].chunk.text, i)]
        if not candidates:
            return []
        scores = self.scorer(question, [s.text for s in candidates])
        ranked = sorted(zip(scores, candidates, strict=True), key=lambda x: x[0], reverse=True)
        keep: list[Sentence] = []
        seen: set[str] = set()
        if must_include is not None:
            keep.append(must_include)
            seen.add(_key(must_include.text))
        for score, s in ranked:  # chunks overlap, so the same sentence can appear twice: keep one
            if score >= MIN_SENTENCE_SCORE and _key(s.text) not in seen and len(keep) < self.max_sentences:
                keep.append(s)
                seen.add(_key(s.text))
        position = {i: n for n, i in enumerate(order)}
        return sorted(keep, key=lambda s: (position[s.source], s.start))

    # ---- answering ----

    def answer(self, question: str, chunks: list[RetrievedChunk], qtype: QuestionType) -> Iterator[AnswerEvent]:
        """Yield the template-composed, fully quoted answer (or NO_ANSWER)."""
        if not chunks:
            yield AnswerEvent("token", NO_ANSWER_TOKEN)
            return
        if qtype == "summary":
            yield from self._summary(chunks)
            return
        if qtype == "explain":
            yield from self._explain(question, chunks)
            return
        chosen = self.choose(question, chunks)
        if chosen is None or chosen[1] < self.min_qa_score:
            yield AnswerEvent("token", NO_ANSWER_TOKEN)
            return
        span, qa = chosen
        text = chunks[span.source - 1].chunk.text
        start, end = span.start, span.end
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        span_text = text[start:end]
        home = next(
            (s for s in split_sentences(text, span.source) if s.start <= start < s.end),
            None,
        )
        sentences = self._supporting(question, chunks, span.source, must_include=home)
        body = [f"**Answer:** {span_text} [{span.source}]", "", "**From the document:**"]
        body += [_quote(s) for s in sentences]
        yield AnswerEvent(
            "meta",
            data={"span": span_text, "span_source": span.source, "span_char_range": [start, end], "confidence": qa},
        )
        yield AnswerEvent("token", "\n".join(body))

    def _summary(self, chunks: list[RetrievedChunk]) -> Iterator[AnswerEvent]:
        picked: list[int] = []
        seen_pages: set[tuple[str, int]] = set()
        for i, c in enumerate(chunks, start=1):  # best-spread: one chunk per (document, page)
            key = (c.chunk.doc_id, c.chunk.page_start)
            if key not in seen_pages:
                picked.append(i)
                seen_pages.add(key)
            if len(picked) == 3:
                break
        lines = ["**Excerpt summary** (opening sentences of the most relevant sections):"]
        for i in picked:
            lines += [_quote(s) for s in split_sentences(chunks[i - 1].chunk.text, i)[:2]]
        yield AnswerEvent("meta", data={"hint": SUMMARY_HINT})
        yield AnswerEvent("token", "\n".join(lines))

    def _explain(self, question: str, chunks: list[RetrievedChunk]) -> Iterator[AnswerEvent]:
        sentences = self._supporting(question, chunks, 1)
        if not sentences:
            yield AnswerEvent("token", NO_ANSWER_TOKEN)
            return
        lines = ["**From the document:**", *[_quote(s) for s in sentences]]
        yield AnswerEvent("meta", data={"hint": EXPLAIN_HINT})
        yield AnswerEvent("token", "\n".join(lines))
