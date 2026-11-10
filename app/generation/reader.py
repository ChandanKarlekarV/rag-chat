"""Local extractive reader: answers from the PDF text itself, no API key.

Uses an extractive question-answering model (default `deepset/roberta-base-squad2`, trained on
SQuAD 2.0, which includes unanswerable questions). For each retrieved passage it scores every
candidate answer span against the model's "no answer" score; the best span wins only if it beats
"no answer" by `min_score`. The reply is the sentence that contains the span, quoted verbatim,
with the span in bold and a citation, so it can never contain words that are not in the document.

Implemented with `AutoTokenizer` / `AutoModelForQuestionAnswering` (verified against the installed
transformers 5.22.2) rather than the `pipeline` helper, so span scoring and the no-answer rule are
explicit and testable.
"""

import re
from dataclasses import dataclass
from functools import lru_cache

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from app.schemas import RetrievedChunk

MAX_ANSWER_TOKENS = 30
TOP_CANDIDATES = 20
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_YES_NO = re.compile(r"^(can|could|does|do|did|is|are|was|were|should|will|would|may|must|has|have)\b", re.I)


@dataclass(frozen=True)
class Span:
    """The best answer found in one passage."""

    source: int  # 1-based source number
    start: int  # character offsets into the passage text
    end: int
    margin: float  # span score minus the "no answer" score


class ExtractiveReader:
    """Finds the answer span in retrieved passages, or reports that there is none."""

    def __init__(self, model_name: str, max_length: int = 384) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForQuestionAnswering.from_pretrained(model_name).eval()

    def best_spans(self, question: str, sources: list[RetrievedChunk]) -> list[Span]:
        """Best span per passage, sorted by margin (highest first)."""
        if not sources:
            return []
        contexts = [s.chunk.text for s in sources]
        enc = self._tok(
            [question] * len(contexts),
            contexts,
            truncation="only_second",
            max_length=self.max_length,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")
        with torch.no_grad():
            out = self._model(**enc)
        spans: list[Span] = []
        for i in range(len(contexts)):
            seq = enc.sequence_ids(i)
            ctx_tokens = [t for t, sid in enumerate(seq) if sid == 1]
            if not ctx_tokens:
                continue
            start_logits, end_logits = out.start_logits[i], out.end_logits[i]
            null = float(start_logits[0] + end_logits[0])
            span = _best_span(start_logits, end_logits, set(ctx_tokens))
            if span is None:
                continue
            score, a, b = span
            spans.append(Span(i + 1, int(offsets[i][a][0]), int(offsets[i][b][1]), score - null))
        spans.sort(key=lambda s: s.margin, reverse=True)
        return spans


def _best_span(start: torch.Tensor, end: torch.Tensor, allowed: set[int]) -> tuple[float, int, int] | None:
    """Highest start+end logit pair inside the passage, end >= start, at most MAX_ANSWER_TOKENS long."""
    k = min(TOP_CANDIDATES, start.shape[0])
    starts = [int(t) for t in torch.topk(start, k).indices if int(t) in allowed]
    ends = [int(t) for t in torch.topk(end, k).indices if int(t) in allowed]
    best: tuple[float, int, int] | None = None
    for a in starts:
        for b in ends:
            if a <= b < a + MAX_ANSWER_TOKENS:
                score = float(start[a] + end[b])
                if best is None or score > best[0]:
                    best = (score, a, b)
    return best


def sentence_around(text: str, start: int, end: int, extra: int = 0) -> tuple[str, int, int]:
    """The sentence(s) of `text` containing [start, end) plus `extra` following sentences,
    and the span offsets inside the returned text."""
    bounds = [0] + [m.end() for m in _SENTENCE_END.finditer(text)] + [len(text)]
    s_start = max(b for b in bounds if b <= start)
    after = [b for b in bounds if b >= end]
    s_end = after[min(extra, len(after) - 1)]
    sentence = text[s_start:s_end]
    lead = len(sentence) - len(sentence.lstrip())
    sentence = sentence.strip()
    return sentence, start - s_start - lead, end - s_start - lead


def is_yes_no(question: str) -> bool:
    """Questions like "Can I…?" / "Does it…?" whose answer often lies in the qualifying next sentence."""
    return bool(_YES_NO.match(question.strip()))


def format_answer(span: Span, sources: list[RetrievedChunk], question: str = "") -> str:
    """Quote the supporting sentence verbatim (plus the next one for yes/no questions),
    bold the answer span, and cite the source."""
    text = sources[span.source - 1].chunk.text
    sentence, a, b = sentence_around(text, span.start, span.end, extra=1 if is_yes_no(question) else 0)
    if 0 <= a < b <= len(sentence):
        inner = sentence[a:b]
        core = inner.strip()
        if core:  # keep surrounding whitespace outside the bold markers
            lead = inner[: len(inner) - len(inner.lstrip())]
            trail = inner[len(inner.rstrip()) :]
            sentence = f"{sentence[:a]}{lead}**{core}**{trail}{sentence[b:]}"
    return f"{sentence} [{span.source}]"


@lru_cache
def get_reader(model_name: str) -> ExtractiveReader:
    """Return the singleton reader for `model_name` (loaded once per process)."""
    return ExtractiveReader(model_name)
