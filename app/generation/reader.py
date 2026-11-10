"""Extractive QA reader used by Local mode: finds the answer span in a passage, or "no answer".

Uses an extractive question-answering model trained on SQuAD 2.0 (which includes unanswerable
questions; default `deepset/deberta-v3-large-squad2`). For each retrieved passage it scores every
candidate span against the model's "no answer" score (the [CLS] position) and reports the margin.

Implemented with `AutoTokenizer` / `AutoModelForQuestionAnswering` (verified against the installed
transformers 5.22.2). The `question-answering` pipeline that ANSWER_MODES.md mentions is no longer
in transformers 5.x (absent from the package and from the current pipeline docs), so the
`handle_impossible_answer` behaviour is implemented here explicitly via the no-answer margin.
Runs on the GPU when one is available (see app/devices.py).
"""

from dataclasses import dataclass
from functools import lru_cache

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from app.schemas import RetrievedChunk

MAX_ANSWER_TOKENS = 30
TOP_CANDIDATES = 20


@dataclass(frozen=True)
class Span:
    """The best answer found in one passage."""

    source: int  # 1-based source number
    start: int  # character offsets into the passage text
    end: int
    margin: float  # span score minus the "no answer" score


class ExtractiveReader:
    """Finds the answer span in retrieved passages, or reports that there is none."""

    def __init__(self, model_name: str, device: str = "cpu", max_length: int = 384) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device).eval()

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
        with torch.inference_mode():
            out = self._model(**{k: v.to(self.device) for k, v in enc.items()})
        start_all, end_all = out.start_logits.float().cpu(), out.end_logits.float().cpu()
        spans: list[Span] = []
        for i in range(len(contexts)):
            seq = enc.sequence_ids(i)
            ctx_tokens = [t for t, sid in enumerate(seq) if sid == 1]
            if not ctx_tokens:
                continue
            start_logits, end_logits = start_all[i], end_all[i]
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


@lru_cache
def get_reader(model_name: str, device: str = "cpu") -> ExtractiveReader:
    """Return the singleton reader for `model_name` on `device` (loaded once per process)."""
    return ExtractiveReader(model_name, device)
