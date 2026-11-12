"""Tiny rule-based question classifier: factoid | summary | explain (docs/ANSWER_MODES.md section 2.5)."""

import re

from app.generation.base import QuestionType

_SUMMARY = re.compile(
    r"\b(summari[sz]e|summary|overview|tl;?dr|gist|main points|key points|key takeaways|"
    r"what(?:'s| is) (?:this|the|that) (?:document|pdf|file|manual|guide|report|paper)\b.*\babout|"
    r"what is (?:this|it) about|what does (?:this|the) (?:document|pdf|manual|guide) (?:cover|say))\b",
    re.I,
)
_EXPLAIN = re.compile(
    r"^\s*why\b|\bwhy (?:does|do|is|are|did|would|should)\b|\bcompare\b|\bcomparison\b|"
    r"\bdifference(?:s)? between\b|\bhow (?:does|do) .+ differ\b|\bversus\b|\bvs\.?\s|\bpros and cons\b|"
    r"\badvantages? and disadvantages?\b",
    re.I,
)


def classify(question: str) -> QuestionType:
    """Classify a question so Local mode can pick the right extractive strategy."""
    if _SUMMARY.search(question):
        return "summary"
    if _EXPLAIN.search(question):
        return "explain"
    return "factoid"
