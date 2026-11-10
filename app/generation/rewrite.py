"""Turn follow-up questions ("and how heavy is it?") into standalone search queries.

* `HeuristicRewriter` (no key): if the question leans on context (pronouns, "and…", "what
  about…", very short), it appends the key terms of the previous question to the *search query*.
  The question shown to the user and given to the reader is unchanged.
* `LLMRewriter` (with Claude): asks the model to rewrite the question into a standalone one;
  falls back to the heuristic if the call fails or returns nothing usable.
"""

import logging
import re
from dataclasses import dataclass
from typing import Protocol

from app.generation.llm_base import LLMClient, LLMError

log = logging.getLogger("rag-chat.rewrite")

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*")
_CONTEXT_WORDS = frozenset(
    [
        "it",
        "its",
        "they",
        "them",
        "their",
        "theirs",
        "this",
        "that",
        "these",
        "those",
        "there",
        "he",
        "she",
        "his",
        "her",
        "one",
        "ones",
    ]
)
_LEADS = ("and ", "also ", "what about", "how about", "and what", "same for", "what else")
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "could",
        "do",
        "does",
        "did",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "should",
        "so",
        "tell",
        "than",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "about",
        "also",
        "any",
        "much",
        "many",
        "more",
        "please",
    ]
)
REWRITE_SYSTEM = (
    "Rewrite the user's latest question so it can be understood without the conversation. "
    "Resolve pronouns and references using earlier turns. Keep the meaning and language. "
    "Do not answer it. Reply with the rewritten question only."
)


@dataclass(frozen=True)
class Turn:
    """One previous exchange."""

    question: str
    answer: str


@dataclass(frozen=True)
class Rewrite:
    """What to search for and what to ask the answerer."""

    search_query: str
    question: str
    rewritten: bool


class QueryRewriter(Protocol):
    def rewrite(self, question: str, history: list[Turn]) -> Rewrite: ...


def needs_context(question: str) -> bool:
    """Heuristic: does this question depend on earlier turns?"""
    q = question.strip().lower()
    words = _WORD.findall(q)
    content = [w for w in words if w not in _STOP]
    return any(w in _CONTEXT_WORDS for w in words) or q.startswith(_LEADS) or len(content) <= 1


def key_terms(text: str) -> list[str]:
    """Content words of a question, in order, without duplicates."""
    seen: dict[str, None] = {}
    for w in _WORD.findall(text):
        if w.lower() not in _STOP and w.lower() not in _CONTEXT_WORDS:
            seen.setdefault(w, None)
    return list(seen)


class HeuristicRewriter:
    """Expands the search query with the previous question's key terms when needed."""

    def rewrite(self, question: str, history: list[Turn]) -> Rewrite:
        """Return the (possibly expanded) search query; the question itself is unchanged."""
        if not history or not needs_context(question):
            return Rewrite(question, question, False)
        present = {w.lower() for w in _WORD.findall(question)}
        extra = [w for w in key_terms(history[-1].question) if w.lower() not in present]
        if not extra:
            return Rewrite(question, question, False)
        return Rewrite(f"{question} {' '.join(extra)}", question, True)


class LLMRewriter:
    """Standalone-question rewriting with an LLM, falling back to the heuristic."""

    def __init__(self, llm: LLMClient, max_turns: int = 3) -> None:
        self.llm = llm
        self.max_turns = max_turns
        self.fallback = HeuristicRewriter()

    def rewrite(self, question: str, history: list[Turn]) -> Rewrite:
        """Rewrite with the model when there is history; use the result for search and answering."""
        if not history:
            return Rewrite(question, question, False)
        turns = "\n".join(f"User: {t.question}\nAssistant: {t.answer[:300]}" for t in history[-self.max_turns :])
        prompt = f"<conversation>\n{turns}\n</conversation>\n\nLatest question: {question}"
        try:
            text = self.llm.complete(REWRITE_SYSTEM, prompt, max_tokens=200).strip().strip('"')
        except LLMError:
            log.warning("rewrite failed; using heuristic")
            return self.fallback.rewrite(question, history)
        if not text or len(text) > 500 or "\n" in text:
            return self.fallback.rewrite(question, history)
        return Rewrite(text, text, text.lower() != question.lower())
