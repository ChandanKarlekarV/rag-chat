"""Prompt templates for grounded, cited answers."""

from app.schemas import RetrievedChunk

NO_ANSWER_TOKEN = "NO_ANSWER"

SYSTEM_PROMPT = f"""You are a careful assistant that answers questions about the user's documents.
You answer ONLY using the numbered sources provided in the user message.
- Cite sources inline like [1] or [2][3] directly after each claim they support.
- Only cite source numbers that exist in the list.
- If the sources do not contain the answer, reply with exactly: {NO_ANSWER_TOKEN}
- Never use outside knowledge, even if you know the answer.
- Be concise: a few sentences or a short list. Use plain text or simple Markdown."""


def page_label(page_start: int, page_end: int) -> str:
    """Render 'p.3' or 'pp.3-4'."""
    return f"p.{page_start}" if page_start == page_end else f"pp.{page_start}-{page_end}"


def build_user_prompt(question: str, sources: list[RetrievedChunk]) -> str:
    """Render the numbered SOURCES block followed by the question."""
    blocks = [
        f"[{i}] ({s.chunk.filename}, {page_label(s.chunk.page_start, s.chunk.page_end)}) {s.chunk.text}"
        for i, s in enumerate(sources, start=1)
    ]
    return "SOURCES:\n" + "\n\n".join(blocks) + f"\n\nQUESTION: {question}"
