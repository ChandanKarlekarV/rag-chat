"""Claude API provider (API mode). The HTTP client itself lives in llm_anthropic.py."""

from app.generation.base import AnswerMode
from app.generation.generative import GenerativeProvider
from app.generation.llm_base import LLMClient


class AnthropicProvider(GenerativeProvider):
    """Generative answers from Claude with the grounded, cited prompt."""

    def __init__(self, llm: LLMClient) -> None:
        super().__init__(llm, AnswerMode.API)
