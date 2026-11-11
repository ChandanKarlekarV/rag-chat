"""Study material route: notes, sticky notes, mind map, flows, key terms and flashcards for one PDF."""

from fastapi import APIRouter, Depends

from app.api.deps import get_container, get_current_user, key_input, require_fetch_header
from app.db.models import User
from app.generation.router import KeyInput
from app.schemas import StudyPack, StudyRequest
from app.services.container import Container

router = APIRouter(tags=["study"])


@router.post("/study", response_model=StudyPack, dependencies=[Depends(require_fetch_header)])
def study(
    body: StudyRequest,
    user: User = Depends(get_current_user),
    c: Container = Depends(get_container),
    key: KeyInput = Depends(key_input),
) -> StudyPack:
    """Build the chosen kinds of study material with the chosen answer mode (Local needs no key)."""
    c.ask_limiter.hit(f"study:{user.id}", "You're making study material very quickly. Please wait a moment.")
    engine = c.engine(key, body.mode, user.preferred_mode, user.ollama_model)
    return c.study.build(user.id, body, engine)
