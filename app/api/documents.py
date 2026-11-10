"""Document routes: upload (background ingest), list, status, delete."""

from fastapi import APIRouter, BackgroundTasks, Depends, File, Response, UploadFile

from app.api.deps import get_container, get_current_user, require_fetch_header
from app.db.models import Document, User
from app.schemas import DocumentOut, DocumentStatus
from app.services.container import Container

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentOut, status_code=202, dependencies=[Depends(require_fetch_header)])
def upload(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    c: Container = Depends(get_container),
) -> Document:
    """Accept a PDF and start ingesting it in the background; poll `/documents/{id}/status`."""
    data = file.file.read(c.settings.max_upload_mb * 1024 * 1024 + 1)
    doc = c.ingest.submit(user, data, file.filename or "document.pdf")
    background.add_task(c.ingest.process, doc.id)
    return doc


@router.get("", response_model=list[DocumentOut])
def list_documents(user: User = Depends(get_current_user), c: Container = Depends(get_container)) -> list[Document]:
    """The signed-in user's documents, newest first."""
    return c.ingest.list_documents(user.id)


@router.get("/{doc_id}/status", response_model=DocumentStatus)
def status(doc_id: str, user: User = Depends(get_current_user), c: Container = Depends(get_container)) -> DocumentStatus:
    """Ingest stage and progress for one document."""
    return c.ingest.status(user.id, doc_id)


@router.delete("/{doc_id}", status_code=204, dependencies=[Depends(require_fetch_header)])
def delete(doc_id: str, user: User = Depends(get_current_user), c: Container = Depends(get_container)) -> Response:
    """Delete a document, its chunks and its keyword-index entries."""
    c.ingest.delete(user.id, doc_id)
    return Response(status_code=204)
