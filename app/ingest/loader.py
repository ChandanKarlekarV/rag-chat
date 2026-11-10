"""Validate uploaded bytes and extract raw per-page text with PyMuPDF."""

import hashlib

import pymupdf

from app.schemas import PageText


class IngestError(ValueError):
    """Raised when an upload cannot be ingested. The message is safe to show to users."""


PDF_MAGIC = b"%PDF-"


def sha256_of(data: bytes) -> str:
    """Return the hex SHA-256 digest used to dedupe uploads."""
    return hashlib.sha256(data).hexdigest()


def validate_pdf_bytes(data: bytes, filename: str, max_mb: int) -> None:
    """Reject empty, oversized or non-PDF uploads with a user-friendly message."""
    if not data:
        raise IngestError("The uploaded file is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise IngestError(f"'{filename}' is larger than the {max_mb} MB limit.")
    if not filename.lower().endswith(".pdf") or not data[:1024].lstrip().startswith(PDF_MAGIC):
        raise IngestError(f"'{filename}' is not a PDF file.")


def load_pdf(data: bytes, filename: str) -> list[PageText]:
    """Extract raw text for every page (1-indexed). Raises IngestError on unreadable PDFs."""
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several unrelated types for corrupt files
        raise IngestError(f"'{filename}' could not be opened as a PDF.") from exc
    with doc:
        if doc.needs_pass:
            raise IngestError(f"'{filename}' is password-protected.")
        # Plain text order (not layout blocks): a heading then runs into its first sentence, which
        # measured better on the eval set (the heading gives the chunk topic context). See README.
        return [PageText(page=i + 1, text=page.get_text("text")) for i, page in enumerate(doc)]
