"""Repositories: every query that touches user data takes `user_id` and filters by it."""

from datetime import datetime

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.db.models import Conversation, Document, Message, User


class UserRepo:
    """User accounts."""

    def __init__(self, s: Session) -> None:
        self.s = s

    def by_email(self, email: str) -> User | None:
        """Find a user by (normalised) email."""
        return self.s.scalar(select(User).where(User.email == email))

    def get(self, user_id: str) -> User | None:
        """Find a user by id."""
        return self.s.get(User, user_id)

    def add(self, user: User) -> User:
        """Insert a user and flush to obtain defaults."""
        self.s.add(user)
        self.s.flush()
        return user

    def expired_guest_ids(self, before: datetime) -> list[str]:
        """Ids of guest accounts created before `before`."""
        return list(self.s.scalars(select(User.id).where(User.is_guest.is_(True), User.created_at < before)))

    def delete(self, user_id: str) -> None:
        """Delete a user; documents, conversations and messages cascade."""
        self.s.execute(delete(User).where(User.id == user_id))


class DocumentRepo:
    """Documents, always scoped to an owner."""

    def __init__(self, s: Session) -> None:
        self.s = s

    def for_user(self, user_id: str) -> list[Document]:
        """The user's documents, newest first."""
        q = select(Document).where(Document.user_id == user_id).order_by(Document.created_at.desc())
        return list(self.s.scalars(q))

    def get(self, user_id: str, doc_id: str) -> Document | None:
        """The document if it exists *and* belongs to the user."""
        return self.s.scalar(select(Document).where(Document.id == doc_id, Document.user_id == user_id))

    def by_sha(self, user_id: str, sha256: str) -> Document | None:
        """The user's document with this content hash, if any."""
        return self.s.scalar(select(Document).where(Document.user_id == user_id, Document.sha256 == sha256))

    def count(self, user_id: str) -> int:
        """Number of documents owned by the user."""
        return int(self.s.scalar(select(func.count()).where(Document.user_id == user_id)) or 0)

    def ready_ids(self, user_id: str) -> list[str]:
        """Ids of the user's searchable documents."""
        q = select(Document.id).where(Document.user_id == user_id, Document.status == "ready")
        return list(self.s.scalars(q))

    def add(self, doc: Document) -> Document:
        """Insert a document row."""
        self.s.add(doc)
        self.s.flush()
        return doc

    def by_id_unscoped(self, doc_id: str) -> Document | None:
        """Internal use by the ingest worker only (it already knows the owner)."""
        return self.s.get(Document, doc_id)

    def delete(self, doc: Document) -> None:
        """Delete a document row."""
        self.s.delete(doc)


class ConversationRepo:
    """Conversations and messages, always scoped to an owner."""

    def __init__(self, s: Session) -> None:
        self.s = s

    def for_user(self, user_id: str) -> list[Conversation]:
        """The user's conversations, newest first."""
        q = select(Conversation).where(Conversation.user_id == user_id).order_by(Conversation.created_at.desc())
        return list(self.s.scalars(q))

    def get(self, user_id: str, conv_id: str) -> Conversation | None:
        """The conversation if it exists *and* belongs to the user."""
        return self.s.scalar(select(Conversation).where(Conversation.id == conv_id, Conversation.user_id == user_id))

    def add(self, conv: Conversation) -> Conversation:
        """Insert a conversation."""
        self.s.add(conv)
        self.s.flush()
        return conv

    def add_message(self, msg: Message) -> Message:
        """Append a message."""
        self.s.add(msg)
        self.s.flush()
        return msg

    def delete(self, conv: Conversation) -> None:
        """Delete a conversation and its messages."""
        self.s.delete(conv)
