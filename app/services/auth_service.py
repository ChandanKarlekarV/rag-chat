"""Accounts, password hashing (argon2id), JWT sessions and guest lifecycle."""

import logging
import re
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from app.config import Settings
from app.db.models import User
from app.db.repositories import UserRepo
from app.db.session import Database
from app.errors import AppError, Conflict, Unauthorized
from app.services.ingest_service import IngestService

log = logging.getLogger("rag-chat.auth")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_ALGO = "HS256"
MIN_PASSWORD = 8


def normalize_email(email: str) -> str:
    """Trim and lowercase an email address."""
    return email.strip().lower()


class AuthService:
    """Registers users, verifies passwords and issues/validates session tokens."""

    def __init__(self, db: Database, settings: Settings, ingest: IngestService, sample_pdf: Path) -> None:
        self.db = db
        self.settings = settings
        self.ingest = ingest
        self.sample_pdf = sample_pdf
        self._hasher = PasswordHasher()  # argon2id with library defaults

    # ---- tokens ----

    def issue_token(self, user_id: str) -> str:
        """Signed JWT valid for `SESSION_DAYS`."""
        now = datetime.now(UTC)
        payload = {"sub": user_id, "iat": now, "exp": now + timedelta(days=self.settings.session_days)}
        return jwt.encode(payload, self.settings.secret_key, algorithm=_ALGO)

    def user_from_token(self, token: str | None) -> User:
        """Resolve the user for a session token or raise `Unauthorized`."""
        if not token:
            raise Unauthorized("not_signed_in", "Please sign in to continue.")
        try:
            payload = jwt.decode(token, self.settings.secret_key, algorithms=[_ALGO])
        except jwt.ExpiredSignatureError as exc:
            raise Unauthorized("session_expired", "Your session expired. Please sign in again.") from exc
        except jwt.InvalidTokenError as exc:
            raise Unauthorized("not_signed_in", "Please sign in to continue.") from exc
        with self.db.session() as s:
            user = UserRepo(s).get(str(payload.get("sub", "")))
        if user is None:
            raise Unauthorized("not_signed_in", "Your account no longer exists. Please sign in again.")
        return user

    # ---- accounts ----

    def register(self, email: str, password: str) -> User:
        """Create an account. Raises friendly errors for bad input or a taken email."""
        email = normalize_email(email)
        if not _EMAIL.match(email):
            raise AppError("invalid_email", "Enter a valid email address, like name@example.com.")
        if len(password) < MIN_PASSWORD:
            raise AppError("weak_password", f"Use at least {MIN_PASSWORD} characters for your password.")
        with self.db.session() as s:
            repo = UserRepo(s)
            if repo.by_email(email):
                raise Conflict("email_taken", "An account with this email already exists. Try signing in.")
            user = repo.add(User(email=email, password_hash=self._hasher.hash(password)))
        log.info("user registered", extra={"user_id": user.id})
        return user

    def login(self, email: str, password: str) -> User:
        """Verify credentials; the same message is used for unknown email and wrong password."""
        bad = Unauthorized("bad_credentials", "That email and password don't match. Please try again.")
        with self.db.session() as s:
            repo = UserRepo(s)
            user = repo.by_email(normalize_email(email))
            if user is None or user.is_guest:
                self._hasher.hash(password)  # similar timing for unknown accounts
                raise bad
            try:
                self._hasher.verify(user.password_hash, password)
            except (VerifyMismatchError, VerificationError, InvalidHashError) as exc:
                raise bad from exc
            if self._hasher.check_needs_rehash(user.password_hash):
                user.password_hash = self._hasher.hash(password)
        return user

    def create_guest(self) -> User:
        """Create a throwaway account and index the bundled sample PDF so it is ready on arrival."""
        with self.db.session() as s:
            user = UserRepo(s).add(
                User(
                    email=f"guest-{secrets.token_hex(6)}@guest.invalid",
                    password_hash=self._hasher.hash(secrets.token_urlsafe(24)),
                    is_guest=True,
                )
            )
        if self.sample_pdf.exists():
            self.ingest.ingest_now(user, self.sample_pdf.read_bytes(), self.sample_pdf.name)
        log.info("guest created", extra={"user_id": user.id})
        return user

    def cleanup_guests(self) -> int:
        """Delete guests older than `GUEST_TTL_HOURS` with all their data. Returns how many."""
        cutoff = datetime.now(UTC) - timedelta(hours=self.settings.guest_ttl_hours)
        with self.db.session() as s:
            ids = UserRepo(s).expired_guest_ids(cutoff)
        for user_id in ids:
            self.delete_user(user_id)
        if ids:
            log.info("expired guests removed", extra={"count": len(ids)})
        return len(ids)

    def delete_user(self, user_id: str) -> None:
        """Remove a user and everything they own (rows, vectors, BM25, files)."""
        self.ingest.purge_user(user_id)
        with self.db.session() as s:
            UserRepo(s).delete(user_id)
