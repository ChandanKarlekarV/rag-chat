"""Engine and session factory."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base


class Database:
    """Owns the engine and hands out short-lived sessions."""

    def __init__(self, url: str) -> None:
        is_sqlite = url.startswith("sqlite")
        self.engine: Engine = create_engine(
            url, connect_args={"check_same_thread": False} if is_sqlite else {}
        )
        if is_sqlite:
            event.listen(self.engine, "connect", _sqlite_pragmas)
        self._factory = sessionmaker(self.engine, expire_on_commit=False)

    def create_all(self) -> None:
        """Create tables if missing (Alembic migrations are future work)."""
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Transactional scope: commit on success, roll back on error."""
        with self._factory() as s:
            try:
                yield s
                s.commit()
            except Exception:
                s.rollback()
                raise


def _sqlite_pragmas(dbapi_conn, _record) -> None:  # noqa: ANN001 - DBAPI connection type is driver-specific
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")  # enforce ON DELETE CASCADE
    cur.execute("PRAGMA journal_mode=WAL")  # readers don't block the ingest writer
    cur.close()
