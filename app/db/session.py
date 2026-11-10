"""Engine, session factory and schema migrations (Alembic)."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event, inspect
from sqlalchemy.orm import Session, sessionmaker

ROOT = Path(__file__).resolve().parents[2]


class Database:
    """Owns the engine and hands out short-lived sessions. Works with SQLite and Postgres."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.is_sqlite = url.startswith("sqlite")
        self.engine: Engine = create_engine(
            url,
            connect_args={"check_same_thread": False} if self.is_sqlite else {},
            pool_pre_ping=not self.is_sqlite,
        )
        if self.is_sqlite:
            event.listen(self.engine, "connect", _sqlite_pragmas)
        self._factory = sessionmaker(self.engine, expire_on_commit=False)

    def migrate(self) -> None:
        """Bring the schema to the latest Alembic revision.

        Databases created by earlier versions with `create_all` (tables but no `alembic_version`)
        are stamped at the initial revision first, then upgraded.
        """
        cfg = Config(str(ROOT / "alembic.ini"))
        cfg.set_main_option("script_location", str(ROOT / "migrations"))
        with self.engine.begin() as conn:
            cfg.attributes["connection"] = conn
            insp = inspect(conn)
            tables = set(insp.get_table_names())
            if "users" in tables and "alembic_version" not in tables:
                user_columns = {c["name"] for c in insp.get_columns("users")}
                command.stamp(cfg, "0002" if "preferred_mode" in user_columns else "0001")
            command.upgrade(cfg, "head")

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


def _sqlite_pragmas(dbapi_conn, _record) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")  # enforce ON DELETE CASCADE
    cur.execute("PRAGMA journal_mode=WAL")  # readers don't block the ingest writer
    cur.close()
