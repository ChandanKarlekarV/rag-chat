"""Multi-server backends: Redis limits/progress/locks, shared BM25 files, migrations, Postgres, Chroma.

Uses fakeredis by default. Set TEST_REDIS_URL / TEST_DATABASE_URL / TEST_CHROMA_URL to run the
same tests against real Redis, Postgres and a Chroma server (CI does this for Redis and Postgres).
"""

import os
import threading
import time
import uuid

import fakeredis
import pytest
from sqlalchemy import create_engine, inspect, text

from app.db.models import Base, User
from app.db.repositories import UserRepo
from app.db.session import Database
from app.errors import TooMany
from app.retrieval.bm25_store import BM25Store
from app.retrieval.vector_store import VectorStore
from app.schemas import Chunk
from app.services.coordination import RedisLocks, RedisProgress
from app.services.rate_limit import RedisRateLimiter
from tests.conftest import CSRF, make_harness, register, upload


@pytest.fixture
def redis_client():
    url = os.environ.get("TEST_REDIS_URL")
    if url:
        import redis

        client = redis.Redis.from_url(url)
        client.flushdb()
        yield client
        client.flushdb()
    else:
        yield fakeredis.FakeRedis(server=fakeredis.FakeServer())


def _chunk(cid: str, doc: str, text_: str) -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id=doc, filename="f.pdf", page_start=1, page_end=1, index=0, text=text_, token_count=3
    )


# ---------- Redis primitives ----------


def test_redis_rate_limiter_is_shared_between_workers(redis_client):
    worker_a = RedisRateLimiter(redis_client, limit=3)
    worker_b = RedisRateLimiter(redis_client, limit=3)
    worker_a.hit("ask:u1", "slow down")
    worker_b.hit("ask:u1", "slow down")
    worker_a.hit("ask:u1", "slow down")
    with pytest.raises(TooMany):
        worker_b.hit("ask:u1", "slow down")
    worker_b.hit("ask:u2", "slow down")  # other keys are independent


def test_redis_rate_limiter_window_expires(redis_client):
    limiter = RedisRateLimiter(redis_client, limit=1, window_s=0.3)
    limiter.hit("k", "m")
    with pytest.raises(TooMany):
        limiter.hit("k", "m")
    time.sleep(0.4)
    limiter.hit("k", "m")


def test_redis_progress_round_trip(redis_client):
    store = RedisProgress(redis_client)
    assert store.get("d1") is None
    store.set("d1", "understanding", 60)
    assert RedisProgress(redis_client).get("d1") == ("understanding", 60)  # visible to another worker
    store.delete("d1")
    assert store.get("d1") is None


def test_redis_lock_serialises_writers(redis_client):
    locks = RedisLocks(redis_client)
    inside, overlaps = [], []

    def work():
        with locks.lock("user:u1"):
            if inside:
                overlaps.append(1)
            inside.append(1)
            time.sleep(0.05)
            inside.pop()

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert overlaps == []


# ---------- app wired to Redis ----------


def test_app_with_redis_shares_ask_limit_and_progress(tmp_path, sample_pdf, redis_client):
    for harness in make_harness(tmp_path, sample_pdf, ask_rate_per_min=1, redis_client=redis_client):
        c = harness.client
        register(c)
        doc_id = upload(c, sample_pdf).json()["id"]
        assert c.get(f"/documents/{doc_id}/status").json()["status"] == "ready"
        assert c.get("/health").json()["redis"] is True
        container = c.app.state.container
        assert isinstance(container.ingest.progress, RedisProgress)
        r1 = c.post(
            "/chat/ask",
            json={"question": "How much energy?"},
            headers=CSRF | {"X-Anthropic-Api-Key": "sk-ant-test-good"},
        )
        r2 = c.post("/chat/ask", json={"question": "How much energy?"}, headers=CSRF)
        assert r1.status_code == 200 and r2.status_code == 429


# ---------- BM25 shared between workers ----------


def test_bm25_reloads_when_another_worker_rebuilds(tmp_path):
    worker_a, worker_b = BM25Store(tmp_path), BM25Store(tmp_path)
    worker_a.rebuild(
        "u1",
        [_chunk("a", "d", "solar inverter manual"), _chunk("b", "d", "battery warranty"), _chunk("c", "d", "x y z")],
    )
    assert worker_b.search("u1", "warranty", 5)[0][0] == "b"
    time.sleep(0.01)
    worker_a.rebuild(
        "u1", [_chunk("e", "d", "charger cable"), _chunk("f", "d", "red ring fault"), _chunk("g", "d", "x y z")]
    )
    assert worker_b.search("u1", "warranty", 5) == []  # B noticed the newer file
    assert worker_b.search("u1", "fault", 5)[0][0] == "f"
    assert not list(tmp_path.glob("*.tmp"))  # atomic write left no temp files


# ---------- migrations ----------


def test_migrations_create_schema_and_are_idempotent(tmp_path):
    db = Database(f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    db.migrate()
    db.migrate()
    tables = set(inspect(db.engine).get_table_names())
    assert {"users", "documents", "conversations", "messages", "alembic_version"} <= tables
    with db.engine.connect() as conn:
        assert conn.execute(text("SELECT version_num FROM alembic_version")).scalar() == "0002"
    cols = {c["name"] for c in inspect(db.engine).get_columns("messages")}
    assert {"mode", "label"} <= cols


def test_unversioned_v1_schema_is_upgraded_with_data_kept(tmp_path):
    """A database at the 0001 schema without alembic_version (pre-migration era) gains the 0002 columns."""
    url = f"sqlite:///{(tmp_path / 'v1.db').as_posix()}"
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE users (id VARCHAR(32) PRIMARY KEY, email VARCHAR(320) UNIQUE NOT NULL, "
                "password_hash VARCHAR(255) NOT NULL, is_guest BOOLEAN NOT NULL, created_at DATETIME NOT NULL)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE conversations (id VARCHAR(32) PRIMARY KEY, user_id VARCHAR(32) NOT NULL, "
                "title VARCHAR(120) NOT NULL, created_at DATETIME NOT NULL)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE messages (id VARCHAR(32) PRIMARY KEY, conversation_id VARCHAR(32) NOT NULL, "
                "role VARCHAR(16) NOT NULL, content TEXT NOT NULL, answerable BOOLEAN, citations_json TEXT, "
                "created_at DATETIME NOT NULL)"
            )
        )
        conn.execute(text("INSERT INTO users VALUES ('u1', 'old@example.com', 'x', 0, '2026-01-01 00:00:00')"))
    Database(url).migrate()
    db = Database(url)
    with db.session() as s:
        user = UserRepo(s).by_email("old@example.com")
        assert user is not None and user.preferred_mode == "auto"  # data kept, new column defaulted


def test_legacy_create_all_database_is_stamped_not_recreated(tmp_path):
    url = f"sqlite:///{(tmp_path / 'legacy.db').as_posix()}"
    Base.metadata.create_all(create_engine(url))
    db = Database(url)
    with db.session() as s:
        UserRepo(s).add(User(email="keep@example.com", password_hash="x"))
    db.migrate()
    with db.session() as s:
        assert UserRepo(s).by_email("keep@example.com") is not None  # data survived


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="set TEST_DATABASE_URL to test Postgres")
def test_postgres_end_to_end(tmp_path, sample_pdf):
    url = os.environ["TEST_DATABASE_URL"]
    with create_engine(url).begin() as conn:  # clean slate
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    for harness in make_harness(tmp_path, sample_pdf, database_url=url):
        c = harness.client
        register(c, f"pg-{uuid.uuid4().hex[:6]}@example.com")
        doc_id = upload(c, sample_pdf).json()["id"]
        assert c.get(f"/documents/{doc_id}/status").json()["status"] == "ready"
        assert c.post("/auth/guest", headers=CSRF).status_code == 201


@pytest.mark.skipif(not os.environ.get("TEST_CHROMA_URL"), reason="set TEST_CHROMA_URL to test a Chroma server")
def test_chroma_server_round_trip(tmp_path):
    store = VectorStore(tmp_path, os.environ["TEST_CHROMA_URL"])
    user = f"u{uuid.uuid4().hex[:8]}"
    store.add_chunks(user, [_chunk(f"{user}:0", "d", "hello world")], [[1.0] + [0.0] * 63])
    hits = store.query(user, [1.0] + [0.0] * 63, 5)
    assert hits and hits[0][0].chunk_id == f"{user}:0"
    assert store.query("someone-else", [1.0] + [0.0] * 63, 5) == []
    store.delete_user(user)
    assert store.user_chunks(user) == []
