"""Integration tests: auth, documents, ask flow, conversations, isolation, CSRF, error shape."""

from datetime import UTC, datetime, timedelta

from app.db.models import User
from app.db.repositories import UserRepo
from tests.conftest import CSRF, ask, make_harness, register, upload

QUESTION = "How much usable energy capacity does the battery have?"


# ---------- auth ----------


def test_register_login_me_logout(h):
    c = h.client
    user = register(c)
    assert user["email"] == "ada@example.com" and user["is_guest"] is False
    cookie = c.cookies.get("rag_session")
    assert cookie and "." in cookie
    assert c.get("/auth/me").json()["id"] == user["id"]
    assert c.post("/auth/logout", headers=CSRF).status_code == 204
    assert c.get("/auth/me").status_code == 401
    r = c.post("/auth/login", json={"email": "ADA@example.com ", "password": "correct-horse"}, headers=CSRF)
    assert r.status_code == 200 and r.json()["id"] == user["id"]


def test_session_cookie_is_httponly_and_lax(h):
    r = h.client.post("/auth/register", json={"email": "c@example.com", "password": "12345678"}, headers=CSRF)
    set_cookie = r.headers["set-cookie"].lower()
    assert "httponly" in set_cookie and "samesite=lax" in set_cookie


def test_auth_errors_are_friendly(h):
    c = h.client
    r = c.post("/auth/register", json={"email": "bad", "password": "longenough"}, headers=CSRF)
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_email"
    r = c.post("/auth/register", json={"email": "a@b.co", "password": "short"}, headers=CSRF)
    assert r.json()["error"]["message"].startswith("Use at least 8")
    register(c, "dup@example.com")
    r = c.post("/auth/register", json={"email": "dup@example.com", "password": "12345678"}, headers=CSRF)
    assert r.status_code == 409
    r = c.post("/auth/login", json={"email": "dup@example.com", "password": "wrong-pass"}, headers=CSRF)
    assert r.status_code == 401 and r.json() == {
        "error": {"code": "bad_credentials", "message": "That email and password don't match. Please try again."}
    }


def test_password_is_stored_as_argon2id(h):
    register(h.client, "hash@example.com", "my-secret-pw")
    with h.client.app.state.container.db.session() as s:
        stored = UserRepo(s).by_email("hash@example.com").password_hash
    assert stored.startswith("$argon2id$") and "my-secret-pw" not in stored


def test_state_changing_requests_require_fetch_header(h):
    r = h.client.post("/auth/register", json={"email": "x@example.com", "password": "12345678"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "csrf"


def test_auth_rate_limit(tmp_path, sample_pdf):
    for harness in make_harness(tmp_path, sample_pdf, auth_rate_per_min=2):
        c = harness.client
        for _ in range(2):
            c.post("/auth/login", json={"email": "n@example.com", "password": "whatever1"}, headers=CSRF)
        r = c.post("/auth/login", json={"email": "n@example.com", "password": "whatever1"}, headers=CSRF)
        assert r.status_code == 429 and "wait a minute" in r.json()["error"]["message"]


def test_guest_lands_on_ready_sample_document(h):
    r = h.client.post("/auth/guest", headers=CSRF)
    assert r.status_code == 201 and r.json()["is_guest"] is True
    docs = h.client.get("/documents").json()
    assert len(docs) == 1 and docs[0]["status"] == "ready" and docs[0]["chunks"] > 0


def test_expired_guests_are_cleaned_up(h):
    h.client.post("/auth/guest", headers=CSRF)
    container = h.client.app.state.container
    with container.db.session() as s:
        guest = s.query(User).filter(User.is_guest.is_(True)).one()
        guest.created_at = datetime.now(UTC) - timedelta(hours=25)
        guest_id = guest.id
    assert container.auth.cleanup_guests() == 1
    assert h.client.get("/auth/me").status_code == 401
    assert container.ingest.vectors.user_chunks(guest_id) == []


def test_unauthenticated_requests_get_401_error_shape(h):
    r = h.client.get("/documents")
    assert r.status_code == 401 and r.json()["error"]["code"] == "not_signed_in"


# ---------- documents ----------


def test_upload_processes_to_ready_and_dedupes(h, sample_pdf):
    c = h.client
    register(c)
    r = upload(c, sample_pdf)
    assert r.status_code == 202 and r.json()["status"] == "processing"
    doc_id = r.json()["id"]
    status = c.get(f"/documents/{doc_id}/status").json()  # background task ran after the response
    assert status["status"] == "ready" and status["stage"] == "ready" and status["progress"] == 100
    dup = upload(c, sample_pdf, "copy.pdf")
    assert dup.status_code == 409 and "already uploaded" in dup.json()["error"]["message"]
    assert c.delete(f"/documents/{doc_id}", headers=CSRF).status_code == 204
    assert c.get("/documents").json() == []
    assert c.get(f"/documents/{doc_id}/status").status_code == 404


def test_upload_rejects_non_pdf_and_textless_pdf(h, tmp_path):
    import pymupdf

    c = h.client
    register(c)
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"hello, I am not a PDF")
    r = upload(c, fake)
    assert r.status_code == 400 and "not a PDF" in r.json()["error"]["message"]

    blank = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(blank)
    r = upload(c, blank)
    assert r.status_code == 202
    doc_id = r.json()["id"]
    status = c.get(f"/documents/{doc_id}/status").json()
    assert status["status"] == "failed" and "no selectable text" in status["error"]
    retried = c.post(f"/documents/{doc_id}/retry", headers=CSRF)
    assert retried.status_code == 202  # runs again (and fails again: still no text)
    assert c.get(f"/documents/{doc_id}/status").json()["status"] == "failed"


def test_retry_only_for_failed_documents(h, sample_pdf):
    register(h.client)
    doc_id = upload(h.client, sample_pdf).json()["id"]
    r = h.client.post(f"/documents/{doc_id}/retry", headers=CSRF)
    assert r.status_code == 400 and r.json()["error"]["code"] == "not_failed"


def test_document_quota(tmp_path, sample_pdf):
    for harness in make_harness(tmp_path, sample_pdf, max_docs_user=1):
        c = harness.client
        register(c)
        assert upload(c, sample_pdf).status_code == 202
        other = tmp_path / "other.pdf"
        import pymupdf

        d = pymupdf.open()
        d.new_page().insert_text((72, 72), "Different content entirely.")
        d.save(other)
        r = upload(c, other)
        assert r.status_code == 429 and "up to 1 documents" in r.json()["error"]["message"]


# ---------- ask ----------


def test_ask_streams_answer_with_citations_and_persists(h, sample_pdf):
    c = h.client
    register(c)
    upload(c, sample_pdf)
    events = ask(c, QUESTION)
    names = [e for e, _ in events]
    assert names[0] == "mode" and names[1] == "token" and names[-2:] == ["citations", "done"]
    assert events[0][1]["mode"] == "api"  # a key was sent, so auto-selection picked API mode
    streamed = "".join(d["text"] for e, d in events if e == "token")
    citations = events[-2][1]["citations"]
    done = events[-1][1]
    assert done["answerable"] is True and streamed == done["answer"]
    assert [x["n"] for x in citations] == [1] and citations[0]["filename"] == "halcyon.pdf"
    assert citations[0]["page_start"] >= 1 and citations[0]["text"]

    convs = c.get("/conversations").json()
    assert len(convs) == 1 and convs[0]["title"].startswith("How much usable")
    detail = c.get(f"/conversations/{done['conversation_id']}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert detail["messages"][1]["citations"][0]["n"] == 1

    follow = ask(c, "And how heavy is the battery enclosure?", conversation_id=done["conversation_id"])
    assert follow[-1][1]["conversation_id"] == done["conversation_id"]
    assert len(c.get(f"/conversations/{done['conversation_id']}").json()["messages"]) == 4


def test_unanswerable_question_falls_back_without_calling_llm(h, sample_pdf):
    c = h.client
    register(c)
    upload(c, sample_pdf)
    events = ask(c, "chocolate cookie recipe")
    assert [e for e, _ in events] == ["mode", "citations", "done"]
    assert events[-1][1]["answerable"] is False
    assert h.made == [] or all(llm.calls == [] for llm in h.made)


def test_api_mode_without_a_key_explains_what_to_do(h, sample_pdf):
    c = h.client
    register(c)
    upload(c, sample_pdf)
    name, data = ask(c, QUESTION, key=None, mode="api")[-1]
    assert name == "error" and data["code"] == "no_api_key" and "Settings" in data["message"]


def test_ask_with_no_ready_documents(h):
    register(h.client)
    r = h.client.post("/chat/ask", json={"question": "anything?"}, headers=CSRF)
    assert r.status_code == 400 and r.json()["error"]["message"].startswith("Upload a PDF to start")


def test_conversation_rename_and_delete(h, sample_pdf):
    c = h.client
    register(c)
    upload(c, sample_pdf)
    conv_id = ask(c, QUESTION)[-1][1]["conversation_id"]
    r = c.patch(f"/conversations/{conv_id}", json={"title": "Battery capacity"}, headers=CSRF)
    assert r.json()["title"] == "Battery capacity"
    assert c.delete(f"/conversations/{conv_id}", headers=CSRF).status_code == 204
    assert c.get(f"/conversations/{conv_id}").status_code == 404


# ---------- isolation ----------


def test_user_b_cannot_see_or_retrieve_user_a_documents(h, sample_pdf):
    alice = h.client
    register(alice, "alice@example.com")
    doc_id = upload(alice, sample_pdf).json()["id"]
    conv_id = ask(alice, QUESTION)[-1][1]["conversation_id"]

    bob = h.new_client()
    register(bob, "bob@example.com")
    assert bob.get("/documents").json() == []
    assert bob.get(f"/documents/{doc_id}/status").status_code == 404
    assert bob.delete(f"/documents/{doc_id}", headers=CSRF).status_code == 404
    assert bob.get(f"/conversations/{conv_id}").status_code == 404
    r = bob.post("/chat/ask", json={"question": QUESTION, "doc_ids": [doc_id]}, headers=CSRF)
    assert r.status_code == 400  # Bob has no ready documents; Alice's id is not accepted

    # Even at the retrieval layer, Bob's searches can never return Alice's chunks.
    retrieval = h.client.app.state.container.retrieval
    bob_id = bob.get("/auth/me").json()["id"]
    assert retrieval.search(bob_id, QUESTION).chunks == []
    assert retrieval.search(bob_id, QUESTION, doc_ids=[doc_id]).chunks == []
    alice_id = alice.get("/auth/me").json()["id"]
    assert retrieval.search(alice_id, QUESTION).chunks


# ---------- health ----------


def test_health_and_index(h):
    body = h.client.get("/health").json()
    assert body["status"] == "ok" and body["database"] is True and body["models_loaded"] is True
    assert "rag-chat" in h.client.get("/").text
    assert h.client.get("/health").headers["X-Request-ID"]
