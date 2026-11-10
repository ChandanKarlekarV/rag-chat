# ARCHITECTURE.md — rag-chat

This file extends and, where they differ, **supersedes** `RAG_PROJECT_PLAN.md` (it adds users, auth, conversations and per-user isolation).

## 1. Goals and non-goals

**Goals:** cited, streamed answers from the user's own PDFs; no guessing when the docs lack the answer; per-user data isolation; one-command run; easy to read for a reviewer.

**Non-goals (v1):** OCR for scanned PDFs, multi-tenant orgs, horizontal scaling, file types other than PDF.

## 2. System context

```
 Browser (static HTML + JS + Tailwind)
      │  HTTPS, JSON + SSE, httpOnly cookie
      ▼
 FastAPI app  ───────────────►  LLM provider (Anthropic API)      [external]
      │
      ├── SQLite (SQLAlchemy)    users, documents, conversations, messages
      ├── ChromaDB (persistent)  chunk vectors + metadata
      ├── BM25 index (pickle)    per-user keyword index
      └── Local ML models        bge-small embedder, MiniLM cross-encoder
```

Everything except the LLM call runs inside one container. Documents and embeddings never leave the machine; only the final prompt (question + top 5 chunks) goes to the LLM.

## 3. Framework and tooling decisions

| Concern | Choice | Reason |
|---|---|---|
| Language | Python 3.11 | ML ecosystem |
| Web framework | FastAPI + Uvicorn | async, SSE, typed, auto OpenAPI |
| Validation | Pydantic v2 + pydantic-settings | one source of truth for schemas and config |
| ORM | SQLAlchemy 2.0 (sync, typed `Mapped[]` style) | standard, swap SQLite → Postgres by changing one URL |
| Migrations | `create_all` at startup in v1; Alembic is listed as future work | keeps the 2-day scope |
| Frontend | Static HTML + vanilla JS modules + Tailwind CDN, driven by the tokens in `DESIGN_SYSTEM.md` | no build step |
| Lint / format | ruff (lint + format), mypy optional | fast, one tool |
| Tests | pytest + httpx `TestClient` | |
| CI | GitHub Actions: ruff, pytest | shows engineering maturity |
| Container | Docker (python:3.11-slim) + compose | |

**Rule:** pin every dependency to the version actually installed (`pip freeze` → `requirements.lock`), and before using any library API, check the current docs through the MCP servers listed in `MCP_SETUP.md`.

## 4. Layered code structure

```
app/
  main.py            # app factory, router wiring, static files, lifespan (load models once)
  config.py          # Settings
  api/               # HTTP layer only: parse request, call service, shape response
    auth.py  documents.py  chat.py  health.py  deps.py
  services/          # business logic, no FastAPI imports
    auth_service.py  ingest_service.py  retrieval_service.py  chat_service.py
  ingest/            # loader.py cleaner.py chunker.py
  retrieval/         # embedder.py vector_store.py bm25_store.py hybrid.py reranker.py
  generation/        # prompt.py llm_base.py llm_anthropic.py llm_fake.py
  db/                # models.py session.py repositories.py
  schemas.py         # Pydantic API models
  static/            # index.html, app.js, styles (tokens)
```

Dependency direction: `api → services → (ingest | retrieval | generation | db)`. Lower layers never import upper layers. This is what lets tests swap `FakeLLM` in.

## 5. Database

### 5.1 Tables (SQLite via SQLAlchemy)

```
users
  id            TEXT PK (uuid4 hex)
  email         TEXT UNIQUE NOT NULL
  password_hash TEXT NOT NULL          -- argon2id
  is_guest      BOOLEAN DEFAULT 0
  created_at    DATETIME

documents
  id          TEXT PK
  user_id     TEXT FK users.id ON DELETE CASCADE, INDEX
  filename    TEXT
  sha256      TEXT
  pages       INTEGER
  chunks      INTEGER
  status      TEXT   -- 'processing' | 'ready' | 'failed'
  error       TEXT NULL
  created_at  DATETIME
  UNIQUE(user_id, sha256)              -- dedupe per user

conversations
  id          TEXT PK
  user_id     TEXT FK, INDEX
  title       TEXT                      -- first question, truncated
  created_at  DATETIME

messages
  id               TEXT PK
  conversation_id  TEXT FK ON DELETE CASCADE, INDEX
  role             TEXT   -- 'user' | 'assistant'
  content          TEXT
  answerable       BOOLEAN NULL
  citations_json   TEXT NULL            -- list of Citation
  created_at       DATETIME
```

### 5.2 Vector store
Chroma collection `chunks`, cosine distance. Metadata on every chunk: `user_id, doc_id, filename, page_start, page_end, index`.

### 5.3 Isolation rule (critical)
Every retrieval call **must** filter by `user_id` (`where={"user_id": current_user.id}`, plus optional `doc_id $in`). BM25 indexes are built and cached **per user**. Add a test that user B can never retrieve user A's chunks.

## 6. Authentication and authorization

- **Method:** email + password. Passwords hashed with argon2id (`argon2-cffi`). Never log or return hashes.
- **Session:** signed JWT (PyJWT, HS256, secret from env) stored in an **httpOnly, SameSite=Lax, Secure-in-prod cookie**, 7-day expiry. No tokens in localStorage.
- **CSRF:** SameSite=Lax plus a required `X-Requested-With: fetch` header on all state-changing requests.
- **Guest mode:** "Try it as a guest" creates a throwaway user (`is_guest=true`) and preloads a small sample PDF so a recruiter can try it in 10 seconds with no signup. Guest data is deleted after 24 h by a startup cleanup task.
- **Authorization:** every route that touches documents, conversations or messages resolves `current_user` via a `Depends(get_current_user)` dependency and checks ownership. Return 404 (not 403) for other users' resources.
- **Abuse limits:** simple in-memory rate limit on `/auth/*` (5/min/IP) and `/ask` (20/min/user); max upload 20 MB; max 10 documents per guest, 50 per user.
- **Password rules:** min 8 chars; show a strength hint; no other composition rules.

### Auth endpoints
`POST /auth/register`, `POST /auth/login`, `POST /auth/guest`, `POST /auth/logout`, `GET /auth/me`.

## 7. API surface

| Method | Path | Purpose |
|---|---|---|
| POST | `/documents` | upload PDF (multipart); returns doc with `status` |
| GET | `/documents` | list current user's documents |
| DELETE | `/documents/{id}` | delete doc, its chunks and BM25 entries |
| GET | `/documents/{id}/status` | poll ingest stage |
| POST | `/chat/ask` | SSE stream: `token`, `citations`, `done`, `error` events |
| GET | `/conversations` | list |
| GET | `/conversations/{id}` | messages |
| DELETE | `/conversations/{id}` | delete |
| GET | `/health` | liveness |

Error shape everywhere: `{"error": {"code": "string", "message": "human readable"}}`. Messages must be friendly and actionable (e.g. "This PDF has no selectable text. Try a text-based PDF.").

## 8. Core flows

**Ingest:** upload → validate (type, size, sha256 dedupe) → row `processing` → extract pages → clean → chunk → embed → write Chroma → rebuild user BM25 → row `ready`. Run in a FastAPI `BackgroundTasks` job; frontend polls `/status` and shows stages: Reading → Splitting → Understanding → Ready.

**Ask:** question → (if conversation has history, rewrite to a standalone query, optional) → embed → vector top 20 + BM25 top 20 (both user-filtered) → RRF → cross-encoder rerank → top 5 → **gate** (best rerank score < `MIN_RERANK_SCORE` → fallback, no LLM call) → prompt with numbered sources → stream LLM → parse `[n]` → build citations → persist messages → `done`.

## 9. Integrations

| Integration | Used for | Where |
|---|---|---|
| Anthropic API | answer generation, streaming | `generation/llm_anthropic.py` behind `LLMClient` protocol |
| Local models (Hugging Face) | embeddings + reranking | `retrieval/` |
| Context7 MCP + Claude Platform docs MCP | **dev-time only**: keeps generated code on current APIs | `MCP_SETUP.md` |
| GitHub Actions | CI | `.github/workflows/ci.yml` |
| Hugging Face Spaces or Render | optional deploy | README |
| Stretch: Google OAuth login | skip unless time remains | — |

`LLMClient` protocol: `stream(system: str, user: str) -> Iterator[str]`. Implementations: `AnthropicLLM`, `FakeLLM` (tests/offline). Provider and model come from env (`LLM_PROVIDER`, `LLM_MODEL`); **verify the current model string against the Claude Platform docs before hard-coding a default.**

## 10. Configuration (`.env.example`)

```
APP_ENV=dev
SECRET_KEY=change-me
DATABASE_URL=sqlite:///./data/app.db
DATA_DIR=./data
ANTHROPIC_API_KEY=
LLM_PROVIDER=anthropic
LLM_MODEL=            # fill from current Claude docs
EMBED_MODEL=BAAI/bge-small-en-v1.5
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
CHUNK_SIZE=800
CHUNK_OVERLAP=150
RETRIEVE_K=20
FINAL_K=5
MIN_RERANK_SCORE=0.0   # tune with scripts/evaluate.py
MAX_UPLOAD_MB=20
```

## 11. Observability
Structured logs (JSON) with request id; log retrieval stats per question (latencies for embed, search, rerank, LLM; chunk ids; scores). Never log document text or API keys. `/health` checks DB and model-loaded flags.

## 12. Security checklist
- [ ] Secrets only from env; `.env` git-ignored
- [ ] Uploads: verify `%PDF` magic bytes, not just extension; store under random names
- [ ] Prompt-injection stance: document text is untrusted; the system prompt tells the LLM to treat sources as data, never as instructions
- [ ] Output escaping in the UI (render answers as text, linkify `[n]` only)
- [ ] Per-user filters tested
- [ ] Dependencies pinned; `pip-audit` run once

## 13. Testing strategy
Unit: chunker, cleaner, RRF, fallback gate, citation parser. Integration (TestClient + `FakeLLM`): register → upload → ask → citations; isolation test; unanswerable question. Evaluation: `scripts/evaluate.py` (Hit@5, MRR, fallback accuracy for vector / hybrid / hybrid+rerank).

## 14. Decisions log (ADR style, copy into README)
1. Hybrid + rerank over vector-only — measured improvement.
2. Local embeddings — privacy, zero cost.
3. SQLite + Chroma — zero ops; Postgres is a one-line swap.
4. httpOnly cookie over localStorage JWT — XSS-safer.
5. Threshold fallback before the LLM — cheaper and more reliable than prompt-only refusal.
6. No JS framework — no build step, reviewer can read it in minutes.
