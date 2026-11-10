# Claude Code prompt (v2, structured)

## Setup before you paste (5 minutes)
1. Create a folder `rag-chat/` and put these files in it: `RAG_PROJECT_PLAN.md`, `ARCHITECTURE.md`, `DESIGN_SYSTEM.md`, `MCP_SETUP.md`.
2. Install the docs MCP servers using the commands in `MCP_SETUP.md` section 1, then run `/mcp` inside Claude Code and confirm they are connected.
3. `export ANTHROPIC_API_KEY=...` in your shell.
4. Open Claude Code inside `rag-chat/` and paste everything between the lines below.

---

# ROLE
You are a senior full-stack ML engineer building a portfolio-grade project end to end, working autonomously. You verify everything you build by running it, and you never guess at library APIs.

# CONTEXT
- **Project:** `rag-chat`, a "Chat with your Docs" web app. Users sign in (or continue as a guest), upload PDFs, and ask questions. Answers stream in with inline citations `[1][2]` that open the exact source passage (file, page, snippet). If the documents don't contain the answer, the app says so instead of guessing.
- **Audience:** recruiters and hiring managers who will open the live demo and the GitHub repo for a few minutes. It must be easy to use, visibly well engineered, and run with one command.
- **Owner's skills:** Python, JS, Node, Tailwind, Docker, SQL, Mongo. Backend language is **Python 3.11 + FastAPI**; frontend is **static HTML + vanilla JS + Tailwind CDN**, no build step.
- **Source of truth documents in this folder (read all four first, they are binding):**
  1. `ARCHITECTURE.md`: framework, layering, database schema, auth, API, integrations, security, testing. Where it differs from `RAG_PROJECT_PLAN.md`, ARCHITECTURE.md wins.
  2. `RAG_PROJECT_PLAN.md`: RAG pipeline details, phases, data structures, evaluation method.
  3. `DESIGN_SYSTEM.md`: color/spacing/typography tokens, component conventions, UX checklist. **No gradients.**
  4. `MCP_SETUP.md`: the docs-first rule and verification loop.
- **Docs tools available:** Context7 MCP (library docs) and claude-platform-docs MCP (Anthropic API/SDK/model names). Check `/mcp` first.
- **Time budget:** the owner wants this complete in about two days of work, so favor the simplest correct design and do not add scope.

# TASK
Build the entire project in one run, in this order. Finish and verify each step before starting the next.

0. **Preflight.** Create `CLAUDE.md` in the repo root using the exact text from `MCP_SETUP.md` section 2. Confirm which docs MCP tools are connected. If none are, tell me and continue with the local-verification fallback described there. `git init`, add `.gitignore`.
1. **Scaffold** per ARCHITECTURE.md section 4: layered folders, `config.py` (pydantic-settings), `.env.example`, ruff config, `requirements.txt`.
2. **Database + auth** per ARCHITECTURE.md sections 5 and 6: SQLAlchemy models, repositories, register/login/guest/logout/me, httpOnly-cookie JWT, argon2id, rate limits, guest cleanup and a bundled sample PDF.
3. **Ingestion** per the plan: PyMuPDF loader, cleaner, sentence-aware chunker (800/150) with page ranges, SHA-256 dedupe per user, background processing with status stages.
4. **Retrieval:** bge-small embeddings (singleton, query prefix), Chroma with `user_id`-filtered queries, per-user BM25, RRF fusion, cross-encoder rerank, top 5, all intermediate scores kept.
5. **Generation:** `LLMClient` protocol, `AnthropicLLM` (streaming), `FakeLLM`; grounded prompt that treats sources as untrusted data; `NO_ANSWER` handling; score-threshold fallback **before** calling the LLM; citation parsing; SSE endpoint; conversations and messages persisted.
6. **Frontend** per DESIGN_SYSTEM.md: tokens, every listed component, light/dark, the full UX checklist in section 10, and a `/static/styleguide.html` page demonstrating all components.
7. **Tests + evaluation:** unit and integration tests, user-isolation test, `scripts/check_contrast.py`, `tests/eval_set.json` (≥15 answerable + 5 unanswerable, with expected pages) and `scripts/evaluate.py` comparing vector-only vs hybrid vs hybrid+rerank (Hit@5, MRR, fallback accuracy). Use the threshold from the evaluation to set `MIN_RERANK_SCORE`.
8. **Docker + CI:** Dockerfile (models pre-downloaded at build time, installs from `requirements.lock`), docker-compose with a `./data` volume and healthcheck, `.dockerignore`, GitHub Actions workflow running ruff + pytest.
9. **README** per ARCHITECTURE.md section 14 and the plan's README checklist: pitch, architecture diagram, features, **real measured** evaluation table, design decisions, run instructions (Docker and local), limitations, demo-GIF placeholder.
10. **Final audit** (see Expected Output).

# CONSTRAINTS

**Correctness and anti-hallucination**
- Before writing code against any library, SDK or CSS framework, fetch its current docs with Context7 (say "use context7") or claude-platform-docs. Do not rely on memory for signatures, config options, CLI flags, Tailwind config syntax, Chroma/SQLAlchemy/Pydantic APIs, or **the Anthropic model ID**. Read the current model name from the Claude Platform docs and put it in `.env.example`; do not invent one.
- If docs and memory disagree, docs win. If something cannot be confirmed, mark it `unverified` in a code comment and in the final report; never guess silently.
- Run everything you write. Code never executed is not done. Treat deprecation warnings as bugs.
- Never invent numbers: all README metrics come from a real run of `scripts/evaluate.py`. If something could not be run (no API key, no Docker), say so explicitly.

**Architecture and quality**
- Obey the layering rule (api → services → ingest/retrieval/generation/db). No upward imports, no FastAPI imports in services.
- Every retrieval and document access is scoped to `current_user`; return 404 for other users' resources.
- Typed, small functions; docstrings on public functions; ruff clean; no dead code, no TODO stubs, no placeholder implementations.
- Secrets only from env; never log document text, passwords or keys; validate uploads by magic bytes and size.
- Pin dependencies to installed versions in `requirements.lock`.

**UI**
- **No gradients of any kind** (`grep -ri gradient app/static` must return nothing).
- All colors, spacing, radii, shadows and type sizes come from the tokens in DESIGN_SYSTEM.md; no raw hex outside the token block; spacing only from the 4 px scale.
- Component naming and state conventions exactly as in DESIGN_SYSTEM.md section 8; WCAG AA contrast verified by the script; keyboard accessible with visible focus; respects `prefers-reduced-motion`; usable at 360 px width.
- User-friendly copy: sentence case, plain language, actionable error messages, never a raw stack trace or status code in the UI.

**Process**
- Start with a task list; build in the order above; small, meaningful git commits per step.
- Do not stop to ask questions unless truly blocked. For ambiguity, choose the simplest reasonable option and record it under "Design decisions" in the README.
- Do not add features beyond the documents (no OCR, no OAuth, no extra file types) unless everything else is finished and verified.

# EXPECTED OUTPUT

**Repository**, runnable with `cp .env.example .env && docker compose up`, serving the app at `http://localhost:8000`:
```
rag-chat/
├── app/ (api, services, ingest, retrieval, generation, db, static)
├── tests/            # unit + integration + eval_set.json
├── scripts/          # evaluate.py, check_contrast.py
├── .github/workflows/ci.yml
├── CLAUDE.md  ARCHITECTURE.md  DESIGN_SYSTEM.md  MCP_SETUP.md  RAG_PROJECT_PLAN.md  README.md
├── Dockerfile  docker-compose.yml  requirements.txt  requirements.lock  .env.example  .gitignore  .dockerignore
```

**Behavior I will check:**
- Register/login/guest works; guest lands on a ready sample document with sample questions.
- Upload shows real stages and ends in "Ready"; duplicates are rejected politely.
- An answerable question streams a grounded answer with working `[n]` chips opening a source drawer with file, page and highlighted snippet.
- An out-of-document question returns the friendly "couldn't find this" card without calling the LLM.
- User B cannot see or retrieve user A's documents (tested).
- Light/dark both look correct; no gradients; styleguide page shows every component.

**Commands that must pass:** `ruff check .`, `pytest -W error::DeprecationWarning` (with only listed, justified exceptions), `python scripts/check_contrast.py`, `python scripts/evaluate.py`.

**Final report (your last message, under 400 words):**
1. What was built and how to run it.
2. The evaluation table exactly as printed by the script.
3. **Docs audit:** a table of every external API/library used → the doc source or command that confirmed it → status (`verified` / `unverified`).
4. Anything you could not run or verify (e.g. no Docker, no API key) and exactly what I should run myself.
5. Three highest-value improvements if I had another day.

---

## Follow-up prompts (use after it finishes)
1. "Review the repo as a skeptical hiring manager. List the top 5 weaknesses and fix them. Re-run all checks."
2. "Deploy to Hugging Face Spaces (Docker SDK). Verify current Spaces requirements from the docs first, then give me exact steps."
3. "Record a 20-second demo flow script (steps to click) for a README GIF."
4. "Add conversation-aware query rewriting for follow-up questions, with tests."
