# MCP_SETUP.md — keep Claude Code on current docs (anti-hallucination setup)

Problem: models write code from training data, which can be old (renamed functions, retired model names, changed Tailwind/Chroma/FastAPI APIs). Fix: give Claude Code live documentation tools and a rule to use them **before** writing code against any library.

MCP reduces this class of error a lot but does not remove it. The second half of this file is the verification loop that catches what is left.

## 1. Install the doc servers (run once, in your terminal)

### Context7 (up-to-date docs for thousands of libraries: FastAPI, ChromaDB, SQLAlchemy, Pydantic, Tailwind, PyMuPDF, sentence-transformers, …)
```bash
claude mcp add --scope user --transport http context7 https://mcp.context7.com/mcp
```
Optional, for higher rate limits (free key from context7.com):
```bash
claude mcp add --scope user --transport http context7 https://mcp.context7.com/mcp --header "CONTEXT7_API_KEY: YOUR_API_KEY"
```
Local alternative (needs Node 18+):
```bash
claude mcp add --scope user context7 -- npx -y @upstash/context7-mcp
```

### Claude Platform docs (Anthropic API, SDKs, current model names, streaming, errors)
```bash
claude mcp add --scope user --transport http claude-platform-docs https://platform.claude.com/docs/mcp
```
This server covers the Claude API and SDKs, not Claude Code itself.

### Claude Code docs (optional, for MCP/CLAUDE.md questions)
```bash
claude mcp add --scope user --transport http claude-code-docs https://code.claude.com/docs/mcp
```

### Verify
Start `claude`, run `/mcp`. All servers should show as connected. If one asks you to authenticate, select it in `/mcp` and complete the browser sign-in.

To share with the repo instead of your user profile, re-run with `--scope project`; that writes `.mcp.json` in the project root.

## 2. Project rule file: `CLAUDE.md` (put in the repo root)

```markdown
# rag-chat — standing rules for Claude Code

## Docs-first rule (anti-hallucination)
- Before writing or changing code that calls ANY third-party library, SDK or CSS framework, fetch its CURRENT docs:
  - libraries / frameworks → Context7 (resolve the library id, then query docs for the exact feature)
  - Anthropic API / SDK / model names → claude-platform-docs
- Never rely on memory for: function signatures, config options, model IDs, CLI flags, Tailwind config syntax, Chroma client API, SQLAlchemy 2.x style, Pydantic v2 style.
- If a docs tool is unavailable, say so, then verify locally instead: `pip show <pkg>`, `python -c "import x; help(x.y)"`, read installed source, or run a 5-line smoke test.
- Record the version you coded against in `requirements.lock`.
- If docs and your memory disagree, the docs win. If you cannot confirm something, say "unverified" in the code comment and the final report; do not guess silently.

## Verification loop
- After writing any integration code, RUN it (import, call, or test). Code that was never executed is not done.
- Fix deprecation warnings; they usually mean you used an old API.

## Project conventions
- Architecture: see ARCHITECTURE.md. UI: see DESIGN_SYSTEM.md. They are binding.
- No gradients in the UI. No raw hex outside the token block.
- Layering: api → services → (ingest | retrieval | generation | db). Never import upward.
- Secrets only from env. Never log document text.
- Small typed functions, docstrings on public ones, ruff clean, no TODO stubs.
```

## 3. How to trigger the docs lookup inside prompts
Add the phrase **"use context7"** (or name the server) to any prompt that touches a library, e.g.:
- "Implement the Chroma wrapper. use context7 for chromadb."
- "Add streaming with the Anthropic SDK. Check the current docs with claude-platform-docs, including the right model ID."

The prompt in `CLAUDE_CODE_PROMPT.md` already contains this instruction.

## 4. Library checklist (verify these through docs before coding)

| Area | Verify |
|---|---|
| Anthropic SDK | client init, streaming helper, current model ID string, error classes, max_tokens requirement |
| FastAPI | SSE with `StreamingResponse` / current SSE support, lifespan handlers (not deprecated `on_event`), cookie API, `Depends` |
| Pydantic v2 | `model_validate`, `ConfigDict`, `pydantic-settings` v2 import path |
| SQLAlchemy 2.0 | `DeclarativeBase`, `Mapped[]`, `select()` style, session handling |
| ChromaDB | `PersistentClient`, collection creation with cosine, `where` filters, `$in` operator, upsert semantics |
| sentence-transformers | `SentenceTransformer`, `encode(normalize_embeddings=True)`, `CrossEncoder.predict`, bge query prefix |
| PyMuPDF | import name (`pymupdf` vs `fitz`), `get_text()` options |
| rank-bm25 | `BM25Okapi` usage |
| argon2-cffi | `PasswordHasher` verify/rehash API |
| PyJWT | `encode`/`decode` signatures, algorithms argument |
| Tailwind (CDN) | current CDN script and how to configure theme tokens for the installed major version |
| Docker | current Python slim tag, pip cache and model-download pattern |

## 5. Verification loop that catches what MCP misses
1. **Pin and lock:** `pip install -r requirements.txt && pip freeze > requirements.lock`; the Dockerfile installs from the lock file.
2. **Run, don't assume:** after each module, run an import + smoke test; after the backend, run the full `pytest`.
3. **Warnings as errors in tests:** `pytest -W error::DeprecationWarning` (relax only for third-party noise you list explicitly).
4. **Contract tests with `FakeLLM`** for the chat flow, plus **one real live call** to the Anthropic API during final verification if a key is present.
5. **Evaluation script** must print real numbers from real runs; the README table is generated from its output.
6. **Final audit step:** Claude Code lists every external API it used, the doc source or command that confirmed it, and anything marked "unverified".

## 6. Honest limits
- Docs servers can lag a brand-new release or lack a niche library; the fallback in section 2 covers that.
- They fix API drift, not logic mistakes. Tests and the evaluation set do that.
- Third-party MCP servers are code you run; stick to the vendors above and review what you add.
