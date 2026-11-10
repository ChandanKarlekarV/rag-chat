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
