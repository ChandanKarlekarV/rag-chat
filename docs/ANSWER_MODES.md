# ANSWER_MODES.md — answer with or without an API key

Adds a second way to answer questions so the app works **with no API key and no internet LLM**, and still lets users switch to an API for fluent, synthesized answers. Binding addendum to `ARCHITECTURE.md` (section 9 and the `/chat/ask` flow) and `DESIGN_SYSTEM.md` (settings UI).

## 1. The three modes

| Mode | Needs | How it answers | Best for |
|---|---|---|---|
| **Local (no key)** — default when no key exists | nothing; models run in the container | **Extractive**: finds the exact answer span and supporting sentences inside your PDF and quotes them with page numbers. Cannot invent text. | factual questions (who, when, how much, what is X), strict accuracy, offline, free, privacy |
| **Local LLM (Ollama)** — optional | Ollama running locally | Generative answer from the retrieved passages using a local model | fluent answers without paying or sending data out |
| **API (Claude)** | API key (server env var **or** pasted by the user in the UI) | Generative, synthesizes across passages | "summarize", "compare", "why" questions needing reasoning |

Auto-selection on first load: API key available → API; else Ollama reachable → Ollama; else Local. The user can change it any time from a segmented control above the chat input. The chosen mode is saved per user.

**All modes share the same retrieval pipeline and the same no-answer gate** (hybrid search → rerank → score threshold). Only the last step ("turn passages into an answer") differs. That is what keeps accuracy comparable and hallucination low in every mode.

## 2. Local (extractive) mode — how it stays accurate

Goal: highest possible accuracy with zero generation. Pipeline after the top 5 reranked chunks:

1. **Extractive QA model** over each chunk: `deepset/roberta-base-squad2` (SQuAD 2.0, trained to say "no answer"). Optional smaller/faster: `distilbert-base-cased-distilled-squad`. Use the Hugging Face `question-answering` pipeline with `handle_impossible_answer=True`, `max_answer_len` capped (~40 tokens), and chunks fed with their natural overlap. Verify the current pipeline arguments in the docs (Context7) before coding.
2. **Pick the best span:** score = `0.6 * qa_score + 0.4 * normalized_rerank_score` of its chunk. If the best span's `qa_score < MIN_QA_SCORE` or the model returns an empty/null answer → **no-answer fallback** (same friendly card as other modes).
3. **Supporting sentences:** split the winning chunk (and the next two best chunks) into sentences, score each with the existing cross-encoder against the question, keep the top 3 in document order. This covers "how/why/explain" questions where the answer is a sentence or two rather than a short span.
4. **Compose the response from templates, not generation:**
   ```
   **Answer:** <best span, original wording>   [1]
   **From the document:**
   > <supporting sentence 1>  [1]
   > <supporting sentence 2>  [2]
   ```
   Highlight the span inside its sentence in the source drawer.
5. **Question-type routing (cheap rules):**
   - "summarize/overview/what is this about" → return the top-ranked section openers (first 2 sentences of the 3 best-spread chunks) labelled as an *excerpt summary*, and show a hint: "For a written summary, switch to API or Ollama mode."
   - "compare/why/difference" → show the supporting sentences without a single span and the same hint.
   - Everything else → span + sentences.
6. **Be honest in the UI:** a small "Extractive · quoted from your document" label on each local answer. Never pretend it is a generated essay.

Accuracy knobs, all in config: `QA_MODEL`, `MIN_QA_SCORE`, `QA_WEIGHT`, `MAX_SENTENCES=3`. Tune `MIN_QA_SCORE` and the retrieval threshold with `scripts/evaluate.py` on the evaluation set.

## 3. Ollama mode (optional local generative)

- Talk to Ollama's local HTTP API (`OLLAMA_BASE_URL`, default `http://localhost:11434`; in Docker use `http://host.docker.internal:11434`). Streaming supported.
- `OLLAMA_MODEL` configurable; the UI lists installed models from Ollama's model-list endpoint. **Verify the current API paths and a good small default model in the Ollama docs before coding; do not guess.**
- Same grounded prompt and `NO_ANSWER` rule as API mode; same citation parser.
- If Ollama is unreachable, show: "Ollama isn't running. Start it, or switch to Local mode," with a Switch button.

## 4. API mode with the user's own key (BYOK)

- Key sources, in order: per-request header `X-LLM-Key` (user pasted it in Settings) → server env `ANTHROPIC_API_KEY`.
- **User-pasted keys are never stored on the server and never logged.** The browser keeps it in `sessionStorage` only (cleared when the tab closes) and sends it per request over HTTPS. A "Remember on this device" checkbox is allowed to use `localStorage` with a clear warning; default off.
- Settings panel: password-type input, "Test key" button (calls a minimal request and reports success/failure in plain words), link text "Where do I get a key?", and a line "Your key stays in this browser and is sent only to run your question."
- Model name comes from `LLM_MODEL` / a dropdown; **read the current IDs from the Claude Platform docs MCP, never hard-code from memory.**
- Optional: an `OpenAILLM` implementation behind the same protocol only if time remains.

## 5. Code structure

```
app/generation/
  base.py            # AnswerProvider protocol, AnswerMode enum, AnswerResult
  extractive.py      # LocalExtractiveProvider (QA model + sentence scoring + templates)
  ollama.py          # OllamaProvider
  anthropic.py       # AnthropicProvider (already planned as llm_anthropic.py)
  fake.py            # FakeProvider for tests
  router.py          # pick_provider(mode, user, request) + auto-selection
  question_type.py   # tiny rule-based classifier (factoid | summary | explain)
```

```python
class AnswerMode(str, Enum):
    LOCAL = "local"
    OLLAMA = "ollama"
    API = "api"

class AnswerProvider(Protocol):
    name: str
    def is_available(self) -> tuple[bool, str]: ...        # (ok, human-readable reason)
    def answer(self, question: str, chunks: list[RetrievedChunk],
               qtype: str) -> Iterator[AnswerEvent]: ...    # yields token / span / done events

class AnswerResult(BaseModel):
    mode: AnswerMode
    answerable: bool
    text: str                       # final text (markdown)
    span: Optional[str] = None      # exact extracted answer (local mode)
    span_char_range: Optional[tuple[int, int]] = None  # within the source chunk, for highlighting
    citations: list[Citation]
    confidence: Optional[float] = None
    label: str                      # "Extractive · quoted from your document" | "Generated by <model>"
```

Providers are selected through `router.py`; `chat_service` never imports a concrete provider. Local mode may emit the whole answer in one event (no token streaming needed) but must use the same SSE event types so the frontend has a single code path.

## 6. API changes

| Method | Path | Change |
|---|---|---|
| POST | `/chat/ask` | body gains `mode?: "local"|"ollama"|"api"`; response events include `mode` and `label` |
| GET | `/chat/modes` | returns each mode with `available`, `reason`, `description`, and the recommended one |
| PUT | `/me/preferences` | saves default mode (and Ollama model name); never the API key |
| POST | `/chat/test-key` | validates a pasted key via header; returns `{ok, message}` |

Persist `mode` on each assistant message so history shows how it was answered.

## 7. UI (follows `DESIGN_SYSTEM.md` tokens, no gradients)

- **Mode control:** segmented buttons above the input: `Local · no key` / `Ollama` / `Claude API`. Unavailable modes are disabled with a tooltip stating why and a "Set up" link.
- **Plain-language helper text** under the control, changing with the selection:
  - Local: "Finds and quotes the answer directly from your PDF. Free, private, works offline."
  - Ollama: "Writes an answer using a model running on your computer."
  - API: "Writes a fuller answer using Claude. Needs an API key."
- **Settings drawer:** API key field, Test key, Ollama URL/model picker, default mode, "Forget my key".
- **Answer card:** mode label chip (`Extractive` or `Generated`), the span highlighted in the source drawer, confidence shown as words ("High / Medium / Low match"), not as a raw number.
- **Hint for weak fits:** summary/compare questions in Local mode show the "switch mode" hint once, dismissible.
- **First-run:** if no key is set, the app just works in Local mode, with a quiet "Want fuller answers? Add an API key" link; no blocking dialogs.

## 8. Evaluation additions (`scripts/evaluate.py`)

Add an answer-quality stage per mode (API and Ollama only when available, otherwise report "skipped"):
- **Answer correctness:** Token-F1 and Exact-Match against short gold answers (extend `eval_set.json` with `gold_answer` on factoid items) and "contains gold span" rate.
- **Citation accuracy:** answer's top citation page equals the expected page.
- **Fallback accuracy:** unanswerable questions correctly refused.
- **Latency per mode.**
Print a Markdown table: rows = modes, columns = F1, EM, citation acc., fallback acc., median latency. Put the real numbers in the README and keep the honest conclusion (local = most faithful on factoids; API = better on synthesis).

## 9. Tests
- Router auto-selection (key present / Ollama up / neither).
- Extractive provider on a fixed fixture PDF returns the exact expected span and page; returns no-answer on an unrelated question.
- BYOK header is used and is never persisted or logged (assert on logs and DB).
- Same SSE event schema across all providers (contract test with `FakeProvider`).
- Ollama unreachable → friendly error event, not a 500.

## 10. Docker and size notes
Pre-download the QA model at build time with the other models. `roberta-base-squad2` adds roughly half a gigabyte; offer `QA_MODEL=distilbert-base-cased-distilled-squad` as a build arg for a smaller image. Ollama is **not** bundled; document how to run it on the host.

## 11. Honest trade-offs (put in README)
- Local extractive answers are very faithful but terse and cannot synthesize across passages or rephrase.
- API/Ollama answers read better but can still err; they are constrained by the grounded prompt, the gate and citations, not eliminated.
- The pasted-key flow trusts the user's browser; the server is a pass-through that does not retain keys.

---

# Paste-ready prompt for Claude Code (append to the main prompt, or run as a follow-up)

```
# ROLE
Same senior engineer as before. Work autonomously; verify by running; never guess library APIs.

# CONTEXT
The rag-chat project follows ARCHITECTURE.md, DESIGN_SYSTEM.md, MCP_SETUP.md and RAG_PROJECT_PLAN.md. A new binding document, ANSWER_MODES.md, adds multiple answer modes so the app works with no API key (Local extractive) and also with Ollama and the Claude API (including a user-pasted key). Read ANSWER_MODES.md fully first.

# TASK
1. Implement the provider layer in app/generation/ exactly as in ANSWER_MODES.md section 5: AnswerMode, AnswerProvider, AnswerResult, router with auto-selection, question-type classifier, LocalExtractiveProvider, OllamaProvider, AnthropicProvider (reuse/refactor the existing one), FakeProvider.
2. Local mode: extractive QA with handle_impossible_answer, combined scoring with rerank score, sentence selection via the existing cross-encoder, template-composed answer, question-type routing, MIN_QA_SCORE gate, span highlighting data for the source drawer.
3. Ollama mode against Ollama's local HTTP API with streaming, model listing and friendly unreachable errors.
4. API mode with key precedence: X-LLM-Key header, then env. Keys never stored or logged. Add POST /chat/test-key.
5. Endpoints per section 6, persist mode on messages, add the user preference for default mode.
6. Frontend per section 7 using the existing design tokens (no gradients): mode control, helper text, settings drawer with key field/Test key/Ollama picker, answer card labels, weak-fit hint, first-run behavior.
7. Extend tests and scripts/evaluate.py per sections 8 and 9; extend eval_set.json with gold_answer for factoid items; regenerate the README evaluation tables from real runs; add the trade-offs section.
8. Update Dockerfile/compose for the QA model (build arg for the distilled option) and document running Ollama on the host.

# CONSTRAINTS
- Before using transformers, Ollama's API, or the Anthropic SDK, fetch current docs via Context7 / claude-platform-docs (use context7). Do not guess pipeline arguments, endpoint paths, or model IDs. Mark anything unconfirmed as "unverified".
- All modes share retrieval and the no-answer gate. Local mode must never generate free text; every sentence it shows must be an exact quote from the PDF with a page citation.
- Keep layering: api -> services -> providers. chat_service must not import concrete providers.
- Same SSE event schema for every provider.
- No gradients; tokens only; plain-language UI copy; accessible.
- No scope beyond ANSWER_MODES.md (OpenAI provider only if everything else is done).

# EXPECTED OUTPUT
- Working app where: with no key and no Ollama, questions are answered in Local mode with quoted spans and page citations; with a pasted or env key, API mode streams fluent cited answers; with Ollama running, Ollama mode works; unanswerable questions are refused in all modes.
- Passing: ruff, pytest (including router, extractive fixture, BYOK-not-persisted, SSE contract tests), contrast script, evaluate.py with the per-mode table.
- Final report (under 300 words): what changed, the real per-mode evaluation table, docs audit (library/API -> doc source -> verified/unverified), anything not run (e.g. no API key or no Ollama available) and what I should run myself.
```
