# Design decisions

Why rag-chat is built the way it is, including experiments that lost. See also [ARCHITECTURE.md](ARCHITECTURE.md) (the original spec) and [EVALUATION.md](EVALUATION.md).

| # | Decision | Why |
|---|---|---|
| 1 | **Hybrid + rerank over vector-only** | Measured in [EVALUATION.md](EVALUATION.md) (Hit@1 0.80 → 1.00). Vectors handle paraphrases; BM25 nails exact tokens (`E-214`, `HB-WM2`, port `502`); the cross-encoder fixes ordering. |
| 2 | **Local embeddings and a local answer model** | Privacy and zero cost. With no key, nothing leaves the machine. With a key, only the question and the top 5 passages go to the provider you chose. |
| 3 | **Extractive reader rather than a small local generative model** | An extractive model can only return text that's in the PDF, which is the strongest accuracy guarantee available without an LLM. It's trained on SQuAD 2.0, which includes unanswerable questions, so "no answer" is a first-class output. A small generative model would write fluent text and hallucinate more. Answers are less polished than an LLM's; that trade-off is the point of making LLMs optional. |
| 4 | **No "training" on your PDFs** | RAG doesn't need it: the model is pretrained to find answers in any text, and retrieval supplies the right text. Fine-tuning on a few PDFs would overfit. What is tuned is the thresholds, by measurement. |
| 5 | **Combined span score + answer-type check + supporting sentences** | Each fixes a measured failure. Mixing in the rerank score (`QA_WEIGHT` 0.6) stopped "red light on the *battery*" being answered from the *charger* manual. The type check (price → money, "how many" → a number) stopped "cost?" being answered with the weight. Quoting up to three cross-encoder-ranked sentences shows the qualifiers ("…plasterboard walls are not suitable") that a bare span misses. |
| 6 | **SQLite + Chroma by default, Postgres/Redis/Chroma server when scaling** | Zero ops for one container; the same code runs multi-worker by setting three URLs. |
| 7 | **httpOnly cookie over a localStorage JWT** | Scripts (including XSS) can't read the session. SameSite=Lax plus a required `X-Requested-With: fetch` header blocks CSRF; no CORS is enabled. |
| 8 | **Threshold before any model** | Cheaper and more reliable than relying on refusal alone. -5.0 is the highest gate value that refused no answerable development question. |
| 9 | **800/150 character chunks, RRF k = 60** | One fact plus its context per chunk; whole-sentence overlap; RRF needs no score normalization. |
| 10 | **No JS framework** | No build step; `app/static/js/` reads in minutes. |
| 11 | **One provider interface for every mode** | `AnswerProvider` (`generation/base.py`) plus a router (`generation/router.py`). Every mode shares retrieval, the gate, the SSE events and the citation parser, so the frontend has one code path and the modes are comparable. |
| 12 | **GPU when available** | `DEVICE=auto` puts all three local models on an NVIDIA GPU (about 12× faster for the QA model on an RTX 5070 Laptop: ~110 ms vs ~1.3 s per question), and `TORCH_THREADS` caps CPU use so the machine stays cool. |

## Choices made along the way (and experiments that lost)

- **Plain-text extraction beat block extraction.** Splitting pages into layout blocks keeps headings separate, but it scored lower: rerank Hit@1 fell to 0.97, and answers 29/30 instead of 30/30. A heading running into its first sentence gives the chunk useful topic context. So the loader keeps `get_text("text")`.
- **BM25+ instead of BM25Okapi:** Okapi's IDF goes negative on tiny per-user corpora (a test caught a one-chunk document scoring below zero). BM25+ keeps it positive, and only chunks containing a query term are returned.
- **Follow-up rewriting without a key** is a heuristic: when a question leans on context (pronouns, "and…", "what about…", very short), the previous question's key terms are added to the *search* query only. Measured: follow-up answers go from 0.70 to 1.00.
- **QA model vs the spec:** docs/ANSWER_MODES.md suggests `roberta-base-squad2`. The default is `deberta-v3-large-squad2` because it measured best (table above); the smaller models are a `QA_MODEL` setting or Docker build arg away.
- **No `question-answering` pipeline:** the spec asks for the Hugging Face pipeline with `handle_impossible_answer=True`. That pipeline is gone from transformers 5.x (absent from the installed 5.22.2 and from the current pipeline docs), so `generation/reader.py` implements the same rule explicitly: best span vs the model's "no answer" score, with `qa_score = sigmoid(margin)`.
- **Key header:** the pasted key travels as `X-LLM-Key` (spec section 4), in `sessionStorage` by default; "Remember on this device" uses `localStorage` with a warning.
- **Ollama default model:** `llama3.2` (used in Ollama's own docs examples). The UI lists whatever models are installed.
- **Multi-worker correctness:** per-user BM25 files are written atomically and reloaded when another worker changes them (mtime check). Index rebuilds take a per-user Redis lock. Embedded Chroma is single-process, so multi-worker mode requires `CHROMA_URL`; Compose runs the Chroma server from the same image, so client and server versions always match.
- **Migrations:** startup runs `alembic upgrade head`. Databases created by earlier versions (tables but no `alembic_version`) are stamped rather than recreated, so data is kept; tested.
- **Contrast fixes:** token values were adjusted (not the rule) where `check_contrast.py` failed, and a `--color-border-input` token meets WCAG 1.4.11.
- **`requirements.lock`** is resolved for the container target (CPython 3.11, Linux) by `scripts/lock.py`. A dev-machine `pip freeze` (Python 3.12) pinned packages that don't install on 3.11.
- **Logo:** an original mark (a document with a "verified answer" bubble) in the design tokens' fountain-pen blue and highlighter yellow, flat. The in-app version is an inline SVG coloured by tokens, so it follows the theme. Standalone files are in `assets/brand/`.
