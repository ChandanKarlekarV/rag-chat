# RAG "Chat with your Docs" — Complete Build Plan (2-day version)

## 1. Language Choice (the short answer)

| Layer | Language | Why |
|---|---|---|
| Backend / RAG pipeline | **Python 3.11** | Every ML/RAG library (sentence-transformers, Chroma, FAISS, PyMuPDF, rank-bm25) is Python-first. Using anything else means wrapping Python anyway. Recruiters expect Python for AI work. |
| Frontend | **Vanilla JavaScript + Tailwind (CDN)** | One HTML file, no build step, fastest to finish. Uses `fetch` + SSE for streaming. |
| Infra | **Docker + docker-compose** | One command to run. |
| Storage | **Chroma (persistent) + SQLite** | Chroma = vectors, SQLite = document metadata. No server to run. |

Do NOT use C/C++/Java/Node for the core. Use them only if you want a bonus (see Phase 8).

---

## 2. Architecture

```
 Browser (HTML + Tailwind + JS)
        |  upload PDF / ask question (SSE stream)
        v
 FastAPI ------------------------------------------------
   |                                                      |
   | INGEST                                               | QUERY
   v                                                      v
 PyMuPDF -> clean -> chunk (w/ page no.)        embed query
   |                                              |        |
   v                                              v        v
 embed chunks (bge-small)                   Vector search  BM25 search
   |                                              \        /
   v                                               RRF fusion (top 20)
 Chroma (vectors+metadata)                              |
 SQLite (docs table)                         Cross-encoder rerank (top 5)
 BM25 index (pickle)                                    |
                                          score < threshold? -> "No answer found"
                                                        |
                                          LLM prompt with numbered sources
                                                        |
                                          stream answer + citations [1][2]
```

---

## 3. Folder Structure

```
rag-chat/
├── app/
│   ├── main.py              # FastAPI app, routes, CORS, static mount
│   ├── config.py            # settings (env vars, model names, thresholds)
│   ├── schemas.py           # Pydantic request/response models
│   ├── ingest/
│   │   ├── loader.py        # PDF -> list[PageText]
│   │   ├── cleaner.py       # whitespace, hyphen fix, header/footer strip
│   │   └── chunker.py       # PageText -> list[Chunk]
│   ├── retrieval/
│   │   ├── embedder.py      # sentence-transformers wrapper (singleton)
│   │   ├── vector_store.py  # Chroma wrapper
│   │   ├── bm25_store.py    # keyword index
│   │   ├── hybrid.py        # RRF fusion
│   │   └── reranker.py      # cross-encoder
│   ├── generation/
│   │   ├── prompt.py        # prompt templates
│   │   └── llm.py           # LLM client (streaming)
│   ├── db.py                # SQLite helpers (documents table)
│   └── static/
│       ├── index.html
│       └── app.js
├── tests/
│   ├── test_chunker.py
│   ├── test_retrieval.py
│   └── eval_set.json        # 15-20 Q&A pairs for evaluation
├── scripts/
│   └── evaluate.py          # hit-rate / MRR / faithfulness check
├── data/                    # uploads + chroma + sqlite (docker volume)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .dockerignore
└── README.md
```

---

## 4. Tech Stack (exact picks)

| Need | Pick | Notes |
|---|---|---|
| Web framework | **FastAPI + Uvicorn** | async, auto docs at `/docs` |
| PDF parsing | **PyMuPDF (`pymupdf`)** | fast, gives page numbers |
| Embeddings | **`BAAI/bge-small-en-v1.5`** via sentence-transformers | free, local, 384-dim, strong quality. Fallback: `all-MiniLM-L6-v2` |
| Vector DB | **ChromaDB (PersistentClient)** | simplest, stores metadata |
| Keyword search | **`rank-bm25`** | enables hybrid search |
| Reranker | **`cross-encoder/ms-marco-MiniLM-L-6-v2`** | big accuracy boost, runs on CPU |
| LLM | **Claude (Anthropic API)** or OpenAI | one env var to switch; keep behind an `llm.py` interface |
| Validation | **Pydantic v2** | |
| Metadata DB | **SQLite** (stdlib `sqlite3`) | |
| Frontend | **Tailwind CDN + vanilla JS** | |
| Tests | **pytest** | |

`requirements.txt`:
```
fastapi
uvicorn[standard]
python-multipart
pymupdf
sentence-transformers
chromadb
rank-bm25
anthropic          # or openai
pydantic-settings
pytest
```

---

## 5. Data Structures

### 5.1 Core Python models (`schemas.py`)

```python
from pydantic import BaseModel
from typing import Optional

class PageText(BaseModel):
    page: int                # 1-indexed
    text: str

class Chunk(BaseModel):
    chunk_id: str            # f"{doc_id}:{index}"
    doc_id: str
    filename: str
    page_start: int
    page_end: int
    index: int               # position within the document
    text: str
    token_count: int

class DocumentMeta(BaseModel):
    doc_id: str              # uuid4 hex
    filename: str
    pages: int
    chunks: int
    uploaded_at: str         # ISO timestamp

class RetrievedChunk(BaseModel):
    chunk: Chunk
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None

class Citation(BaseModel):
    n: int                   # [1], [2] ...
    filename: str
    page_start: int
    page_end: int
    snippet: str             # first ~300 chars of chunk

class AskRequest(BaseModel):
    question: str
    doc_ids: Optional[list[str]] = None   # restrict to selected docs
    top_k: int = 5

class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    answerable: bool
```

### 5.2 Storage layout

**SQLite `documents` table**
```sql
CREATE TABLE documents (
  doc_id TEXT PRIMARY KEY,
  filename TEXT NOT NULL,
  pages INTEGER,
  chunks INTEGER,
  sha256 TEXT UNIQUE,        -- dedupe re-uploads
  uploaded_at TEXT
);
```

**Chroma collection `chunks`**
- `id` = `chunk_id`
- `document` = chunk text
- `embedding` = 384-float vector
- `metadata` = `{doc_id, filename, page_start, page_end, index}`
- Distance metric: **cosine** (`metadata={"hnsw:space": "cosine"}`)

**BM25**: in-memory `BM25Okapi` built from tokenized chunk texts; persist as pickle `{chunk_ids: [...], corpus_tokens: [[...]]}` and rebuild on startup / after each upload.

### 5.3 Key algorithms and their complexity
- **Chunking**: sliding window over sentences, O(n).
- **Vector search**: HNSW in Chroma, ~O(log n).
- **RRF fusion**: `score(d) = Σ 1 / (k + rank_i(d))` with k=60 across vector and BM25 rankings — a dict merge, O(m).
- **Rerank**: cross-encoder on top 20 pairs only (cost is bounded).

---

## 6. Phases (with time budget)

### Phase 0 — Setup (30 min)
- [ ] `python -m venv .venv`, install requirements
- [ ] Create folder structure above
- [ ] `config.py` with `pydantic-settings` reading `.env` (API key, model names, `CHUNK_SIZE=800`, `CHUNK_OVERLAP=150`, `TOP_K=5`, `MIN_RERANK_SCORE`)
- [ ] `git init`, `.gitignore` (`.env`, `data/`, `.venv/`)

**Done when:** `uvicorn app.main:app` serves `/docs`.

### Phase 1 — PDF Ingestion (1.5 hr)
- [ ] `loader.py`: open with PyMuPDF, return `list[PageText]`
- [ ] `cleaner.py`: collapse whitespace, join hyphenated line breaks (`exam-\nple`), drop lines repeating on >50% of pages (headers/footers)
- [ ] Reject files >20 MB / non-PDF / scanned PDFs with no text (clear error message)
- [ ] Compute SHA-256 to skip duplicate uploads

**Done when:** a test PDF returns clean text per page.

### Phase 2 — Chunking (1 hr) — *where quality is won*
- [ ] Split by paragraph, then sentence; pack sentences into chunks of ~800 characters (or ~200 tokens) with 150 overlap
- [ ] Never cross a document boundary; allow crossing pages but record `page_start`/`page_end`
- [ ] Assign `chunk_id = f"{doc_id}:{i}"`
- [ ] Unit tests: no empty chunks, overlap correct, page ranges correct

**Done when:** `pytest tests/test_chunker.py` passes.

### Phase 3 — Embeddings + Vector Store (1.5 hr)
- [ ] `embedder.py`: load model once (singleton), `embed_documents(list[str])` in batches of 32, `embed_query(str)` (bge needs the prefix `"Represent this sentence for searching relevant passages: "` on queries only)
- [ ] Normalize embeddings
- [ ] `vector_store.py`: `add_chunks`, `query(embedding, k, doc_ids)`, `delete_doc(doc_id)`
- [ ] `/upload` endpoint: save file → ingest → chunk → embed → store → insert SQLite row

**Done when:** upload works and Chroma returns sensible neighbors for a manual query.

### Phase 4 — Hybrid Retrieval + Rerank (2 hr) — *the recruiter-impressing part*
- [ ] `bm25_store.py`: tokenize (lowercase, regex words), build index, `search(query, k)`
- [ ] `hybrid.py`: get top 20 from vector + top 20 from BM25 → RRF merge → top 20 candidates
- [ ] `reranker.py`: score `(question, chunk.text)` pairs with the cross-encoder → keep top 5
- [ ] Keep all intermediate scores in `RetrievedChunk` (useful for the debug panel)

**Done when:** on your eval set, hybrid+rerank beats vector-only (you will measure this in Phase 7).

### Phase 5 — Generation + "No Answer" Fallback (1.5 hr)
- [ ] **Fallback gate:** if the best rerank score < `MIN_RERANK_SCORE` (tune on your eval set; start around 0), return `answerable=False` and "I couldn't find this in the uploaded documents." **without calling the LLM**
- [ ] `prompt.py` template:

```
You answer ONLY using the numbered sources below.
- Cite sources inline like [1], [2] after each claim.
- If the sources do not contain the answer, reply exactly: NO_ANSWER
- Never use outside knowledge.

SOURCES:
[1] (file.pdf, p.3) <chunk text>
[2] (file.pdf, p.7) <chunk text>

QUESTION: <question>
```
- [ ] `llm.py`: streaming call; detect `NO_ANSWER` and map it to `answerable=False`
- [ ] Parse `[n]` markers from the answer → build `Citation` list (only cited sources)
- [ ] `/ask` endpoint with **Server-Sent Events**: stream tokens, then a final event with citations

**Done when:** a question outside the document returns the fallback, and in-document questions return cited answers.

### Phase 6 — Frontend (2 hr)
- [ ] `index.html`: left sidebar (upload dropzone + document list with delete), right chat panel
- [ ] `app.js`: upload via `FormData`, show progress; ask via `fetch` + `ReadableStream` to render streamed tokens
- [ ] Render `[1]` as clickable chips → opens a side drawer showing filename, page, and highlighted snippet
- [ ] Checkbox per document to scope the question (`doc_ids`)
- [ ] Optional "debug" toggle showing vector/BM25/rerank scores

**Done when:** a non-technical person can use it without instructions.

### Phase 7 — Evaluation (1.5 hr) — *what makes it "senior-level"*
- [ ] Write `tests/eval_set.json`: 15–20 questions on a public PDF (e.g. a research paper) with the expected page number
- [ ] `scripts/evaluate.py` computes:
  - **Hit@5**: is the correct page among the top 5 retrieved chunks?
  - **MRR**
  - **Fallback accuracy**: add 5 unanswerable questions, check they are refused
- [ ] Compare 3 configs: vector-only, hybrid, hybrid+rerank. Put the table in the README.

Example README table:
| Config | Hit@5 | MRR |
|---|---|---|
| Vector only | 0.70 | 0.52 |
| Hybrid (RRF) | 0.80 | 0.61 |
| Hybrid + Rerank | 0.90 | 0.78 |

(Those are placeholders; use your real numbers.)

### Phase 8 — Docker + Polish (1.5 hr)
- [ ] `Dockerfile` (python:3.11-slim, install deps, **pre-download models at build time** so startup is fast)
- [ ] `docker-compose.yml`: one service, volume `./data:/app/data`, env file
- [ ] Healthcheck endpoint `/health`
- [ ] Basic error handling: LLM timeout, empty PDF, no documents uploaded
- [ ] Rate limit or max upload size
- [ ] README (see section 8)
- [ ] Deploy: Hugging Face Spaces (Docker) or Render so you have a **live link**

---

## 7. Two-Day Schedule

| Day | Block | Phases |
|---|---|---|
| **Day 1** | Morning | 0, 1, 2 |
| | Afternoon | 3, 4 |
| | Evening | 5 (backend fully working via `/docs`) |
| **Day 2** | Morning | 6 (frontend) |
| | Afternoon | 7 (evaluation), 8 (Docker) |
| | Evening | README, GIF, deploy, push to GitHub |

Cut list if you fall behind (in this order): debug panel, doc-scoped questions, deploy to cloud, rate limiting. **Never cut:** citations, fallback, evaluation table.

---

## 8. README Checklist (recruiters read this first)

- [ ] One-line pitch + demo GIF at the top
- [ ] Architecture diagram (copy the ASCII one above or draw it)
- [ ] Features: hybrid search, reranking, citations, no-answer fallback, streaming
- [ ] Evaluation table (vector vs hybrid vs hybrid+rerank)
- [ ] Design decisions: why bge-small, why chunk 800/150, why RRF, why a threshold
- [ ] Run instructions: `docker compose up`
- [ ] Limitations and future work (scanned PDFs/OCR, multi-user auth, tables)

---

## 9. Common Pitfalls

1. **Forgetting the bge query prefix** → worse retrieval.
2. **Re-loading the model per request** → 10 s latency. Load once at startup.
3. **Chunks too large** → vague matches. Too small → no context. Tune with your eval set.
4. **Trusting the LLM to refuse** → use the score threshold *and* the prompt rule.
5. **Not storing page numbers** → citations become useless.
6. **Putting the API key in code** → use `.env`, never commit it.
7. **BM25 index out of sync** after upload/delete → rebuild on every change (fine at this scale).

---

## 10. Optional Bonuses (only if time remains)

- **Multilingual**: swap in `bge-m3`.
- **Conversation memory**: rewrite follow-up questions into standalone queries before retrieval.
- **C++/Java angle**: none needed. If you want to show it off, write a small C++ chunk-tokenizer via pybind11, but it's not worth the time for a 2-day project.
- **Streaming citations highlight** in the PDF viewer using PDF.js.
