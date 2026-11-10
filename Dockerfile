# python:3.11-slim: the image tag the architecture specifies (Python 3.11 runtime).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    TOKENIZERS_PARALLELISM=false \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# Install the exact, tested versions. torch resolves to the CPU-only wheel (smaller image).
COPY requirements.lock .
RUN pip install -r requirements.lock

# Bake every model into the image so the container starts fast and works offline:
# embeddings, the reranker, and the local answer model (used when no API key is set).
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
ARG RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
# QA model for Local mode. Smaller image: --build-arg QA_MODEL=deepset/roberta-base-squad2
ARG QA_MODEL=deepset/deberta-v3-large-squad2
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
from transformers import AutoTokenizer, AutoModelForQuestionAnswering; \
SentenceTransformer('${EMBED_MODEL}'); CrossEncoder('${RERANK_MODEL}'); \
AutoTokenizer.from_pretrained('${QA_MODEL}'); AutoModelForQuestionAnswering.from_pretrained('${QA_MODEL}')"

COPY alembic.ini ./
COPY migrations ./migrations
COPY app ./app
COPY assets ./assets
COPY scripts ./scripts
COPY samples ./samples

# Runs as root so the ./data bind mount is writable on any host (Linux creates it root-owned).
RUN mkdir -p /app/data

ENV HF_HUB_OFFLINE=1 \
    DATA_DIR=/app/data \
    QA_MODEL=${QA_MODEL} \
    WEB_CONCURRENCY=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

# WEB_CONCURRENCY > 1 needs the multi-server setup (Redis + Chroma server): see docker-compose.scale.yml.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY}"]
