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

# Bake both models into the image so the container starts fast and works offline.
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
ARG RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBED_MODEL}'); CrossEncoder('${RERANK_MODEL}')"

COPY app ./app
COPY assets ./assets
COPY scripts ./scripts
COPY samples ./samples

# Runs as root so the ./data bind mount is writable on any host (Linux creates it root-owned).
RUN mkdir -p /app/data

ENV HF_HUB_OFFLINE=1 \
    DATA_DIR=/app/data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
