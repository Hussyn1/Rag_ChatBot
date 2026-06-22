# ---- Build Stage ----
FROM python:3.12-slim AS builder

WORKDIR /build

# Install dependencies first (caching layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download the embedding model so it's baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ---- Runtime Stage ----
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local
COPY --from=builder /root/.cache /root/.cache

# Copy application code
COPY app/ ./app/
COPY static/ ./static/
COPY documents/ ./documents/

# Expose port (Railway injects PORT env var)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-8000}/api/health')" || exit 1

# Start the server
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
