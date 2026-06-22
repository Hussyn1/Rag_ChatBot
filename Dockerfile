# ---- Build Stage ----
FROM python:3.12-slim AS builder

WORKDIR /build

# Install dependencies first (caching layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so it's baked into the image
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ---- Runtime Stage ----
FROM python:3.12-slim

# Limit PyTorch CPU thread count to avoid memory overhead/OOM on multi-core hosts
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1

WORKDIR /app

# Copy installed packages and pre-downloaded model cache from builder
COPY --from=builder /usr/local /usr/local
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
CMD ["python", "-m", "app.main"]
