# Multi-stage build for smaller image size (v2.0 - with semantic models)
FROM python:3.11-slim as builder

# Build argument for optional model preloading
ARG PRELOAD_EMBEDDINGS=false

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements-dev.txt ./
COPY pyproject.toml setup.py ./

# Install Python dependencies
RUN pip install --no-cache-dir --user -e .

# Optionally preload sentence-transformers model to reduce cold start
RUN if [ "$PRELOAD_EMBEDDINGS" = "true" ]; then \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || true; \
    fi

# Final stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.local/bin:$PATH" \
    NETHICAL_SEMANTIC=1

# Create app user for security
RUN useradd -m -u 1000 nethical && \
    mkdir -p /app /data /root/.cache && \
    chown -R nethical:nethical /app /data

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /root/.cache /root/.cache

# Copy application code
COPY --chown=nethical:nethical . .

# Install the package
RUN pip install --no-cache-dir -e .

# Switch to non-root user
USER nethical

# Expose port for API
EXPOSE 8000

# Volume for persistent data
VOLUME ["/data"]

# Health check (v2.0 - checks API)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=2)" || exit 1

# Default command - run API server (v2.0)
CMD ["uvicorn", "nethical.api:app", "--host", "0.0.0.0", "--port", "8000"]
