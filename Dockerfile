# Multi-stage build for hardened container (CIS Docker Benchmark & SOC 2 compliant)
FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704 as builder

# Build argument for optional model preloading
ARG PRELOAD_EMBEDDINGS=false

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging specifications to leverage Docker layer cache
COPY requirements.txt requirements-hashed.txt requirements-dev.txt pyproject.toml setup.py ./

# Install Python dependencies into isolated /install prefix
RUN pip install --no-cache-dir --timeout=120 --retries=5 --prefix=/install --require-hashes -r requirements-hashed.txt

# Final runtime stage
FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/usr/local/bin:/home/nethical/.local/bin:$PATH" \
    NETHICAL_SEMANTIC=1

# Upgrade system packages to fix CVEs and remove package manager caches
RUN apt-get update && apt-get upgrade -y --no-install-recommends && \
    apt-get install -y --no-install-recommends ca-certificates && \
    update-ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder into system location
COPY --from=builder /install /usr/local

# Create dedicated non-root application user
RUN useradd -m -u 1000 -s /bin/bash nethical && \
    mkdir -p /app /data /home/nethical/.cache && \
    chown -R nethical:nethical /app /data /home/nethical

WORKDIR /app

# Copy application source code with non-root ownership
COPY --chown=nethical:nethical . .

# Switch to non-root user BEFORE installing package in user space
USER nethical

# Install nethical package in user mode without re-downloading dependencies
RUN pip install --no-cache-dir --no-deps -e .

# Expose API port
EXPOSE 8000

# Volume for persistent data
VOLUME ["/data"]

# Health check using built-in urllib to avoid external dependencies
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)" || exit 1

# Run API server with unprivileged user
CMD ["uvicorn", "nethical.api:app", "--host", "0.0.0.0", "--port", "8000"]
