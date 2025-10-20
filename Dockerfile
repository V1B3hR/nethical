# Multi-stage build for smaller image size
FROM python:3.14-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.14-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.local/bin:$PATH"

# Create app user for security
RUN useradd -m -u 1000 nethical && \
    mkdir -p /app /data && \
    chown -R nethical:nethical /app /data

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY --chown=nethical:nethical . .

# Install the package
RUN pip install --no-cache-dir -e .

# Switch to non-root user
USER nethical

# Expose port (if running a server)
EXPOSE 8000

# Volume for persistent data
VOLUME ["/data"]

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import nethical; print('healthy')" || exit 1

# Default command
CMD ["python", "-c", "from nethical.core import IntegratedGovernance; print('Nethical container ready')"]
