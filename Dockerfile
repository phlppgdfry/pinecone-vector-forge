# ==============================================================================
# Pinecone Vector Forge — Production Container Image
# Multi-stage, minimal attack-surface, non-root runner with healthcheck
# ==============================================================================

# Stage 1: Build virtual environment with dependencies
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

# Stage 2: Minimal runtime image
FROM python:3.12-slim AS runtime

# Set security and performance runtime flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    APP_ENV=production

WORKDIR /app

# Create unprivileged system user
RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /sbin/nologin -d /app appuser

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application source code
COPY . /app

# Set appropriate permissions
RUN chown -R appuser:appgroup /app

# Switch to unprivileged user
USER appuser

# Expose FastAPI HTTP port
EXPOSE 8000

# Docker healthcheck querying Pinecone status endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/status', timeout=3).getcode() == 200 else 1)"

# Start production ASGI server
CMD ["python", "-m", "uvicorn", "src.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
