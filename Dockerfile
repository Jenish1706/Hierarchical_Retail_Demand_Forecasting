# =============================================================================
# Supply Chain Platform — FastAPI Service
# Multi-stage build: builder → runtime
# =============================================================================

# Stage 1: Builder — install dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for compiled packages (psycopg2, scipy, lightgbm, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into /install
COPY pyproject.toml .
RUN pip install --upgrade pip wheel setuptools && \
    pip install --prefix=/install --no-cache-dir \
        fastapi==0.104.0 \
        uvicorn[standard]==0.24.0 \
        pydantic==2.5.0 \
        pydantic-settings==2.1.0 \
        sqlalchemy==2.0.0 \
        asyncpg==0.29.0 \
        psycopg2-binary==2.9.9 \
        slowapi==0.1.9 \
        prometheus-client==0.19.0 \
        httpx==0.25.0 \
        python-multipart==0.0.6 \
        mlflow==2.9.0 \
        lightgbm==4.2.0 \
        xgboost==2.0.0 \
        catboost==1.2.2 \
        scikit-learn==1.3.2 \
        numpy==1.26.0 \
        pandas==2.1.0 \
        scipy==1.11.0 \
        loguru==0.7.2 \
        structlog==23.2.0 \
        python-dotenv==1.0.0 \
        tenacity==8.2.3 \
        pyyaml==6.0.1 \
        holidays==0.40

# Stage 2: Runtime — lean image
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN groupadd -r scplatform && useradd -r -g scplatform -d /app scplatform

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=scplatform:scplatform src/ ./src/
COPY --chown=scplatform:scplatform configs/ ./configs/

# Environment
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

USER scplatform

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Uvicorn with graceful shutdown
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--access-log", \
     "--log-level", "info", \
     "--timeout-keep-alive", "30"]
