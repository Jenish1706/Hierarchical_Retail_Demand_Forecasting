"""
Supply Chain Intelligence Platform — FastAPI Application.

Production-grade REST API with:
  - Request validation (Pydantic v2)
  - Rate limiting (slowapi)
  - Structured logging with correlation IDs
  - Prometheus metrics
  - Health checks
  - OpenAPI documentation
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.routers import forecast, inventory, pricing, simulation, optimization, monitoring
from src.utils.database import check_db_health
from src.utils.logging import configure_logging, get_logger, set_correlation_id
from src.utils.settings import settings

configure_logging(
    log_level=settings.log_level,
    json_logs=settings.is_production,
)
log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "API request latency",
    ["method", "endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
)

# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("api_starting", version=settings.app_version, env=settings.environment)
    # Startup: verify DB connection
    health = await check_db_health()
    if health["status"] != "healthy":
        log.error("startup_db_check_failed", **health)
    else:
        log.info("startup_db_ok", version=health.get("version", ""))
    yield
    log.info("api_shutdown")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    app = FastAPI(
        title="Supply Chain Intelligence Platform",
        description="""
## Intelligent Supply Chain Decision Intelligence API

Provides ML-powered endpoints for:
- **Demand Forecasting** with probabilistic intervals
- **Inventory Optimization** (EOQ, safety stock, reorder points)
- **Dynamic Pricing** recommendations
- **Supply Chain Simulation** and stress testing
- **Multi-Objective Optimization** with Pareto fronts
- **Model Monitoring** and drift detection
        """,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else ["https://dashboard.scplatform.io"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ---------------------------------------------------------------------------
    # Middleware: correlation ID + metrics
    # ---------------------------------------------------------------------------
    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4())[:8])
        set_correlation_id(correlation_id)
        start_time = time.monotonic()

        response = await call_next(request)

        duration = time.monotonic() - start_time
        endpoint = request.url.path
        method = request.method

        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=response.status_code).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Response-Time"] = f"{duration:.4f}s"
        return response

    # ---------------------------------------------------------------------------
    # System endpoints
    # ---------------------------------------------------------------------------
    @app.get("/health", tags=["System"], summary="Health check")
    async def health_check():
        db_health = await check_db_health()
        return {
            "status": "healthy" if db_health["status"] == "healthy" else "degraded",
            "app_version": settings.app_version,
            "environment": settings.environment,
            "database": db_health,
        }

    @app.get("/metrics", tags=["System"], summary="Prometheus metrics")
    async def prometheus_metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/", tags=["System"], include_in_schema=False)
    async def root():
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
        }

    # ---------------------------------------------------------------------------
    # Business routers
    # ---------------------------------------------------------------------------
    app.include_router(forecast.router, prefix="/forecast", tags=["Forecast"])
    app.include_router(inventory.router, prefix="/inventory", tags=["Inventory"])
    app.include_router(pricing.router, prefix="/pricing", tags=["Pricing"])
    app.include_router(simulation.router, prefix="/simulation", tags=["Simulation"])
    app.include_router(optimization.router, prefix="/optimization", tags=["Optimization"])
    app.include_router(monitoring.router, prefix="/monitoring", tags=["Monitoring"])

    return app


app = create_app()
