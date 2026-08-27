"""
Production-grade structured logging using structlog + loguru.
Provides JSON logs for production and colorized logs for development.
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

import structlog
from loguru import logger

# Context variable for request/pipeline correlation
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")
_pipeline_name: ContextVar[str] = ContextVar("pipeline_name", default="")


def get_correlation_id() -> str:
    """Get current correlation ID or generate a new one."""
    cid = _correlation_id.get()
    return cid if cid else str(uuid.uuid4())[:8]


def set_correlation_id(cid: str) -> None:
    _correlation_id.set(cid)


def set_pipeline_name(name: str) -> None:
    _pipeline_name.set(name)


def add_correlation_id(logger: Any, method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Structlog processor to inject correlation ID."""
    event_dict["correlation_id"] = get_correlation_id()
    pipeline = _pipeline_name.get()
    if pipeline:
        event_dict["pipeline"] = pipeline
    return event_dict


def safe_add_logger_name(logger: Any, method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Safely inject logger name if logger object has .name attribute."""
    if hasattr(logger, "name"):
        event_dict["logger"] = logger.name
    return event_dict


def configure_logging(
    log_level: str = "INFO",
    json_logs: bool = False,
    log_file: str | None = None,
) -> None:
    """
    Configure application-wide logging.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: Use JSON format (True for production)
        log_file: Optional file path for log output
    """
    # Remove default loguru handler
    logger.remove()

    # Structlog processors
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        safe_add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        add_correlation_id,
        structlog.processors.StackInfoRenderer(),
    ]

    if json_logs:
        # Production: JSON output
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
        log_format = "{message}"
    else:
        # Development: colorized human-readable
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[correlation_id]}</cyan> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, log_level.upper())),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Loguru for non-structlog code
    logger.add(
        sys.stdout,
        format=log_format,
        level=log_level.upper(),
        colorize=not json_logs,
        serialize=json_logs,
        backtrace=True,
        diagnose=not json_logs,
    )

    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {correlation_id} | {message}",
            level=log_level.upper(),
            rotation="100 MB",
            retention="30 days",
            compression="gz",
            serialize=True,
        )

    # Suppress noisy libraries in production
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.DEBUG if log_level == "DEBUG" else logging.WARNING
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a named structlog logger."""
    return structlog.get_logger(name)


class PipelineLogger:
    """Context manager for pipeline-level logging with timing."""

    def __init__(self, pipeline_name: str, run_id: str | None = None):
        self.pipeline_name = pipeline_name
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self._log = get_logger(pipeline_name)

    def __enter__(self) -> PipelineLogger:
        set_pipeline_name(self.pipeline_name)
        set_correlation_id(self.run_id)
        self._log.info("pipeline_started", run_id=self.run_id)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type:
            self._log.error(
                "pipeline_failed",
                run_id=self.run_id,
                error=str(exc_val),
                error_type=exc_type.__name__,
            )
        else:
            self._log.info("pipeline_completed", run_id=self.run_id)

    def info(self, event: str, **kwargs: Any) -> None:
        self._log.info(event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._log.warning(event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._log.error(event, **kwargs)
