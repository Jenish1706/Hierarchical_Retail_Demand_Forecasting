"""
Database connection management using SQLAlchemy 2.0 with async support.
Provides both sync (for Airflow/scripts) and async (for FastAPI) interfaces.
"""
from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

from src.utils.logging import get_logger
from src.utils.settings import settings

log = get_logger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""
    pass


# ---------------------------------------------------------------------------
# Synchronous engine (Airflow, scripts, batch jobs)
# ---------------------------------------------------------------------------

def _create_sync_engine():
    engine = create_engine(
        settings.database.dsn,
        pool_size=settings.database.pool_size,
        max_overflow=settings.database.max_overflow,
        pool_timeout=settings.database.pool_timeout,
        pool_pre_ping=True,  # Validate connections before use
        echo=settings.database.echo,
        poolclass=QueuePool,
    )

    @event.listens_for(engine, "connect")
    def set_search_path(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("SET search_path TO public, processed, features, ml, inventory, pricing, monitoring, simulation, audit")
        cursor.close()

    return engine


sync_engine = _create_sync_engine()
SyncSessionFactory = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


@contextmanager
def get_sync_db() -> Generator[Session, None, None]:
    """Context manager for sync database sessions."""
    session = SyncSessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Async engine (FastAPI)
# ---------------------------------------------------------------------------

async_engine = create_async_engine(
    settings.database.async_dsn,
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_overflow,
    pool_pre_ping=True,
    echo=settings.database.echo,
)

AsyncSessionFactory = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


@asynccontextmanager
async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for database sessions."""
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def check_db_health() -> dict[str, str]:
    """Health check for database connectivity."""
    try:
        async with get_async_db() as session:
            result = await session.execute(text("SELECT version()"))
            version = result.scalar()
            return {"status": "healthy", "version": version}
    except Exception as e:
        log.error("db_health_check_failed", error=str(e))
        return {"status": "unhealthy", "error": str(e)}


def init_db() -> None:
    """Initialize database tables (for development/testing)."""
    Base.metadata.create_all(bind=sync_engine)
    log.info("database_initialized")
