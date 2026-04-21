"""
Database session management
SQLAlchemy 2.0 async session handling
"""

from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

# Create async engine
engine = create_async_engine(
    settings.async_database_url,
    echo=False,  # Set to True for SQL logging in development
    future=True,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,  # Verify connections before using
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Prevent expired object issues
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting database session
    Usage: async with get_db() as db:
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables (for development)"""
    from app.models.base import Base
    
    async with engine.begin() as conn:
        # await conn.run_sync(Base.metadata.create_all)
        pass  # Use Alembic migrations in production


async def close_db() -> None:
    """Close database connections"""
    await engine.dispose()
