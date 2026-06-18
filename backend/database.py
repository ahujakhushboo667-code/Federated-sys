from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from backend.config import settings
import asyncio
import logging
from sqlalchemy import text

logger = logging.getLogger(__name__)

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
)

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

async def get_db():
    max_retries = 3
    retry_delay = 2
    
    # Simple retry logic for connection acquisition
    for attempt in range(max_retries):
        try:
            async with AsyncSessionLocal() as session:
                # Ping to check if connection is alive
                await session.execute(text("SELECT 1"))
                break
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"DB connection failed, retrying in {retry_delay}s (Attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                logger.error("DB connection failed after max retries")
                raise e

    # Yield session for actual use
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
