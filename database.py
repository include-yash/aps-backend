import os
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import logging

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set in .env file")

# Optional: toggle SQL echo via env
SQL_ECHO = os.getenv("SQL_ECHO", "False").lower() == "true"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("database")

# ---------------------------
# Create async SQLAlchemy engine
# ---------------------------
engine = create_async_engine(
    DATABASE_URL,
    echo=SQL_ECHO,          # Show SQL statements if True
    future=True,
    pool_size=int(os.getenv("DB_POOL_SIZE", 10)),        # Max connections in pool
    max_overflow=int(os.getenv("DB_MAX_OVERFLOW", 20)),  # Extra connections allowed beyond pool_size
    pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", 30)),  # Seconds to wait for a connection
)

# ---------------------------
# Base class for models
# ---------------------------
Base = declarative_base()

# ---------------------------
# Async session factory
# ---------------------------
async_session = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Avoid detached instances
)

# ---------------------------
# Dependency for FastAPI routes
# ---------------------------
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Async generator to provide a database session to FastAPI routes.
    Automatically commits if no exceptions; rolls back and closes session on error.
    """
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session rollback due to exception: {e}")
            raise
        finally:
            await session.close()

# ---------------------------
# Optional startup check
# ---------------------------
async def check_db_connection():
    """
    Optional: run this at startup to ensure DB is reachable.
    """
    try:
        async with engine.begin() as conn:
            await conn.execute("SELECT 1")
        logger.info("Database connection verified successfully")
    except Exception as e:
        logger.critical(f"Database connection failed: {e}")
        raise
