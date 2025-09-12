import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Async NeonDB URL
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set. Check your .env file.")

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=True,       # Set to False in production
    future=True
)

# Async session factory
async_session = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Dependency for FastAPI
async def get_db():
    async with async_session() as session:
        yield session
