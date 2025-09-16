from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.core.config import settings

# Create async engine with proper connection pooling
engine = create_async_engine(
    settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"),
    echo=False,  # Disable SQL logging for performance
    future=True,
    # Connection pooling configuration
    pool_size=20,           # Number of persistent connections
    max_overflow=10,        # Additional connections when pool is full
    pool_timeout=30,        # Max seconds to wait for connection
    pool_recycle=3600,      # Refresh connections after 1 hour
    pool_pre_ping=True,     # Validate connections before use
    # Connection arguments for asyncpg
    connect_args={
        "server_settings": {
            "jit": "off"  # Disable JIT for faster connection setup
        }
    }
)

# Create async session maker
AsyncSessionLocal = async_sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

Base = declarative_base()

async def get_db():
    """Dependency to get database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()