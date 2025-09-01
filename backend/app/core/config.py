from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    SECRET_KEY: str = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://contentrunway:password123@localhost:5432/contentrunway")
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # Milvus Vector Database
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Google AI (optional)
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    
    # Anthropic (optional)
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Personal Blog API
    BLOG_WEBHOOK_URL: str = os.getenv("BLOG_WEBHOOK_URL", "")
    BLOG_API_KEY: str = os.getenv("BLOG_API_KEY", "")
    
    # SearXNG Configuration
    SEARXNG_BASE_URL: str = os.getenv("SEARXNG_BASE_URL", "http://localhost/search")
    SEARXNG_SECRET_KEY: str = os.getenv("SEARXNG_SECRET_KEY", "your-secret-key-change-in-production")
    SEARXNG_ENABLED: bool = os.getenv("SEARXNG_ENABLED", "true").lower() == "true"
    SEARXNG_RATE_LIMIT: int = int(os.getenv("SEARXNG_RATE_LIMIT", "60"))  # searches per hour
    
    # Security
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:80",
        "http://127.0.0.1:3000",
    ]
    ALLOWED_HOSTS: List[str] = [
        "localhost",
        "127.0.0.1",
        "backend",
        "frontend",
    ]
    
    # Content Pipeline Settings
    QUALITY_THRESHOLD_OVERALL: float = 0.85
    QUALITY_THRESHOLD_TECHNICAL: float = 0.90
    QUALITY_THRESHOLD_DOMAIN: float = 0.90
    QUALITY_THRESHOLD_STYLE: float = 0.88
    QUALITY_THRESHOLD_COMPLIANCE: float = 0.95
    
    # Content Settings
    CONTENT_WORD_COUNT_MIN: int = 1200
    CONTENT_WORD_COUNT_MAX: int = 1800
    
    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()