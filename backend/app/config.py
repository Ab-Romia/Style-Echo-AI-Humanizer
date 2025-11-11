"""Application configuration."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings."""

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True

    # Database Configuration
    database_url: str = "postgresql://user:password@localhost:5432/voiceprint"
    redis_url: str = "redis://localhost:6379/0"

    # ChromaDB Configuration
    chroma_persist_dir: str = "./chroma_data"

    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # OpenAI API (optional)
    openai_api_key: str = ""

    # Rate Limiting
    free_tier_daily_limit: int = 3

    # Model Configuration
    sentence_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    spacy_model: str = "en_core_web_sm"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
