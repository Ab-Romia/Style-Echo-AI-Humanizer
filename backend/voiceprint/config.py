"""Application configuration."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings."""

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True

    # Model Configuration
    sentence_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    style_model: str = "StyleDistance/styledistance"
    spacy_model: str = "en_core_web_sm"

    # Profile storage. SQLite file lives under a gitignored data dir.
    profile_db_path: str = "data/voiceprint.db"

    # Optional rewrite LLM (bring your own key). All empty by default; a key is
    # never committed. The client is OpenAI-compatible: leave the base URL at
    # the OpenAI default, or set it to https://openrouter.ai/api/v1 for
    # OpenRouter. Read from the environment so nothing sensitive lives in code.
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = "gpt-4o-mini"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
