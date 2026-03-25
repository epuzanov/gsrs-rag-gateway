"""
GSRS RAG Gateway Configuration
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
import os


def _get_bool_env(name: str, default: bool) -> bool:
    """Parse a boolean environment variable with a sensible default."""
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore unused environment variables
    )

    # Database URL
    # PostgreSQL: postgresql://user:pass@host:port/dbname
    # ChromaDB: chroma://./chroma_data/substance_chunks
    database_url: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/substance_chunks")

    # Embedding API Configuration
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")
    embedding_url: str = os.getenv("EMBEDDING_URL", "https://api.openai.com/v1/embeddings")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    embedding_verify_ssl: bool = _get_bool_env("EMBEDDING_VERIFY_SSL", True)

    # API
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    # Authentication
    api_username: str = os.getenv("API_USERNAME", "admin")
    api_password: str = os.getenv("API_PASSWORD", "admin123")

    # Vector Search
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))


settings = Settings()
