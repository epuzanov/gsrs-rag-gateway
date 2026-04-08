"""
GSRS RAG Gateway Configuration
"""
import json
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


_DEFAULT_IDENTIFIERS_ORDER = [
    "FDA UNII",
    "UNII",
    "SMS_ID",
    "SMSID",
    "ASK",
    "ASKP",
    "SVGID",
    "EVMPD",
    "xEVMPD",
    "CAS",
    "DRUG BANK",
    "RXCUI",
    "CHEMBL",
    "PUBCHEM",
]

_DEFAULT_CLASSIFICATIONS_ORDER = [
    "WHO-ATC",
    "WHO-VATC",
    "NCI_THESAURUS",
    "EMA ASSESSMENT REPORTS",
    "WHO-ESSENTIAL MEDICINES LIST",
    "NDF-RT",
    "LIVERTOX",
    "FDA ORPHAN DRUG",
    "EU-ORPHAN DRUG",
]


def _get_bool_env(name: str, default: bool) -> bool:
    """Parse a boolean environment variable with a sensible default."""
    value = os.getenv(name)
    if value is None:
        return default

    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_list_env(name: str, default: list[str]) -> list[str]:
    """Parse a list environment variable from JSON array or comma-separated values."""
    value = os.getenv(name)
    if value is None:
        return list(default)

    raw_value = value.strip()
    if not raw_value:
        return []

    if raw_value.startswith("["):
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [item for item in (str(entry).strip() for entry in parsed) if item]

    return [item for item in (part.strip() for part in raw_value.split(",")) if item]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore unused environment variables
    )

    # Database URL
    # PostgreSQL: postgresql://user:pass@host:port/dbname
    # ChromaDB: chroma://./chroma_data/chunks
    database_url: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/gsrs_rag")

    # Embedding API Configuration
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", "")
    embedding_url: str = os.getenv("EMBEDDING_URL", "https://api.openai.com/v1/embeddings")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    embedding_verify_ssl: bool = _get_bool_env("EMBEDDING_VERIFY_SSL", True)

    # Chunking Configuration
    chunker_identifiers_order: list[str] = _get_list_env(
        "CHUNKER_IDENTIFIERS_ORDER",
        _DEFAULT_IDENTIFIERS_ORDER,
    )
    chunker_classifications_order: list[str] = _get_list_env(
        "CHUNKER_CLASSIFICATIONS_ORDER",
        _DEFAULT_CLASSIFICATIONS_ORDER,
    )

    # API
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    # Authentication
    api_username: str = os.getenv("API_USERNAME", "admin")
    api_password: str = os.getenv("API_PASSWORD", "admin123")

    # Vector Search
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))


settings = Settings()
