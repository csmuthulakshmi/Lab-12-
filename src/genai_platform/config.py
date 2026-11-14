from functools import lru_cache
from pathlib import Path
from typing import Literal, Set
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT: Path = Path(__file__).resolve().parent.parent.parent
ARTIFACTS: Path = ROOT / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
DATA_RAW: Path = ROOT / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Settings
    LLM_PROVIDER: Literal["ollama", "gemini", "openai"] = "gemini"
    LLM_FALLBACK_PROVIDER: Literal["ollama", "gemini", "openai", "none"] | None = "ollama"
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    OPENAI_MODEL: str = "gemini-2.5-flash"

    # Embedding Settings
    EMBEDDING_BACKEND: Literal["local", "gemini", "hash"] = "local"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    GEMINI_EMBEDDING_MODEL: str = "text-embedding-004"
    GEMINI_EMBEDDING_DIM: int = 768

    # Vector Store & Cache
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "knowledge_base"
    REDIS_URL: str = "redis://localhost:6379/0"

    # Langfuse v3 Observability
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "http://localhost:3000"

    # Security & API
    API_KEYS: str = "dev-secret-key,test-key-123"
    MAX_UPLOAD_MB: int = 25

    # Reranker Settings
    RERANKER_BACKEND: Literal["local", "lexical", "none"] = "lexical"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_CANDIDATES: int = 20

    # Dev / Test overrides
    DEV_FAKE_INFRA: bool = False

    @property
    def api_key_set(self) -> Set[str]:
        if not self.API_KEYS:
            return set()
        return {k.strip() for k in self.API_KEYS.split(",") if k.strip()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
