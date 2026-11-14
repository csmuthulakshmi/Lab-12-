import os
import sys
from pathlib import Path
import pytest

# Ensure `src` is in sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Set test environment variable defaults before settings is loaded
os.environ.setdefault("EMBEDDING_BACKEND", "hash")
os.environ.setdefault("QDRANT_URL", ":memory:")
os.environ.setdefault("RERANKER_BACKEND", "lexical")
os.environ.setdefault("OLLAMA_HOST", "http://localhost:1")
os.environ.setdefault("DEV_FAKE_INFRA", "1")
os.environ.setdefault("API_KEYS", "test-key-123,dev-secret-key")

from genai_platform.config import get_settings


@pytest.fixture(autouse=True)
def reset_settings(monkeypatch):
    """Ensure environment is cleanly configured for each test."""
    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("QDRANT_URL", ":memory:")
    monkeypatch.setenv("RERANKER_BACKEND", "lexical")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:1")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
