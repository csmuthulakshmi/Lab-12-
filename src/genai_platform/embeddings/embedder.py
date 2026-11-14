"""Embedder protocols and backends (Local sentence-transformers, Gemini, Hash)."""

from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache
from typing import Any, Protocol, Sequence

import numpy as np

from genai_platform.config import Settings, get_settings
from genai_platform.embeddings.math_ops import l2_normalize
from genai_platform.llm.gateway import (
    RateLimitError,
    RetryConfig,
    TransientError,
    call_with_retry,
    parse_retry_after,
)

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder for fast offline testing."""

    name = "hash"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _hash_text(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = re.findall(r"\w+", text.lower())
        if not words:
            # Deterministic non-zero fallback for empty strings
            vec[0] = 1.0
            return vec
        for word in words:
            digest = hashlib.md5(word.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.dim
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vec[idx] += sign
        return l2_normalize(vec)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        arr = np.vstack([self._hash_text(t) for t in texts])
        normalized = l2_normalize(arr)
        return normalized.astype(np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self._hash_text(text)
        return l2_normalize(vec).astype(np.float32).tolist()


class LocalEmbedder:
    """Local SentenceTransformer embedder."""

    name = "local"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: Any = None
        self._dim: int | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._get_model()
        return self._dim or 384

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        embeddings = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return l2_normalize(embeddings).astype(np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        model = self._get_model()
        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return l2_normalize(embedding).astype(np.float32).tolist()


class GeminiEmbedder:
    """Gemini Embeddings using official google-genai SDK."""

    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-004",
        dim: int = 768,
    ):
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self._client: Any = None
        self._retry_cfg = RetryConfig(max_retries=3, base_delay=1.0, cap_delay=15.0)

    def _get_client(self) -> Any:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY required for GeminiEmbedder")
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _embed_batch(self, texts: list[str], task_type: str) -> list[list[float]]:
        client = self._get_client()
        from collections import Counter
        from google.genai import types

        def _do_call():
            try:
                res = client.models.embed_content(
                    model=self.model,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.dim,
                    ),
                )
                embeddings = [e.values for e in res.embeddings]
                arr = np.asarray(embeddings, dtype=np.float32)
                return l2_normalize(arr).astype(np.float32).tolist()
            except Exception as e:
                msg = str(e)
                if "429" in msg or "quota" in msg.lower():
                    raise RateLimitError(msg, retry_after=parse_retry_after(None, msg)) from e
                raise TransientError(f"Gemini embedding transient failure: {msg}") from e

        stats = Counter()
        return call_with_retry(_do_call, self._retry_cfg, stats, label="gemini_embed")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        # Batch by 100
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            results.extend(self._embed_batch(batch, task_type="RETRIEVAL_DOCUMENT"))
        return results

    def embed_query(self, text: str) -> list[float]:
        res = self._embed_batch([text], task_type="RETRIEVAL_QUERY")
        return res[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    settings = get_settings()
    backend = settings.EMBEDDING_BACKEND.lower()
    if backend == "hash":
        return HashEmbedder(dim=256)
    elif backend == "gemini":
        return GeminiEmbedder(
            api_key=settings.GEMINI_API_KEY,
            model=settings.GEMINI_EMBEDDING_MODEL,
            dim=settings.GEMINI_EMBEDDING_DIM,
        )
    elif backend == "local":
        return LocalEmbedder(model_name=settings.EMBEDDING_MODEL)
    else:
        raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend}")
