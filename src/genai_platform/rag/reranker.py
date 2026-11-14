"""Cross-encoder, lexical, and noop rerankers and RerankingRetriever wrapper."""

from __future__ import annotations

import logging
import math
import re
import time
from functools import lru_cache
from typing import Any, Protocol, Sequence

from genai_platform.config import Settings, get_settings
from genai_platform.vectorstore.qdrant_store import Hit

logger = logging.getLogger(__name__)

# Lazy import / reference for prometheus metric
try:
    from genai_platform.observability.metrics import RERANK_DURATION
except Exception:
    RERANK_DURATION = None


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> list[Hit]:
        ...


class CrossEncoderReranker:
    """Reranker using sentence-transformers CrossEncoder."""

    name = "local"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        try:
            model = self._get_model()
            pairs = [[query, h.text] for h in hits]
            scores = model.predict(pairs)
        except Exception as e:
            logger.warning(f"CrossEncoder prediction failed: {e}. Falling back to lexical reranker.")
            lexical = LexicalReranker()
            return lexical.rerank(query, hits, top_k)

        reranked_hits: list[Hit] = []
        for h, score in zip(hits, scores):
            score_float = float(score)
            new_hit = Hit(
                id=h.id,
                score=score_float,
                text=h.text,
                source=h.source,
                page=h.page,
                doc_type=h.doc_type,
                chunk_id=h.chunk_id,
                retrieval_score=h.retrieval_score if h.retrieval_score is not None else h.score,
                rerank_score=score_float,
            )
            reranked_hits.append(new_hit)

        reranked_hits.sort(key=lambda x: x.score, reverse=True)
        return reranked_hits[:top_k]


class LexicalReranker:
    """Lightweight token-overlap / BM25-style reranker requiring zero external models."""

    name = "lexical"

    def _tokenize(self, text: str) -> list[str]:
        return [w for w in re.findall(r"\b\w+\b", text.lower()) if len(w) > 2]

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return list(hits[:top_k])

        reranked: list[Hit] = []
        for h in hits:
            doc_tokens = self._tokenize(h.text)
            overlap = 0.0
            doc_len = max(len(doc_tokens), 1)
            for qt in q_tokens:
                tf = doc_tokens.count(qt)
                if tf > 0:
                    overlap += (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * (doc_len / 50.0)))

            # Combine lexical score with existing retrieval score
            combined = overlap + 0.1 * (h.score)
            new_hit = Hit(
                id=h.id,
                score=round(combined, 4),
                text=h.text,
                source=h.source,
                page=h.page,
                doc_type=h.doc_type,
                chunk_id=h.chunk_id,
                retrieval_score=h.retrieval_score if h.retrieval_score is not None else h.score,
                rerank_score=round(combined, 4),
            )
            reranked.append(new_hit)

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_k]


class NoopReranker:
    """Pass-through reranker that keeps the original retrieval order."""

    name = "none"

    def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> list[Hit]:
        return list(hits[:top_k])


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    settings = get_settings()
    backend = settings.RERANKER_BACKEND.lower()
    if backend == "local":
        return CrossEncoderReranker(model_name=settings.RERANKER_MODEL)
    elif backend == "lexical":
        return LexicalReranker()
    elif backend == "none":
        return NoopReranker()
    else:
        logger.warning(f"Unknown RERANKER_BACKEND '{backend}', defaulting to LexicalReranker")
        return LexicalReranker()


class RerankingRetriever:
    """
    Retriever wrapper that pulls max(k, candidates) from a base retriever
    and applies reranking. Exposes retrieve(query, k, source, config).
    """

    def __init__(
        self,
        base_retriever: Any,
        reranker: Reranker | None = None,
        candidates: int = 20,
    ):
        self.base_retriever = base_retriever
        self.reranker = reranker or get_reranker()
        self.candidates = candidates

    def retrieve(
        self,
        query: str,
        k: int = 4,
        source: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> list[Hit]:
        fetch_k = max(k, self.candidates)
        base_hits = self.base_retriever.retrieve(query, k=fetch_k, source=source, config=config)
        if not base_hits:
            return []

        start = time.perf_counter()
        try:
            reranked = self.reranker.rerank(query, base_hits, top_k=k)
        except Exception as e:
            logger.warning(f"Reranking failed with error: {e}. Falling back to un-reranked hits.")
            reranked = base_hits[:k]
        finally:
            duration = time.perf_counter() - start
            if RERANK_DURATION:
                RERANK_DURATION.observe(duration)

        return reranked
