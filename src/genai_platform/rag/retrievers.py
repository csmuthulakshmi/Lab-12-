"""Retrievers: DenseRetriever, BM25Index, RRF, QueryRewriter, and HybridRetriever."""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Sequence

from rank_bm25 import BM25Okapi

from genai_platform.embeddings.embedder import Embedder, get_embedder
from genai_platform.ingestion.schemas import Chunk
from genai_platform.llm.gateway import LLMError, LLMGateway, get_gateway
from genai_platform.rag.prompts import MULTI_QUERY_PROMPT
from genai_platform.vectorstore.qdrant_store import Hit, QdrantStore, get_store

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she",
    "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves",
}


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words, filtering common stopwords."""
    words = re.findall(r"\b\w+\b", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


class BM25Index:
    """In-memory BM25 index built from Chunks."""

    def __init__(self, chunks: Sequence[Chunk]):
        self.chunks = list(chunks)
        self.corpus = [tokenize(c.text) for c in self.chunks]
        self._index = BM25Okapi(self.corpus) if self.corpus else None

    def search(self, query: str, k: int = 4, source: str | None = None) -> list[Hit]:
        if not self._index or not self.chunks:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._index.get_scores(tokens)
        candidates = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            if source is None or chunk.source == source:
                # BM25 can legitimately return zero for a one-document corpus
                # because its IDF is zero; token overlap still makes it useful.
                overlap = len(set(tokens) & set(self.corpus[idx]))
                if score > 0.0 or overlap:
                    candidates.append((score + overlap * 1e-6, chunk))

        candidates.sort(key=lambda x: x[0], reverse=True)
        top = candidates[:k]

        hits: list[Hit] = []
        for score, chunk in top:
            hits.append(
                Hit(
                    id=chunk.id,
                    score=float(score),
                    text=chunk.text,
                    source=chunk.source,
                    page=chunk.page,
                    doc_type=chunk.doc_type,
                    chunk_id=chunk.id,
                    retrieval_score=float(score),
                )
            )
        return hits


def rrf(rankings: Sequence[Sequence[Hit]], k: int = 60) -> list[Hit]:
    """
    Reciprocal Rank Fusion.
    Combined score for doc d = sum(1 / (k + rank_i(d))).
    """
    scores: dict[str, float] = {}
    hit_map: dict[str, Hit] = {}

    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            key = hit.chunk_id or hit.id
            hit_map[key] = hit
            scores[key] = scores.get(key, 0.0) + (1.0 / (k + rank))

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    fused: list[Hit] = []
    for key in sorted_keys:
        orig = hit_map[key]
        fused.append(
            Hit(
                id=orig.id,
                score=round(scores[key], 6),
                text=orig.text,
                source=orig.source,
                page=orig.page,
                doc_type=orig.doc_type,
                chunk_id=orig.chunk_id,
                retrieval_score=round(scores[key], 6),
            )
        )
    return fused


class DenseRetriever:
    """Standard dense vector retriever querying Qdrant."""

    def __init__(self, store: QdrantStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        k: int = 4,
        source: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> list[Hit]:
        vec = self.embedder.embed_query(query)
        return self.store.search(vec, k=k, source=source)


class QueryRewriter:
    """Uses LLM to rewrite user query into alternative perspectives."""

    def __init__(self, llm: LLMGateway | None = None, n: int = 3):
        self.llm = llm or get_gateway()
        self.n = n

    def rewrite(self, question: str) -> list[str]:
        prompt = MULTI_QUERY_PROMPT.format(n=self.n, question=question)
        messages = [{"role": "user", "content": prompt}]
        try:
            res = self.llm.chat(messages, temperature=0.7)
            lines = [line.strip().lstrip("0123456789.-) ") for line in res.text.splitlines() if line.strip()]
            queries = [question]
            for l in lines:
                if l and l.lower() != question.lower() and l not in queries:
                    queries.append(l)
            return queries[: self.n + 1]
        except (LLMError, Exception) as e:
            logger.warning(f"Query rewrite failed ({e}), falling back to original query")
            return [question]


class HybridRetriever:
    """
    Hybrid retriever combining Dense embeddings, BM25 keyword matching,
    optional multi-query expansion, and Reciprocal Rank Fusion (RRF).
    Auto-refreshes BM25 index when store chunk count changes.
    """

    def __init__(
        self,
        store: QdrantStore,
        embedder: Embedder | None = None,
        rewriter: QueryRewriter | None = None,
    ):
        self.store = store
        self.embedder = embedder or get_embedder()
        self.rewriter = rewriter
        self._lock = threading.Lock()
        self._bm25_index: BM25Index | None = None
        self._last_count = -1
        self._last_check_ts = 0.0
        self.refresh()

    def refresh(self) -> None:
        with self._lock:
            chunks = self.store.all_chunks()
            self._bm25_index = BM25Index(chunks)
            self._last_count = len(chunks)
            self._last_check_ts = time.time()

    def _maybe_auto_refresh(self) -> None:
        now = time.time()
        # Check at most every 30 seconds
        if now - self._last_check_ts > 30.0:
            self._last_check_ts = now
            current_count = self.store.count()
            if current_count != self._last_count:
                logger.info(f"Store chunk count changed from {self._last_count} to {current_count}. Refreshing BM25 index.")
                self.refresh()

    def retrieve(
        self,
        query: str,
        k: int = 4,
        source: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> list[Hit]:
        self._maybe_auto_refresh()

        queries = [query]
        if self.rewriter:
            queries = self.rewriter.rewrite(query)

        rankings: list[list[Hit]] = []

        for q in queries:
            # 1. Dense search
            dense_vec = self.embedder.embed_query(q)
            dense_hits = self.store.search(dense_vec, k=k * 2, source=source)
            if dense_hits:
                rankings.append(dense_hits)

            # 2. BM25 search
            if self._bm25_index:
                bm25_hits = self._bm25_index.search(q, k=k * 3, source=source)
                if bm25_hits:
                    rankings.append(bm25_hits)

        if not rankings:
            return []

        fused = rrf(rankings, k=60)
        return fused[:k]
