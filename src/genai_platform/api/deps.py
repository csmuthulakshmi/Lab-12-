"""Dependency injection and singletons for FastAPI API routes."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from fastapi import Header, HTTPException, Security, status
import redis

from genai_platform.config import Settings, get_settings
from genai_platform.embeddings.embedder import Embedder, get_embedder
from genai_platform.llm.gateway import LLMGateway, get_gateway
from genai_platform.memory.redis_memory import RedisMemory, make_summarizer
from genai_platform.rag.pipeline import RagPipeline
from genai_platform.rag.reranker import (
    NoopReranker,
    RerankingRetriever,
    get_reranker,
)
from genai_platform.rag.retrievers import DenseRetriever, HybridRetriever, QueryRewriter
from genai_platform.vectorstore.qdrant_store import QdrantStore, get_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API Key Authentication
# ---------------------------------------------------------------------------

def verify_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> str:
    """Validate X-API-Key header. Returns 401 on missing or invalid key."""
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing required X-API-Key header",
        )
    settings = get_settings()
    if x_api_key not in settings.api_key_set:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key provided",
        )
    return x_api_key


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_redis_client() -> Any:
    settings = get_settings()
    if settings.DEV_FAKE_INFRA:
        try:
            import fakeredis
            return fakeredis.FakeRedis()
        except ImportError:
            pass
    try:
        return redis.Redis.from_url(settings.REDIS_URL, decode_responses=False)
    except Exception as e:
        logger.warning(f"Could not connect to Redis at {settings.REDIS_URL}: {e}")
        import fakeredis
        return fakeredis.FakeRedis()


@lru_cache(maxsize=1)
def get_memory() -> RedisMemory:
    client = get_redis_client()
    gw = get_gateway()
    summarizer = make_summarizer(gw)
    embedder = get_embedder()
    return RedisMemory(
        client=client,
        summarizer=summarizer,
        embedder=embedder,
        max_turns=10,
        token_budget=1500,
    )


@lru_cache(maxsize=1)
def get_hybrid_retriever() -> HybridRetriever:
    store = get_store()
    embedder = get_embedder()
    gw = get_gateway()
    rewriter = QueryRewriter(llm=gw, n=2)
    return HybridRetriever(store=store, embedder=embedder, rewriter=rewriter)


@lru_cache(maxsize=1)
def get_dense_retriever() -> DenseRetriever:
    store = get_store()
    embedder = get_embedder()
    return DenseRetriever(store=store, embedder=embedder)


@lru_cache(maxsize=1)
def get_rag_pipeline(hybrid: bool = True, rerank: bool = True) -> RagPipeline:
    """Build RagPipeline with appropriate retrieval and reranking configuration."""
    base = get_hybrid_retriever() if hybrid else get_dense_retriever()
    gw = get_gateway()
    if rerank:
        reranker = get_reranker()
        settings = get_settings()
        retriever = RerankingRetriever(base, reranker=reranker, candidates=settings.RERANK_CANDIDATES)
    else:
        retriever = base
    return RagPipeline(retriever=retriever, llm=gw)
