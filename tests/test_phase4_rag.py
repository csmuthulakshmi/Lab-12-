import pytest

from genai_platform.embeddings.embedder import HashEmbedder
from genai_platform.ingestion.schemas import Chunk
from genai_platform.llm.gateway import BaseBackend, LLMGateway, LLMResult, Usage
from genai_platform.rag.pipeline import RagPipeline
from genai_platform.rag.prompts import (
    REFUSAL,
    format_context,
    validate_citations,
)
from genai_platform.rag.reranker import (
    LexicalReranker,
    NoopReranker,
    RerankingRetriever,
)
from genai_platform.rag.retrievers import (
    BM25Index,
    DenseRetriever,
    HybridRetriever,
    QueryRewriter,
    rrf,
    tokenize,
)
from genai_platform.vectorstore.qdrant_store import Hit, QdrantStore


def test_format_context_and_citations():
    hits = [
        Hit("1", 0.9, "First passage content.", "doc.pdf", 1, "pdf", "c1"),
        Hit("2", 0.8, "Second passage content.", "doc.pdf", 2, "pdf", "c2"),
    ]
    ctx = format_context(hits)
    assert "There are 2 passage(s)" in ctx
    assert "[1] (doc.pdf, p.1)" in ctx
    assert "[2] (doc.pdf, p.2)" in ctx

    # Grounded answer
    grounded = validate_citations("This statement is factual [1] and supported [2].", n=2)
    assert grounded["grounded"] is True
    assert grounded["cited"] == [1, 2]
    assert grounded["invalid"] == []
    assert grounded["refused"] is False

    # Invalid citation out of range
    invalid = validate_citations("This statement cites non-existent passage [5].", n=2)
    assert invalid["grounded"] is False
    assert 5 in invalid["invalid"]

    # Refusal
    refused = validate_citations(f"I don't know based on the provided documents.", n=2)
    assert refused["refused"] is True
    assert refused["grounded"] is True


def test_bm25_index_and_rrf():
    chunks = [
        Chunk(
            text="FastAPI is a modern web framework for Python.",
            source="fastapi.txt",
            doc_type="txt",
            page=1,
            index=0,
            strategy="recursive",
        ),
        Chunk(
            text="Qdrant provides vector search and similarity indexing.",
            source="qdrant.txt",
            doc_type="txt",
            page=1,
            index=1,
            strategy="recursive",
        ),
    ]

    index = BM25Index(chunks)
    hits = index.search("modern web framework", k=1)
    assert len(hits) == 1
    assert "FastAPI" in hits[0].text

    # Test RRF fusion
    r1 = [hits[0]]
    r2 = [Hit("q1", 0.95, "Other text", "q.txt", 1, "txt", "q1"), hits[0]]
    fused = rrf([r1, r2], k=60)
    assert len(fused) == 2
    # Chunk with appearances in both should rank top
    assert fused[0].id == hits[0].id


def test_feature_b_reranker_reorders_and_fallback():
    hits = [
        Hit("1", 0.70, "Unrelated topic on astrophysics and nebulae.", "astro.txt", 1, "txt", "c1"),
        Hit("2", 0.65, "Detailed guide on cross-encoder reranking algorithms.", "rerank.txt", 1, "txt", "c2"),
    ]
    reranker = LexicalReranker()
    reranked = reranker.rerank("cross-encoder reranking", hits, top_k=2)

    # Passage 2 must be moved to top
    assert reranked[0].id == "2"
    assert reranked[0].retrieval_score == 0.65
    assert reranked[0].rerank_score > 0

    # RerankingRetriever integration
    class FakeRetriever:
        def retrieve(self, query, k=4, source=None, config=None):
            return hits

    rr = RerankingRetriever(FakeRetriever(), reranker=reranker, candidates=10)
    out = rr.retrieve("cross-encoder reranking", k=1)
    assert len(out) == 1
    assert out[0].id == "2"


def test_hybrid_retriever_end_to_end():
    dim = 256
    embedder = HashEmbedder(dim=dim)
    store = QdrantStore(url=":memory:", collection=f"hybrid_test_{dim}", dim=dim, recreate=True)

    chunks = [
        Chunk(
            text="Machine learning models generalize patterns from data.",
            source="ml.txt",
            doc_type="txt",
            page=1,
            index=0,
            strategy="recursive",
        ),
        Chunk(
            text="Redis provides in-memory key-value storage with low latency.",
            source="redis.txt",
            doc_type="txt",
            page=1,
            index=1,
            strategy="recursive",
        ),
    ]
    store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))

    retriever = HybridRetriever(store=store, embedder=embedder)
    results = retriever.retrieve("in-memory redis storage", k=1)
    assert len(results) == 1
    assert "Redis" in results[0].text
