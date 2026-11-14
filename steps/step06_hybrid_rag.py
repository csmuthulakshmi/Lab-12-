#!/usr/bin/env python3
"""Step 06: Hybrid Dense + BM25 Retrieval and Reciprocal Rank Fusion Demo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genai_platform.embeddings.embedder import HashEmbedder
from genai_platform.ingestion.schemas import Chunk
from genai_platform.rag.retrievers import HybridRetriever
from genai_platform.vectorstore.qdrant_store import QdrantStore


def main():
    print("=== Step 06: Hybrid Dense + BM25 Retrieval Demo ===")

    dim = 256
    embedder = HashEmbedder(dim=dim)
    store = QdrantStore(url=":memory:", collection=f"hybrid_demo_{dim}", dim=dim, recreate=True)

    chunks = [
        Chunk(
            text="PostgreSQL is an advanced, enterprise-class open-source relational database supporting SQL.",
            source="db.txt",
            doc_type="txt",
            page=1,
            index=0,
            strategy="recursive",
        ),
        Chunk(
            text="Redis is an in-memory key-value data structure store used as a database, cache, and message broker.",
            source="cache.txt",
            doc_type="txt",
            page=1,
            index=1,
            strategy="recursive",
        ),
        Chunk(
            text="Qdrant is a vector similarity search engine that provides production-ready vector indexing.",
            source="vector.txt",
            doc_type="txt",
            page=1,
            index=2,
            strategy="recursive",
        ),
    ]

    store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))

    retriever = HybridRetriever(store=store, embedder=embedder)

    query = "in-memory key-value cache"
    print(f"Query: '{query}'")

    hits = retriever.retrieve(query, k=2)
    print(f"\nTop {len(hits)} RRF results:")
    for i, h in enumerate(hits, start=1):
        print(f"  [{i}] (score: {h.score:.6f}) ({h.cite}) {h.text}")

    assert len(hits) > 0
    assert "Redis" in hits[0].text

    print("\nStep 06 completed successfully!")


if __name__ == "__main__":
    main()
