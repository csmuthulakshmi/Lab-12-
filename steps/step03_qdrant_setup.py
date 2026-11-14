#!/usr/bin/env python3
"""Step 03: Qdrant Setup, In-Memory Store, and Idempotent Upserts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genai_platform.embeddings.embedder import HashEmbedder
from genai_platform.ingestion.schemas import Chunk
from genai_platform.vectorstore.qdrant_store import QdrantStore


def main():
    print("=== Step 03: Qdrant Vector Store Setup ===")

    dim = 256
    embedder = HashEmbedder(dim=dim)
    store = QdrantStore(url=":memory:", collection=f"test_coll_{dim}", dim=dim, recreate=True)

    print(f"Collection initialized in :memory: with dim={dim}")

    # Create toy chunks
    chunks = [
        Chunk(
            text="Vector databases store embeddings with fast approximate nearest neighbor search.",
            source="manual.pdf",
            doc_type="pdf",
            page=1,
            index=0,
            strategy="recursive",
        ),
        Chunk(
            text="Qdrant uses HNSW indexing and supports payload filtering.",
            source="manual.pdf",
            doc_type="pdf",
            page=2,
            index=1,
            strategy="recursive",
        ),
        Chunk(
            text="Python is an interpreted, high-level, general-purpose programming language.",
            source="python.txt",
            doc_type="txt",
            page=1,
            index=0,
            strategy="recursive",
        ),
    ]

    texts = [c.text for c in chunks]
    vectors = embedder.embed_documents(texts)

    print(f"\nUpserting {len(chunks)} chunks into Qdrant...")
    store.upsert(chunks, vectors)
    print(f"Total count after first upsert: {store.count()}")

    print("Re-upserting identical chunks to test idempotency...")
    store.upsert(chunks, vectors)
    print(f"Total count after second upsert (should remain 3): {store.count()}")
    assert store.count() == 3, "Idempotent upsert failed!"

    print("\nExecuting vector search for 'vector databases':")
    q_vec = embedder.embed_query("vector databases")
    hits = store.search(q_vec, k=2)
    for h in hits:
        print(f"  - [{h.score:.4f}] ({h.cite}) {h.text}")

    print("\nSource Statistics:")
    for s in store.source_stats():
        print(f"  - {s['source']} ({s['doc_type']}): {s['chunks']} chunks, {s['pages']} pages")

    print("\nStep 03 completed successfully!")


if __name__ == "__main__":
    main()
