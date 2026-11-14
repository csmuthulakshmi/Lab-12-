#!/usr/bin/env python3
"""Step 04: Document Ingestion, Loaders, and JSONL Persistence."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genai_platform.config import DATA_RAW
from genai_platform.embeddings.embedder import HashEmbedder
from genai_platform.ingestion.pipeline import ingest_path, load_chunks
from genai_platform.vectorstore.qdrant_store import QdrantStore


def main():
    print("=== Step 04: Document Ingestion Pipeline Demo ===")

    # Create sample document in data/raw
    sample_file = DATA_RAW / "architecture_overview.md"
    sample_content = """# GenAI Platform Architecture Overview

The GenAI Platform Plus integrates a resilient multi-provider LLM Gateway with an advanced RAG retrieval engine.

## Vector Search & Embeddings
All incoming documents are parsed, cleaned, and split into chunks.
Embeddings are L2-normalized float32 vectors, stored in Qdrant with HNSW indexes.

## Multi-Provider Gateway
The gateway supports Gemini, Ollama, and OpenAI-compatible endpoints with automatic retries and fallbacks.
"""
    sample_file.write_text(sample_content, encoding="utf-8")
    print(f"Created sample document: {sample_file}")

    dim = 256
    embedder = HashEmbedder(dim=dim)
    store = QdrantStore(url=":memory:", collection=f"ingest_demo_{dim}", dim=dim, recreate=True)

    print("\nIngesting sample file...")
    stats = ingest_path(
        path=sample_file,
        store=store,
        embedder=embedder,
        strategy="recursive",
        chunk_size=300,
        overlap=50,
    )

    print(f"\nIngest Statistics:")
    print(f"  Total Chunks : {stats.get('total_chunks')}")
    print(f"  Avg Chars    : {stats.get('avg_chars'):.1f}")
    print(f"  Min Chars    : {stats.get('min_chars')}")
    print(f"  Max Chars    : {stats.get('max_chars')}")

    # Verify JSONL persistence
    saved_chunks = load_chunks()
    print(f"\nLoaded {len(saved_chunks)} chunks from artifacts/chunks.jsonl:")
    for c in saved_chunks[:2]:
        print(f"  - [{c.id}] ({c.cite}): {c.text[:80]}...")

    print("\nStep 04 completed successfully!")


if __name__ == "__main__":
    main()
