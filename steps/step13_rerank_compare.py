#!/usr/bin/env python3
"""Step 13: Feature B Demo - Retrieval with and without Cross-Encoder Reranking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genai_platform.embeddings.embedder import HashEmbedder
from genai_platform.ingestion.schemas import Chunk
from genai_platform.rag.reranker import LexicalReranker, RerankingRetriever
from genai_platform.rag.retrievers import DenseRetriever
from genai_platform.vectorstore.qdrant_store import QdrantStore


def main():
    print("=== Step 13: Feature B - Reranker Retrieval Comparison ===")

    dim = 256
    embedder = HashEmbedder(dim=dim)
    store = QdrantStore(url=":memory:", collection=f"rerank_demo_{dim}", dim=dim, recreate=True)

    # Ingest a set of passages where lexical/semantic relevance differs
    chunks = [
        Chunk(
            text="The company announced its annual financial results and quarterly revenues yesterday.",
            source="finance.txt",
            doc_type="txt",
            page=1,
            index=0,
            strategy="recursive",
        ),
        Chunk(
            text="Cross-encoder rerankers compute joint attention between query and passage, improving precision.",
            source="rag_rerank.txt",
            doc_type="txt",
            page=1,
            index=1,
            strategy="recursive",
        ),
        Chunk(
            text="Bi-encoder embedding models map queries and documents into a shared vector space independently.",
            source="embeddings.txt",
            doc_type="txt",
            page=1,
            index=2,
            strategy="recursive",
        ),
        Chunk(
            text="Reranking candidate passages fetched from vector search significantly enhances top-1 accuracy.",
            source="rag_rerank.txt",
            doc_type="txt",
            page=2,
            index=3,
            strategy="recursive",
        ),
    ]

    store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))

    base_retriever = DenseRetriever(store=store, embedder=embedder)
    reranker = LexicalReranker()
    rerank_retriever = RerankingRetriever(base_retriever=base_retriever, reranker=reranker, candidates=4)

    query = "Why use cross-encoder rerankers over bi-encoder vector search?"
    print(f"Query: '{query}'\n")

    # 1. Base Retrieval without reranking
    print("1. Baseline Dense Vector Search (Without Reranking):")
    base_hits = base_retriever.retrieve(query, k=3)
    for i, h in enumerate(base_hits, start=1):
        print(f"   [{i}] (Score: {h.score:.4f}) ({h.cite}): {h.text}")

    # 2. Retrieval with Reranking
    print("\n2. Enhanced Retrieval (With Cross-Encoder / Lexical Reranker):")
    reranked_hits = rerank_retriever.retrieve(query, k=3)
    for i, h in enumerate(reranked_hits, start=1):
        print(f"   [{i}] (Rerank: {h.score:.4f} | Orig: {h.retrieval_score:.4f}) ({h.cite}): {h.text}")

    print("\nObservation:")
    print("  Notice how reranking surfaces the exact matching cross-encoder and accuracy passages")
    print("  to the top, while preserving original retrieval score in `retrieval_score`.")

    print("\nStep 13 completed successfully!")


if __name__ == "__main__":
    main()
