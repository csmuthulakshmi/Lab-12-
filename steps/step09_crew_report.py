#!/usr/bin/env python3
"""Step 09: CrewAI Multi-Agent Report Pipeline Demo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genai_platform.agents.crew import run_report
from genai_platform.embeddings.embedder import HashEmbedder
from genai_platform.ingestion.schemas import Chunk
from genai_platform.rag.retrievers import DenseRetriever
from genai_platform.vectorstore.qdrant_store import QdrantStore


def main():
    print("=== Step 09: CrewAI Multi-Agent Report Demo ===")

    dim = 256
    embedder = HashEmbedder(dim=dim)
    store = QdrantStore(url=":memory:", collection=f"crew_demo_{dim}", dim=dim, recreate=True)

    chunks = [
        Chunk(
            text="Autonomous AI agents utilize recursive thought patterns, memory systems, and tools.",
            source="agents_whitepaper.pdf",
            doc_type="pdf",
            page=1,
            index=0,
            strategy="recursive",
        ),
        Chunk(
            text="Multi-agent frameworks like CrewAI enable specialized roles such as researchers, writers, and fact-checkers.",
            source="crewai_guide.pdf",
            doc_type="pdf",
            page=1,
            index=1,
            strategy="recursive",
        ),
    ]

    store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))
    retriever = DenseRetriever(store=store, embedder=embedder)

    topic = "Role-based AI multi-agent architectures"
    print(f"Executing report pipeline on topic: '{topic}'...")

    result = run_report(retriever=retriever, topic=topic)

    print(f"\nGenerated Report Summary:")
    print(f"Topic: {result.get('topic')}")
    print(f"Report Output:\n{result.get('report')[:300]}...\n")
    print(f"Tasks executed: {len(result.get('task_outputs', []))}")

    print("\nStep 09 completed successfully!")


if __name__ == "__main__":
    main()
