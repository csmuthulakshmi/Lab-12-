#!/usr/bin/env python3
"""Step 05: Basic Dense RAG Query and Citation Validation Demo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genai_platform.embeddings.embedder import HashEmbedder
from genai_platform.ingestion.schemas import Chunk
from genai_platform.llm.gateway import BaseBackend, LLMGateway, LLMResult, Usage
from genai_platform.rag.pipeline import RagPipeline
from genai_platform.rag.retrievers import DenseRetriever
from genai_platform.vectorstore.qdrant_store import QdrantStore


class MockLLM(BaseBackend):
    name = "mock_llm"
    model = "mock-model"

    def chat(self, messages, temperature=0.7, max_tokens=None, stop=None):
        return LLMResult(
            text="All embeddings in the platform are strictly L2-normalized float32 vectors [1].",
            usage=Usage(prompt_tokens=45, completion_tokens=15, total=60),
            provider=self.name,
            model=self.model,
            latency_s=0.1,
        )

    def stream(self, messages, temperature=0.7, max_tokens=None, stop=None):
        yield from []


def main():
    print("=== Step 05: Basic Dense RAG Pipeline Demo ===")

    dim = 256
    embedder = HashEmbedder(dim=dim)
    store = QdrantStore(url=":memory:", collection=f"dense_rag_{dim}", dim=dim, recreate=True)

    chunks = [
        Chunk(
            text="All embeddings in the platform are strictly L2-normalized float32 vectors.",
            source="embeddings_guide.txt",
            doc_type="txt",
            page=1,
            index=0,
            strategy="recursive",
        ),
        Chunk(
            text="Qdrant provides high-throughput similarity search with HNSW graphs.",
            source="vectorstore_guide.txt",
            doc_type="txt",
            page=1,
            index=0,
            strategy="recursive",
        ),
    ]

    store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))

    retriever = DenseRetriever(store=store, embedder=embedder)
    llm = LLMGateway(primary=MockLLM())
    pipeline = RagPipeline(retriever=retriever, llm=llm)

    question = "How are embeddings normalized?"
    print(f"Question: {question}")

    # Answer query
    ans = pipeline.answer(question, k=2)

    print(f"\nAnswer: {ans.answer}")
    print(f"Retrieved {len(ans.hits)} passages:")
    for src in ans.sources():
        print(f"  [{src['n']}] ({src['cite']}): {src['snippet']}")

    print(f"\nCitation Verification:")
    print(f"  Grounded : {ans.citations['grounded']}")
    print(f"  Cited    : {ans.citations['cited']}")
    print(f"  Invalid  : {ans.citations['invalid']}")

    print("\nStep 05 completed successfully!")


if __name__ == "__main__":
    main()
