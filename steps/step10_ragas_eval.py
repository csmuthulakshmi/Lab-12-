#!/usr/bin/env python3
"""Step 10: RAGAS Automated Evaluation Pipeline Demo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genai_platform.embeddings.embedder import HashEmbedder
from genai_platform.evaluation.ragas_eval import run_ragas
from genai_platform.ingestion.schemas import Chunk
from genai_platform.llm.gateway import BaseBackend, LLMGateway, LLMResult, Usage
from genai_platform.rag.pipeline import RagPipeline
from genai_platform.rag.retrievers import DenseRetriever
from genai_platform.vectorstore.qdrant_store import QdrantStore


class MockJudgeLLM(BaseBackend):
    name = "mock_judge"
    model = "mock-judge"

    def chat(self, messages, temperature=0.7, max_tokens=None, stop=None):
        return LLMResult(
            text="The retrieved context directly supports the answer with high precision and faithfulness [1].",
            usage=Usage(prompt_tokens=60, completion_tokens=20, total=80),
            provider=self.name,
            model=self.model,
            latency_s=0.08,
        )

    def stream(self, messages, temperature=0.7, max_tokens=None, stop=None):
        yield from []


def main():
    print("=== Step 10: RAGAS Automated Evaluation Demo ===")

    dim = 256
    embedder = HashEmbedder(dim=dim)
    store = QdrantStore(url=":memory:", collection=f"eval_demo_{dim}", dim=dim, recreate=True)

    chunks = [
        Chunk(
            text="The platform provides a production RAG and agent service with multi-provider fallbacks and document management.",
            source="platform_spec.txt",
            doc_type="txt",
            page=1,
            index=0,
            strategy="recursive",
        ),
        Chunk(
            text="All embeddings are strictly L2-normalized float32 vectors.",
            source="embeddings_guide.txt",
            doc_type="txt",
            page=1,
            index=1,
            strategy="recursive",
        ),
    ]

    store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))
    retriever = DenseRetriever(store=store, embedder=embedder)
    llm = LLMGateway(primary=MockJudgeLLM())
    pipeline = RagPipeline(retriever=retriever, llm=llm)

    test_cases = [
        {
            "question": "What is the primary function of the platform?",
            "ground_truth": "The platform provides a production RAG and agent service.",
        },
        {
            "question": "How are embeddings normalized?",
            "ground_truth": "Embeddings are strictly L2-normalized float32 vectors.",
        },
    ]

    print(f"Running evaluation over {len(test_cases)} cases...")
    report = run_ragas(pipeline=pipeline, cases=test_cases, k=2, save=True)

    print("\nRagas Evaluation Report Scores:")
    for metric, score in report.get("scores", {}).items():
        print(f"  - {metric:20s}: {score:.4f}")

    print("\nReport persisted to artifacts/ragas_report.json")
    print("\nStep 10 completed successfully!")


if __name__ == "__main__":
    main()
