# Platform Runnable Steps & Demos

This directory contains standalone, runnable demonstration scripts for each platform subsystem.

| Step | Script | Description |
|------|--------|-------------|
| 01 | `step01_gateway_retry.py` | LLM Gateway retry logic, jittered backoff, and automatic fallback switch. |
| 02 | `step02_embedding_math.py` | Hand-written loop calculations vs NumPy and unit vector geometry. |
| 03 | `step03_qdrant_setup.py` | Qdrant in-memory vector store, HNSW indexing, and idempotent UUID5 upserts. |
| 04 | `step04_ingest.py` | Ingestion pipeline: file loading, chunking, and JSONL persistence. |
| 05 | `step05_basic_rag.py` | Basic dense RAG query answering and strict citation verification. |
| 06 | `step06_hybrid_rag.py` | Hybrid Dense + BM25 keyword search with Reciprocal Rank Fusion (RRF). |
| 07 | `step07_memory_chat.py` | Multi-turn Redis conversation memory with lock-protected auto-compression. |
| 08 | `step08_react_agent.py` | From-scratch ReAct agent reasoning with safe AST calculator and tools. |
| 09 | `step09_crew_report.py` | Sequential CrewAI multi-agent report pipeline (Researcher -> Writer -> Reviewer). |
| 10 | `step10_ragas_eval.py` | Automated RAGAS evaluation with custom LLM and Embedder judge adapters. |
| 11 | `step11_tracing_demo.py` | Langfuse v3 observability, TTFT telemetry, and Prometheus metrics. |
| 12 | `step12_start_api.py` | Helper script to launch the FastAPI API server and web UI. |
| 13 | `step13_rerank_compare.py` | **Feature B**: Comparing retrieval precision with and without cross-encoder reranking. |

## Running any step

```bash
# Run with platform in PYTHONPATH
PYTHONPATH=src python steps/step01_gateway_retry.py
```
