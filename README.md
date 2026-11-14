# GenAI Platform Plus ⚡

**GenAI Platform Plus** is an enterprise-grade, production-shaped Retrieval-Augmented Generation (RAG) and Multi-Agent platform built in Python 3.12 with FastAPI, Qdrant, Redis, and LangChain/CrewAI.

It features a resilient multi-provider LLM Gateway (Gemini, Ollama, OpenAI) with automatic retry and fallback, in-app document management, cross-encoder reranking, session-based conversation memory with automatic compression, a from-scratch ReAct agent, and end-to-end observability (Prometheus + Langfuse v3).

---

## 🏗️ Architecture

```mermaid
flowchart TD
    Client["Web Client / API Consumer"] -->|HTTP / SSE / REST| API["FastAPI Gateway (Port 8000)"]

    subgraph Security & Middleware
        Auth["X-API-Key Validator"]
        MetricsMW["Pure ASGI MetricsMiddleware"]
    end
    API --> Auth
    API --> MetricsMW

    subgraph LLM Gateway
        GW["LLMGateway (Primary + Fallback)"]
        Retry["Full-Jitter Retry + Exponential Backoff"]
        GW --> Retry
        Retry --> Gemini["Gemini Backend (google-genai)"]
        Retry --> Ollama["Ollama Backend (httpx)"]
        Retry --> OpenAI["OpenAI-Compatible Backend"]
    end

    subgraph Storage & Retrieval
        Store["Qdrant Vector Store (HNSW + Cosine)"]
        BM25["BM25Okapi Keyword Index"]
        RRF["Reciprocal Rank Fusion (RRF)"]
        Reranker["Cross-Encoder / Lexical Reranker"]
        Store --> RRF
        BM25 --> RRF
        RRF --> Reranker
    end

    subgraph Conversation Memory
        Redis["Redis (TTL + Distributed Locks)"]
        Compress["LLM Summarizer / Compressor"]
        Redis --> Compress
    end

    subgraph Multi-Agent Systems
        ReAct["ReAct Agent (AST Calculator, Wikipedia, RAG)"]
        Crew["CrewAI Report Crew (Researcher -> Writer -> Reviewer)"]
    end

    API --> GW
    API --> Storage & Retrieval
    API --> Conversation Memory
    API --> Multi-Agent Systems
```

---

## 📦 Technology Stack

| Layer | Component | Description |
|-------|-----------|-------------|
| **API Framework** | FastAPI + Uvicorn | High-performance ASGI API with streaming Server-Sent Events (SSE). |
| **LLM Gateway** | Custom `LLMGateway` | Resilient abstraction over Google GenAI, Ollama, and OpenAI endpoints with retry & automatic fallback. |
| **Vector Database** | Qdrant | In-memory or standalone vector search with HNSW indexes and payload filtering. |
| **Embeddings** | Local / Gemini / Hash | Sentence-transformers, Gemini embedding API, or offline deterministic HashEmbedder. |
| **Reranker** | CrossEncoder / Lexical | Joint-attention neural cross-encoders or BM25-style lexical rerankers. |
| **Memory** | Redis / fakeredis | Session-based memory with distributed locking, TTL, and automatic summarization. |
| **Agents** | ReAct / CrewAI | Custom ReAct agent with whitelist AST math evaluator, and sequential CrewAI report team. |
| **Evaluation** | RAGAS | Automated Faithfulness, Relevancy, Precision, and Recall scoring. |
| **Observability** | Prometheus + Langfuse | Real-time metrics exposition, TTFT, token telemetry, and execution traces. |

---

## 🌟 What's New vs. A Basic RAG API

1. **Feature A - Full Document Management**:
   - Web UI and REST API for dragging-and-dropping multi-format documents (`.pdf`, `.html`, `.txt`, `.md`).
   - Strict file validation, MIME type checks (415 on invalid extensions, 413 on files exceeding max size), and path traversal prevention.
   - Synchronized physical file deletion and vector database cleanup.
2. **Feature B - Cross-Encoder Reranking**:
   - Two-stage retrieval: candidates are fetched broadly via dense vectors and BM25 keywords, fused using Reciprocal Rank Fusion (RRF), and then precision-reranked using a Cross-Encoder.
   - Preserves original retrieval scores in `retrieval_score` while updating `score` with reranker outputs.
   - Live Prometheus metrics tracking `rerank_duration_seconds`.
3. **Resilient LLM Gateway with Fallback**:
   - Zero single point of failure: primary LLM failures (rate limits, 5xx server errors, network drops) automatically trigger exponential backoff with full jitter, server `Retry-After` parsing, and automatic fallback to a secondary LLM.
4. **Session Memory with Auto-Compression**:
   - Prevents context overflow by automatically folding older conversation turns into a dense summary under a Redis distributed lock when turn count or token budgets are reached.

---

## 🚀 Quick Start (Under 5 Minutes)

### 1. Prerequisites
- Python 3.12+
- `pip` or `uv`

### 2. Setup & Installation
```bash
# Clone or navigate to the repository
cd paradise

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env to set your GEMINI_API_KEY or configure Ollama
```

### 4. Run the API and Web UI
```bash
PYTHONPATH=src python steps/step12_start_api.py
```
Open **[http://localhost:8000/](http://localhost:8000/)** in your browser.

---

## 💻 Web UI Features

- **Theme Toggle**: Seamless switching between dark and light themes with automatic system preference detection.
- **API Key Management**: Secure storage in browser `localStorage`.
- **Live Health Monitor**: Automatic polling of `/health` indicating platform connectivity.
- **Document Management**: Drag-and-drop file upload with live progress, chunk counts, and deletion.
- **RAG vs. ReAct Agent**:
  - **Chat (RAG)**: Streams answer tokens in real-time. Numbered citations (e.g. `[1]`) are converted into interactive chips that open a context passage drawer. Grounding badges verify factual accuracy.
  - **ReAct Agent**: Live collapsible timeline visualizing `Thought -> Action -> Observation` steps.
- **Search Controls**: Real-time toggles for Hybrid search, Cross-Encoder Reranking, source filtering, and `k` passage count.

---

## 📡 API Reference & SSE Streaming

All endpoints except `/health`, `/metrics`, and `/` require the `X-API-Key` header. Missing or invalid keys return **401 Unauthorized**.

### Health & Metrics
- `GET /health`: Check Qdrant, Redis, LLM, and Reranker status (`ok` / `degraded`).
- `GET /metrics`: Prometheus metrics exposition.

### Ingestion & Documents
- `POST /documents/upload` (multipart): Upload and immediately ingest `.pdf`, `.html`, `.txt`, `.md` files.
- `GET /documents`: List all ingested sources, document types, chunk counts, and page counts.
- `DELETE /documents/{source}`: Remove all vector points and delete physical file from disk.
- `POST /ingest`: Ingest existing path inside `data/` (guarded against path traversal).

### Chat & Agent Streaming (SSE)
- `POST /chat/stream`:
  ```json
  {
    "message": "Explain vector indexing",
    "session_id": "session-123",
    "k": 4,
    "hybrid": true,
    "rerank": true
  }
  ```
  **SSE Event Order Guarantee:**
  1. `event: sources` (numbered passages emitted BEFORE generation starts)
  2. `event: metrics` (retrieval latency)
  3. `event: token` (repeated tokens as generated)
  4. `event: citations` (grounding verdict: cited, invalid, refused, grounded)
  5. `event: metrics` (final total latency, TTFT, and token usage)
  6. `event: done`

- `POST /agent/stream`: Streams `thought`, `action`, `observation`, `token`, `final`, and `done`.

### Multi-Agent & Evaluation
- `POST /crew/report`: Execute CrewAI report pipeline on a given topic.
- `POST /eval/ragas`: Run automated RAGAS evaluation or compare retrieval with/without reranking (`compare=true`).
- `DELETE /sessions/{session_id}`: Clear session memory in Redis.

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `gemini` | Primary LLM provider (`gemini`, `ollama`, `openai`). |
| `LLM_FALLBACK_PROVIDER` | `ollama` | Fallback LLM provider if primary fails retries. |
| `GEMINI_API_KEY` | `""` | Google GenAI API key. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name across all environments. |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API host. |
| `OLLAMA_MODEL` | `llama3.2` | Ollama model name. |
| `EMBEDDING_BACKEND` | `local` | Embeddings backend (`local`, `gemini`, `hash`). |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence transformer model. |
| `GEMINI_EMBEDDING_DIM`| `768` | Output dimensionality for Gemini embeddings. |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant vector database URL (or `:memory:`). |
| `QDRANT_COLLECTION` | `knowledge_base` | Collection name (auto-suffixed with dimension). |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL. |
| `RERANKER_BACKEND` | `lexical` | Reranker engine (`local`, `lexical`, `none`). |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model name. |
| `RERANK_CANDIDATES` | `20` | Number of base candidates to fetch before reranking. |
| `API_KEYS` | `dev-secret-key,test-key-123` | Comma-separated list of valid API keys. |
| `MAX_UPLOAD_MB` | `25` | Maximum allowed upload size per file in MB. |

---

## 🧪 Testing

The test suite runs completely offline without requiring network access, live LLMs, or external servers (utilizing HashEmbedder, in-memory Qdrant, and fakeredis).

```bash
PYTHONPATH=src pytest -q
```

The current offline suite contains 41 tests covering all phases, math operations, idempotent vector stores, RAG citation grounding, AST security, and API endpoints.

---

## 🐳 Docker Deployment

### Core Stack (API, Qdrant, Redis, Ollama)
```bash
docker compose --profile core up -d
```

### Full Stack (adds Prometheus, Grafana on port 3001, and Langfuse v3)
```bash
docker compose --profile full up -d
```

---

## 📝 Design Decisions & Defaults

Where the specification was silent, the following safe and complete defaults were selected:
1. **Gemini Model Uniformity**: `gemini-2.5-flash` is used consistently across `.env.example`, `config.py`, and `docker-compose.yml`.
2. **Qdrant Collection Dimensioning**: Collections are named with their embedding dimension suffix (e.g. `knowledge_base_768` or `knowledge_base_256`) to guarantee that switching embedding backends never corrupts an existing index.
3. **AST Calculator Whitelist**: Exponentiation (`**` and `^`) caps powers at 1000 to prevent computational denial-of-service.
4. **Pure ASGI Metrics Middleware**: Implemented as pure ASGI rather than Starlette `BaseHTTPMiddleware` to prevent buffering or breaking streaming SSE responses.
5. **No External CDN Dependencies**: The single-page web UI is completely self-contained with vanilla HTML, modern CSS variables, and native JavaScript without build steps.
