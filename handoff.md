# GenAI Platform Plus - Handoff Guide

This file is the first document an implementation or operations agent should
read before changing or running the project. It describes the supported setup,
the dependency split, the safe first run, and the behavior expected from the
platform.

## 1. What this project is

GenAI Platform Plus is a Python 3.12 FastAPI application providing:

- RAG chat with dense, BM25, hybrid/RRF retrieval, citations, and reranking.
- PDF, HTML, Markdown, and text document upload, ingestion, listing, and delete.
- Streaming SSE chat and ReAct agent endpoints.
- Redis-backed session memory with automatic summarization.
- Gemini, Ollama, and OpenAI-compatible LLM providers with retries and fallback.
- Qdrant vector storage, Prometheus metrics, optional Langfuse tracing.
- Optional CrewAI reports and RAGAS evaluation.
- A browser UI served at `/`.

The offline test suite uses hash embeddings, in-memory Qdrant, fakeredis, and fake
LLM implementations. It does not need API keys, Docker, Redis, Qdrant, or model
downloads.

## 2. Prerequisites

Install these on the target machine:

- Python 3.12 or newer.
- Git.
- `pip` or `uv`.
- Docker Desktop and Docker Compose only if using the containerized services.
- An LLM credential or a local Ollama installation for real answers.

Recommended checks:

```bash
python3.12 --version
git --version
docker --version       # only required for Docker mode
docker compose version # only required for Docker mode
```

## 3. Clone and create the environment

```bash
git clone https://github.com/csmuthulakshmi/Lab-12-.git
cd Lab-12-

python3.12 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements-core.txt
```

`requirements-core.txt` installs the API, retrieval, embedding interfaces,
testing, metrics, and local runtime dependencies. It includes
`sentence-transformers`; the package is installed during setup, but its model
weights are downloaded only when `EMBEDDING_BACKEND=local` is actually used.

For the smallest offline smoke test, the test suite does not require the local
embedding model. Run:

```bash
PYTHONPATH=src python -m pytest -q
```

Expected result: all current tests pass. Warnings about deprecated third-party
APIs or in-memory Qdrant behavior are non-fatal.

## 4. Optional dependencies

Do not install these for the first smoke test. Install them only when the
corresponding features are needed:

```bash
pip install -r requirements-optional.txt
```

This downloads CrewAI, RAGAS, and Hugging Face datasets plus their transitive
dependencies. It can be large and may take several minutes. Without them:

- `/crew/report` returns a graceful fallback report.
- `/eval/ragas` returns a graceful fallback evaluation result.
- Core API, upload, ingestion, chat, agent, and retrieval functionality remains
  available.

## 5. Configure the application

```bash
cp .env.example .env
```

Edit `.env`. Never commit `.env` or API keys.

### Fastest real setup: Gemini

```dotenv
LLM_PROVIDER=gemini
LLM_FALLBACK_PROVIDER=ollama
GEMINI_API_KEY=replace_with_your_key
EMBEDDING_BACKEND=hash
QDRANT_URL=:memory:
RERANKER_BACKEND=lexical
API_KEYS=dev-secret-key
```

This avoids downloading an embedding model and runs Qdrant in process. Data is
not durable across application restarts in this mode.

### Persistent local-quality setup

```dotenv
LLM_PROVIDER=gemini
LLM_FALLBACK_PROVIDER=ollama
GEMINI_API_KEY=replace_with_your_key
EMBEDDING_BACKEND=local
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379/0
RERANKER_BACKEND=lexical
```

The first local embedding request downloads the sentence-transformers model
weights. A local cross-encoder reranker also downloads model weights and is
therefore not the recommended first-run setting:

```dotenv
RERANKER_BACKEND=local
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Ollama setup

Install Ollama separately, then download the configured model:

```bash
ollama pull llama3.2
ollama serve
```

Use:

```dotenv
LLM_PROVIDER=ollama
LLM_FALLBACK_PROVIDER=gemini
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

The Gemini fallback requires `GEMINI_API_KEY`.

## 6. Start supporting services

For persistent Qdrant and Redis without starting the entire observability stack:

```bash
docker compose --profile core up -d qdrant redis
```

Then use:

```dotenv
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379/0
```

The Compose file also defines Ollama, Prometheus, Grafana, and Langfuse under
the `full` profile. The full profile is resource-heavy:

```bash
docker compose --profile full up -d
```

Do not expose Redis, Qdrant, Prometheus, Grafana, or Langfuse directly to the
public internet without adding network and authentication controls.

## 7. Start the API and web UI

From the repository root with the virtual environment active:

```bash
PYTHONPATH=src python steps/step12_start_api.py
```

Alternatively:

```bash
PYTHONPATH=src uvicorn genai_platform.api.main:app --host 127.0.0.1 --port 8000
```

Open <http://localhost:8000/>. The API documentation is at
<http://localhost:8000/docs>.

Check readiness:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

`/health` may report `degraded` when an optional provider or external service
is unavailable. That is expected if the configured fallback is not running.

## 8. First-use workflow

1. Open the web UI.
2. Enter the value from `API_KEYS` into the API-key field. The default local
   development key is `dev-secret-key`.
3. Upload a `.pdf`, `.html`, `.htm`, `.txt`, or `.md` document.
4. Wait for the upload result and confirm it appears in the document list.
5. Ask a question in Chat.
6. Inspect the numbered sources and citations in the streamed answer.
7. Use the Agent view for calculator, Wikipedia, or RAG tool calls.
8. Delete test documents when finished.

Equivalent upload request:

```bash
curl -X POST http://localhost:8000/documents/upload \
  -H 'X-API-Key: dev-secret-key' \
  -F 'files=@data/raw/example.txt'
```

Equivalent document listing:

```bash
curl -H 'X-API-Key: dev-secret-key' \
  http://localhost:8000/documents
```

## 9. Important runtime expectations

- Protected routes require `X-API-Key`; missing or invalid keys return `401`.
- `/health`, `/metrics`, and `/` are public.
- Upload limits are controlled by `MAX_UPLOAD_MB` (default 25 MB per file).
- Uploaded files are sanitized and stored under `data/raw/`.
- Ingestion accepts only paths inside the repository `data/` directory.
- Chat streams sources before generated tokens, then citations, metrics, and
  a final `done` event.
- Answers are instructed to use only retrieved passages and cite factual claims.
- If context is insufficient, the expected response is:
  `I don't know based on the provided documents.`
- LLM retries use jittered backoff and may switch to the fallback provider.
- `QDRANT_URL=:memory:` loses vectors on restart; persistent Qdrant does not.
- Redis memory expires according to the configured TTL and may summarize older
  turns when the turn or token budget is exceeded.
- `RERANKER_BACKEND=lexical` requires no model download. `local` is more
  expensive and downloads a cross-encoder on first use.

## 10. Verification and maintenance commands

```bash
# Full offline tests
PYTHONPATH=src python -m pytest -q

# Syntax/import smoke check
PYTHONPATH=src python -m compileall -q src steps loadtest
PYTHONPATH=src python -c "from genai_platform.api.main import app; print(app.title)"

# Run individual demonstrations
PYTHONPATH=src python steps/step02_embedding_math.py
PYTHONPATH=src python steps/step03_qdrant_setup.py
PYTHONPATH=src python steps/step05_basic_rag.py
```

Run the tests after changing configuration, providers, ingestion, retrieval,
memory, API routes, or dependency manifests. Keep optional packages out of the
core installation unless the feature is being exercised.

## 11. Troubleshooting

### `ModuleNotFoundError: genai_platform`

Run commands from the repository root with `PYTHONPATH=src`, or install the
project in editable mode if packaging is added later.

### Gemini authentication or model errors

Check `GEMINI_API_KEY`, `GEMINI_MODEL`, and network access. The gateway may then
try `LLM_FALLBACK_PROVIDER`; configure a working fallback or expect degraded
chat behavior.

### Qdrant or Redis connection errors

Start the services with the core Compose profile, verify their ports, and
confirm `.env` URLs. For a local smoke run, use `QDRANT_URL=:memory:` and the
development fake-infrastructure behavior used by the tests.

### Slow first request

Local embedding and local reranker backends download model files on first use.
Use `EMBEDDING_BACKEND=hash` and `RERANKER_BACKEND=lexical` for a fast functional
check.

### Upload succeeds but retrieval is empty

Confirm the upload response reports chunks, check `/documents`, ensure the chat
and upload requests use the same running process, and avoid switching embedding
dimensions without using the dimension-suffixed collection behavior.

## 12. Handoff rules for future agents

Before editing:

1. Read this file and `README.md`.
2. Inspect `requirements-core.txt` and `requirements-optional.txt`.
3. Run the existing tests before making changes.
4. Preserve offline testability and graceful degradation.
5. Never commit `.env`, credentials, model caches, uploaded documents, or
   generated artifacts.

After editing:

1. Run the smallest relevant tests, then the full offline suite.
2. Run the import smoke check.
3. Update this guide or `README.md` when setup or behavior changes.
4. Check `git status` for accidental generated files.
5. Commit only source, tests, documentation, and intentional configuration.

