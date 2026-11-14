"""FastAPI route handlers for health, metrics, ingestion, chat, agent, crew, and documents."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any, List, Literal, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Response,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from genai_platform.api.deps import (
    get_dense_retriever,
    get_hybrid_retriever,
    get_memory,
    get_rag_pipeline,
    get_redis_client,
    verify_api_key,
)
from genai_platform.api.sse import guarded_stream, sse
from genai_platform.config import DATA_RAW, ROOT, get_settings
from genai_platform.embeddings.embedder import get_embedder
from genai_platform.evaluation.ragas_eval import run_ragas, run_ragas_compare
from genai_platform.ingestion.pipeline import ingest_path
from genai_platform.observability.metrics import (
    HTTP_REQUESTS_TOTAL,
    RAG_UNGROUNDED_ANSWERS,
    TelemetryHandler,
    render_metrics,
)
from genai_platform.rag.prompts import validate_citations
from genai_platform.vectorstore.qdrant_store import get_store

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    path: str
    strategy: Literal["recursive", "semantic"] = "recursive"
    chunk_size: int = Field(default=800, ge=100, le=4000)
    overlap: int = Field(default=100, ge=0)
    recreate_collection: bool = False


class ChatRequest(BaseModel):
    message: str
    session_id: str
    k: int = 4
    source: Optional[str] = None
    hybrid: bool = True
    rerank: bool = True


class AgentRequest(BaseModel):
    message: str
    session_id: str


class CrewRequest(BaseModel):
    topic: str


class EvalRequest(BaseModel):
    k: int = 5
    compare: bool = False


# ---------------------------------------------------------------------------
# Public Health & Metrics Endpoints
# ---------------------------------------------------------------------------

@router.get("/health")
def health_check() -> dict[str, Any]:
    """Health check reporting qdrant, redis, llm_provider, and reranker status."""
    settings = get_settings()
    store = get_store()
    redis_mem = get_memory()

    qdrant_ok = False
    chunk_count = 0
    try:
        chunk_count = store.count()
        qdrant_ok = True
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")

    redis_ok = redis_mem.ping()
    overall = "ok" if (qdrant_ok and redis_ok) else "degraded"

    return {
        "status": overall,
        "components": {
            "qdrant": {
                "status": "ok" if qdrant_ok else "error",
                "collection": store.collection,
                "chunks": chunk_count,
            },
            "redis": {
                "status": "ok" if redis_ok else "error",
            },
            "llm_provider": settings.LLM_PROVIDER,
            "reranker_backend": settings.RERANKER_BACKEND,
        },
    }


@router.get("/metrics")
def metrics_endpoint() -> Response:
    """Expose Prometheus metrics."""
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)


# ---------------------------------------------------------------------------
# Ingestion Endpoint
# ---------------------------------------------------------------------------

@router.post("/ingest", dependencies=[Depends(verify_api_key)])
def ingest_documents(req: IngestRequest) -> dict[str, Any]:
    """Ingest documents from a path, validating against path traversal attacks."""
    resolved_path = Path(req.path).resolve()
    data_dir = (ROOT / "data").resolve()

    # Security fix: reject anything outside data/
    try:
        resolved_path.relative_to(data_dir)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Path traversal rejected: path '{req.path}' must be inside data/ directory",
        )

    if not resolved_path.exists():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Path does not exist: {req.path}",
        )

    store = get_store(recreate=req.recreate_collection)
    embedder = get_embedder()
    stats = ingest_path(
        path=resolved_path,
        store=store,
        embedder=embedder,
        strategy=req.strategy,
        chunk_size=req.chunk_size,
        overlap=req.overlap,
    )
    # Refresh BM25 index on hybrid retriever
    get_hybrid_retriever().refresh()
    return stats


# ---------------------------------------------------------------------------
# Chat Streaming Endpoint (SSE)
# ---------------------------------------------------------------------------

@router.post("/chat/stream", dependencies=[Depends(verify_api_key)])
async def chat_stream(
    req: ChatRequest,
    pipeline=Depends(get_rag_pipeline),
):
    """
    SSE chat endpoint with ordered events:
    sources -> metrics (retrieval) -> repeated token -> citations -> metrics (telemetry) -> done.
    """
    memory = get_memory()

    async def _event_generator():
        t0 = time.perf_counter()
        # 1. Retrieve first to emit sources before LLM generation
        hits = pipeline.retrieve(req.message, k=req.k, source=req.source)
        retrieval_latency = time.perf_counter() - t0

        formatted_sources = []
        for i, h in enumerate(hits, start=1):
            formatted_sources.append({
                "n": i,
                "source": h.source,
                "page": h.page,
                "score": round(float(h.score), 4),
                "snippet": h.text[:200] + ("..." if len(h.text) > 200 else ""),
                "cite": h.cite,
            })

        # Emit sources event
        yield sse("sources", formatted_sources)

        # Emit initial retrieval metrics
        yield sse("metrics", {"retrieval_latency_s": round(retrieval_latency, 3)})

        # Fetch conversation history for reference resolution
        history = memory.context(req.session_id)

        # 2. Generation stream
        telemetry = TelemetryHandler()
        tokens: list[str] = []

        for token in pipeline.stream_tokens(req.message, hits=hits, history=history):
            tokens.append(token)
            telemetry.on_llm_new_token(token)
            yield sse("token", {"token": token})

        full_answer = "".join(tokens)

        # 3. Validate citations
        citations = validate_citations(full_answer, len(hits))
        if not citations["grounded"]:
            RAG_UNGROUNDED_ANSWERS.inc()

        yield sse("citations", citations)

        # 4. Final latency and token telemetry metrics
        summary = telemetry.summary()
        summary["retrieval_latency_s"] = round(retrieval_latency, 3)
        yield sse("metrics", summary)

        # 5. Persist turns in memory
        memory.add_turn(req.session_id, "user", req.message)
        memory.add_turn(req.session_id, "assistant", full_answer)

        # 6. Stream completion done
        yield sse("done", {"status": "complete"})

    return EventSourceResponse(guarded_stream(_event_generator()))


# ---------------------------------------------------------------------------
# Agent Streaming Endpoint (SSE)
# ---------------------------------------------------------------------------

@router.post("/agent/stream", dependencies=[Depends(verify_api_key)])
async def agent_stream(req: AgentRequest):
    """
    ReAct agent stream emitting events: thought, action, observation, token, final, done.
    """
    from genai_platform.agents.react import ReActAgent
    from genai_platform.agents.tools import default_tools

    retriever = get_hybrid_retriever()
    tools = default_tools(retriever)
    agent = ReActAgent(tools=tools)
    memory = get_memory()

    async def _event_generator():
        final_answer = ""
        for event in agent.run_stream(req.message):
            ev_type = event.get("type", "message")
            if ev_type == "final":
                final_answer = event.get("answer", "")
                yield sse("final", event)
            elif ev_type == "token":
                yield sse("token", {"token": event.get("content", "")})
            else:
                yield sse(ev_type, event)

        if final_answer:
            memory.add_turn(req.session_id, "user", req.message)
            memory.add_turn(req.session_id, "assistant", final_answer)

        yield sse("done", {"status": "complete"})

    return EventSourceResponse(guarded_stream(_event_generator()))


# ---------------------------------------------------------------------------
# Crew Report Endpoint
# ---------------------------------------------------------------------------

@router.post("/crew/report", dependencies=[Depends(verify_api_key)])
def generate_crew_report(req: CrewRequest) -> dict[str, Any]:
    """Execute CrewAI Researcher -> Writer -> Reviewer report pipeline."""
    from genai_platform.agents.crew import run_report
    retriever = get_hybrid_retriever()
    return run_report(retriever=retriever, topic=req.topic)


# ---------------------------------------------------------------------------
# Ragas Evaluation Endpoint
# ---------------------------------------------------------------------------

@router.post("/eval/ragas", dependencies=[Depends(verify_api_key)])
def run_evaluation(req: EvalRequest) -> dict[str, Any]:
    """Execute RAGAS evaluation or comparative reranking assessment."""
    if req.compare:
        p_base = get_rag_pipeline(hybrid=True, rerank=False)
        p_rerank = get_rag_pipeline(hybrid=True, rerank=True)
        return run_ragas_compare(p_base, p_rerank, k=req.k)
    else:
        pipeline = get_rag_pipeline(hybrid=True, rerank=True)
        return run_ragas(pipeline, k=req.k)


# ---------------------------------------------------------------------------
# NEW FEATURE A - Document Management Endpoints
# ---------------------------------------------------------------------------

ALLOWED_EXTENSIONS = {".pdf", ".html", ".htm", ".txt", ".md"}


def _pipeline_store_and_embedder(pipeline=None):
    pipeline = pipeline or get_rag_pipeline()
    retriever = pipeline.retriever
    store = getattr(retriever, "store", None)
    embedder = getattr(retriever, "embedder", None)
    if store is None or embedder is None:
        store, embedder = get_store(), get_embedder()
    return store, embedder


@router.post("/documents/upload", dependencies=[Depends(verify_api_key)])
async def upload_documents(
    files: List[UploadFile] = File(...),
    pipeline=Depends(get_rag_pipeline),
) -> list[dict[str, Any]]:
    """
    Multipart document upload: validates extension and file size, sanitizes filename,
    saves to data/raw/, immediately ingests into vector store, and refreshes index.
    """
    settings = get_settings()
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    results = []

    # 1. Validation pass
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type '{ext}' for file '{f.filename}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )

    store, embedder = _pipeline_store_and_embedder(pipeline)

    # 2. Save and ingest
    for f in files:
        raw_name = Path(f.filename or "uploaded_doc.txt").name
        # Sanitise: no path separators or ..
        clean_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", raw_name)
        dest_path = DATA_RAW / clean_name

        content = await f.read()
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File '{f.filename}' exceeds maximum upload size of {settings.MAX_UPLOAD_MB}MB",
            )

        with open(dest_path, "wb") as out_file:
            out_file.write(content)

        try:
            stats = ingest_path(dest_path, store=store, embedder=embedder)
            results.append({
                "filename": clean_name,
                "chunks": stats.get("total_chunks", 0),
                "status": "success",
            })
        except Exception as e:
            logger.error(f"Failed to ingest uploaded file {clean_name}: {e}")
            results.append({
                "filename": clean_name,
                "chunks": 0,
                "status": "error",
                "error": str(e),
            })

    # Refresh BM25 search index with new chunks
    get_hybrid_retriever().refresh()
    return results


@router.get("/documents", dependencies=[Depends(verify_api_key)])
def list_documents(pipeline=Depends(get_rag_pipeline)) -> list[dict[str, Any]]:
    """Return list of distinct ingested documents and source statistics."""
    store, _ = _pipeline_store_and_embedder(pipeline)
    return store.source_stats()


@router.delete("/documents/{source:path}", dependencies=[Depends(verify_api_key)])
def delete_document(source: str, pipeline=Depends(get_rag_pipeline)) -> dict[str, Any]:
    """Delete vectors for source and remove saved file from data/raw/."""
    store, _ = _pipeline_store_and_embedder(pipeline)
    deleted_chunks = store.delete_source(source)

    # Check and remove file on disk if exists
    file_path = Path(source)
    if not file_path.is_absolute():
        file_path = DATA_RAW / file_path.name
    file_removed = False
    if file_path.exists() and file_path.is_file():
        try:
            file_path.unlink()
            file_removed = True
        except Exception as e:
            logger.warning(f"Could not delete physical file {file_path}: {e}")

    if deleted_chunks == 0 and not file_removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{source}' not found",
        )

    # Refresh BM25 index after deletion
    get_hybrid_retriever().refresh()
    return {
        "source": source,
        "deleted_chunks": deleted_chunks,
        "file_deleted": file_removed,
        "status": "deleted",
    }


@router.delete("/sessions/{session_id}", dependencies=[Depends(verify_api_key)])
def clear_session(session_id: str) -> dict[str, Any]:
    """Clear conversation history and summary memory for a specific session."""
    memory = get_memory()
    memory.clear(session_id)
    return {"session_id": session_id, "cleared": True}
