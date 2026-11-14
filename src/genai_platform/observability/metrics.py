"""Prometheus metrics, pure ASGI MetricsMiddleware, and TelemetryHandler."""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Registry
REGISTRY = CollectorRegistry(auto_describe=True)

# ---------------------------------------------------------------------------
# Metrics definitions
# ---------------------------------------------------------------------------

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests handled",
    ["route", "method", "status"],
    registry=REGISTRY,
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["route"],
    registry=REGISTRY,
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "Total LLM tokens consumed",
    ["kind"],  # prompt, completion, total
    registry=REGISTRY,
)

LLM_CALLS_TOTAL = Counter(
    "llm_calls_total",
    "Total LLM calls made",
    ["status"],  # success, error, fallback
    registry=REGISTRY,
)

LLM_CALL_DURATION = Histogram(
    "llm_call_duration_seconds",
    "LLM call latency in seconds",
    registry=REGISTRY,
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0],
)

LLM_TTFT = Histogram(
    "llm_time_to_first_token_seconds",
    "Time to first token in streaming responses",
    registry=REGISTRY,
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

RETRIEVAL_DURATION = Histogram(
    "retrieval_duration_seconds",
    "Vector retrieval latency in seconds",
    registry=REGISTRY,
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

AGENT_STEPS = Histogram(
    "agent_steps",
    "Number of reasoning steps taken by ReAct agent",
    registry=REGISTRY,
    buckets=[1, 2, 3, 4, 5, 6, 8, 10],
)

RAG_UNGROUNDED_ANSWERS = Counter(
    "rag_ungrounded_answers_total",
    "Total answers failing citation grounding check",
    registry=REGISTRY,
)

RERANK_DURATION = Histogram(
    "rerank_duration_seconds",
    "Duration of document reranking in seconds",
    registry=REGISTRY,
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# Gauge for active streams with multiprocess support
try:
    SSE_ACTIVE_STREAMS = Gauge(
        "sse_active_streams",
        "Number of active SSE streaming connections",
        multiprocess_mode="livesum",
        registry=REGISTRY,
    )
except TypeError:
    SSE_ACTIVE_STREAMS = Gauge(
        "sse_active_streams",
        "Number of active SSE streaming connections",
        registry=REGISTRY,
    )


# ---------------------------------------------------------------------------
# Metrics Renderer
# ---------------------------------------------------------------------------

def render_metrics() -> tuple[bytes, str]:
    """Generate Prometheus exposition format, supporting multiprocess mode."""
    multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if multiproc_dir and os.path.isdir(multiproc_dir):
        from prometheus_client import multiprocess
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry), CONTENT_TYPE_LATEST
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


# ---------------------------------------------------------------------------
# Pure ASGI MetricsMiddleware (BaseHTTPMiddleware breaks streaming)
# ---------------------------------------------------------------------------

class MetricsMiddleware:
    """Pure ASGI middleware to track HTTP request count and latency without breaking streaming."""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET")
        start_time = time.perf_counter()
        status_code = 200

        async def send_wrapper(message: dict[str, Any]) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 200)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration = time.perf_counter() - start_time
            # Normalize route for metrics
            route = path
            if path.startswith("/documents/") and len(path.split("/")) > 2:
                route = "/documents/{source}"
            elif path.startswith("/sessions/") and len(path.split("/")) > 2:
                route = "/sessions/{session_id}"

            HTTP_REQUESTS_TOTAL.labels(route=route, method=method, status=str(status_code)).inc()
            HTTP_REQUEST_DURATION.labels(route=route).observe(duration)


# ---------------------------------------------------------------------------
# LangChain Telemetry Handler
# ---------------------------------------------------------------------------

class TelemetryHandler:
    """Captures latency, TTFT, and token usage for requests."""

    def __init__(self):
        self.start_time = time.perf_counter()
        self.first_token_time: float | None = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def on_llm_new_token(self, token: str) -> None:
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
            ttft = self.first_token_time - self.start_time
            LLM_TTFT.observe(ttft)

    def record_usage(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += (prompt + completion)
        LLM_TOKENS_TOTAL.labels(kind="prompt").inc(prompt)
        LLM_TOKENS_TOTAL.labels(kind="completion").inc(completion)
        LLM_TOKENS_TOTAL.labels(kind="total").inc(prompt + completion)

    def summary(self) -> dict[str, Any]:
        total_latency = time.perf_counter() - self.start_time
        ttft = (self.first_token_time - self.start_time) if self.first_token_time else None
        return {
            "total_latency_s": round(total_latency, 3),
            "ttft_s": round(ttft, 3) if ttft is not None else None,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
