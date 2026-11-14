"""Main FastAPI application entry point with metrics middleware and static web UI."""

from __future__ import annotations

import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from genai_platform.api.routes import router
from genai_platform.observability.metrics import MetricsMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("genai_platform")

app = FastAPI(
    title="GenAI Platform Plus",
    version="0.1.0",
    description="Production-shaped RAG and Multi-Agent Platform",
)

# 1. Pure ASGI Metrics Middleware
app.add_middleware(MetricsMiddleware)

# 2. API Routes
app.include_router(router)

# 3. Mount Web UI (Static Files with index.html fallback)
WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="static")
