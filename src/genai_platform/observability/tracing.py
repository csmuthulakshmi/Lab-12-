"""Langfuse v3 tracing helpers that degrade gracefully to no-ops when unconfigured."""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable

from genai_platform.config import get_settings

logger = logging.getLogger(__name__)


def tracing_enabled() -> bool:
    """Return True if Langfuse credentials are configured."""
    settings = get_settings()
    return bool(settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY)


def get_langfuse_callback() -> Any:
    """Instantiate Langfuse CallbackHandler for LangChain if configured."""
    if not tracing_enabled():
        return None
    try:
        from langfuse.callback import CallbackHandler
        settings = get_settings()
        return CallbackHandler(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
    except Exception as e:
        logger.warning(f"Failed to initialize Langfuse callback: {e}")
        return None


def lc_config(session_id: str | None = None, tags: list[str] | None = None) -> dict[str, Any]:
    """Build LangChain config dictionary with Langfuse callback if enabled."""
    cb = get_langfuse_callback()
    if not cb:
        return {}
    config: dict[str, Any] = {"callbacks": [cb]}
    metadata: dict[str, Any] = {}
    if session_id:
        metadata["session_id"] = session_id
    if tags:
        metadata["tags"] = tags
    if metadata:
        config["metadata"] = metadata
    return config


def observe_fn(name: str | None = None) -> Callable:
    """Decorator to observe function execution in Langfuse (no-op if unconfigured)."""
    def decorator(fn: Callable) -> Callable:
        if not tracing_enabled():
            return fn
        try:
            from langfuse.decorators import observe
            return observe(name=name)(fn)
        except Exception:
            return fn
    return decorator


def flush() -> None:
    """Flush pending Langfuse events to server."""
    if not tracing_enabled():
        return
    try:
        from langfuse import Langfuse
        settings = get_settings()
        client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
        client.flush()
    except Exception as e:
        logger.warning(f"Error flushing Langfuse client: {e}")
