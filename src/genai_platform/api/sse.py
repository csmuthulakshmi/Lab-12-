"""Server-Sent Events (SSE) utilities and error guards."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Callable
from sse_starlette.sse import ServerSentEvent

from genai_platform.observability.metrics import SSE_ACTIVE_STREAMS

logger = logging.getLogger(__name__)


def sse(event: str, data: Any) -> ServerSentEvent:
    """Helper to format a ServerSentEvent."""
    if isinstance(data, (dict, list)):
        payload = json.dumps(data)
    else:
        payload = str(data)
    return ServerSentEvent(event=event, data=payload)


async def guarded_stream(generator: AsyncIterator[ServerSentEvent]) -> AsyncIterator[ServerSentEvent]:
    """
    Wrap an async SSE generator to catch any unhandled exceptions,
    emit an 'error' event, and guarantee a final 'done' event.
    Tracks active streams in Prometheus SSE_ACTIVE_STREAMS gauge.
    """
    SSE_ACTIVE_STREAMS.inc()
    done_emitted = False
    try:
        async for item in generator:
            if item.event == "done":
                done_emitted = True
            yield item
    except Exception as e:
        logger.error(f"Unhandled error in SSE stream: {e}", exc_info=True)
        yield sse("error", {"error": str(e)})
    finally:
        if not done_emitted:
            yield sse("done", {"status": "complete"})
        SSE_ACTIVE_STREAMS.dec()
