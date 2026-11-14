"""RAG Pipeline: retrieve, generate answer, stream, format sources and validate citations."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Sequence

from genai_platform.llm.gateway import LLMGateway, get_gateway
from genai_platform.rag.prompts import (
    REFUSAL,
    SYSTEM_INSTRUCTION,
    format_context,
    validate_citations,
)
from genai_platform.vectorstore.qdrant_store import Hit

logger = logging.getLogger(__name__)


@dataclass
class RagAnswer:
    answer: str
    hits: list[Hit]
    citations: dict[str, Any]
    timings: dict[str, float] = field(default_factory=dict)

    def sources(self) -> list[dict[str, Any]]:
        """Return structured list of numbered source passages."""
        items: list[dict[str, Any]] = []
        for i, h in enumerate(self.hits, start=1):
            snippet = h.text[:200] + ("..." if len(h.text) > 200 else "")
            items.append({
                "n": i,
                "source": h.source,
                "page": h.page,
                "score": round(float(h.score), 4),
                "snippet": snippet,
                "cite": h.cite,
            })
        return items


class RagPipeline:
    """End-to-end RAG pipeline with distinct retrieval and generation phases."""

    def __init__(self, retriever: Any, llm: LLMGateway | None = None):
        self.retriever = retriever
        self.llm = llm or get_gateway()

    def retrieve(
        self,
        question: str,
        k: int = 4,
        source: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> list[Hit]:
        """Perform retrieval independently so sources can be emitted early."""
        return self.retriever.retrieve(question, k=k, source=source, config=config)

    def _build_messages(
        self,
        question: str,
        hits: Sequence[Hit],
        history: Sequence[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        context_str = format_context(hits)

        user_content_parts = []
        if history:
            history_text = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:])
            user_content_parts.append(f"Conversation history:\n{history_text}\n")

        user_content_parts.append(f"Context passages:\n{context_str}\n")
        user_content_parts.append(f"Question: {question}")

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": "\n".join(user_content_parts)},
        ]
        return messages

    def answer(
        self,
        question: str,
        hits: Sequence[Hit] | None = None,
        history: Sequence[dict[str, str]] | None = None,
        config: dict[str, Any] | None = None,
        k: int = 4,
        source: str | None = None,
    ) -> RagAnswer:
        timings: dict[str, float] = {}

        if hits is None:
            t0 = time.perf_counter()
            retrieved_hits = self.retrieve(question, k=k, source=source, config=config)
            timings["retrieval_s"] = time.perf_counter() - t0
        else:
            retrieved_hits = list(hits)

        messages = self._build_messages(question, retrieved_hits, history=history)

        t1 = time.perf_counter()
        llm_res = self.llm.chat(messages, temperature=0.2)
        timings["generation_s"] = time.perf_counter() - t1
        timings["total_s"] = timings.get("retrieval_s", 0.0) + timings["generation_s"]

        citations = validate_citations(llm_res.text, len(retrieved_hits))

        return RagAnswer(
            answer=llm_res.text,
            hits=retrieved_hits,
            citations=citations,
            timings=timings,
        )

    def stream_tokens(
        self,
        question: str,
        hits: Sequence[Hit],
        history: Sequence[dict[str, str]] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Synchronous token stream generator."""
        messages = self._build_messages(question, hits, history=history)
        for chunk in self.llm.stream(messages, temperature=0.2):
            if chunk.text:
                yield chunk.text

    async def astream(
        self,
        question: str,
        hits: Sequence[Hit],
        history: Sequence[dict[str, str]] | None = None,
        config: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Asynchronous token stream generator."""
        messages = self._build_messages(question, hits, history=history)
        # Run generator in thread to avoid blocking event loop
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def _worker():
            try:
                for chunk in self.llm.stream(messages, temperature=0.2):
                    if chunk.text:
                        loop.call_soon_threadsafe(queue.put_nowait, chunk.text)
            except Exception as e:
                logger.error(f"Error in stream worker: {e}")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=_worker, daemon=True).start()

        while True:
            token = await queue.get()
            if token is None:
                break
            yield token
