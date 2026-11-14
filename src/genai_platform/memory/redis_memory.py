"""Redis conversation memory with session TTL, lock-protected compression, and summarization."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Sequence

from genai_platform.embeddings.embedder import Embedder
from genai_platform.llm.gateway import LLMGateway, estimate_tokens
from genai_platform.rag.prompts import SUMMARY_PROMPT

logger = logging.getLogger(__name__)


def make_summarizer(llm: LLMGateway) -> Callable[[str, list[dict[str, str]]], str]:
    """Factory creating an LLM summarizer function from the platform LLM Gateway."""

    def _summarize(previous_summary: str, new_turns: list[dict[str, str]]) -> str:
        lines = "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in new_turns)
        prompt = SUMMARY_PROMPT.format(
            summary=previous_summary or "(None yet)",
            new_lines=lines,
        )
        messages = [{"role": "user", "content": prompt}]
        res = llm.chat(messages, temperature=0.3, max_tokens=200)
        return res.text.strip()

    return _summarize


class RedisMemory:
    """
    Session-based conversation memory backed by Redis.
    Compresses older turns into a dense summary when turns exceed max_turns
    or token budget exceeds threshold under a distributed lock.
    """

    def __init__(
        self,
        client: Any,
        summarizer: Callable[[str, list[dict[str, str]]], str] | None = None,
        embedder: Embedder | None = None,
        max_turns: int = 10,
        token_budget: int = 1500,
        ttl_s: int = 86400,
        lock_timeout_s: float = 5.0,
    ):
        self.client = client
        self.summarizer = summarizer
        self.embedder = embedder
        self.max_turns = max_turns
        self.token_budget = token_budget
        self.ttl_s = ttl_s
        self.lock_timeout_s = lock_timeout_s

    def _turns_key(self, sid: str) -> str:
        return f"chat:{sid}:turns"

    def _summary_key(self, sid: str) -> str:
        return f"chat:{sid}:summary"

    def _summary_vec_key(self, sid: str) -> str:
        return f"chat:{sid}:summary_vec"

    def _lock_key(self, sid: str) -> str:
        return f"chat:{sid}:lock"

    def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            return bool(self.client.ping())
        except Exception as e:
            logger.warning(f"Redis ping failed: {e}")
            return False

    def add_turn(self, sid: str, role: str, content: str) -> None:
        """Add a conversation turn and trigger background compression if needed."""
        turn_data = {
            "role": role,
            "content": content,
            "ts": time.time(),
        }
        turns_key = self._turns_key(sid)
        raw_json = json.dumps(turn_data)

        # RPUSH and refresh TTL
        pipe = self.client.pipeline()
        pipe.rpush(turns_key, raw_json)
        pipe.expire(turns_key, self.ttl_s)
        pipe.execute()

        # Check for compression
        self._maybe_compress(sid)

    def _maybe_compress(self, sid: str) -> None:
        """Compress older turns into summary if limits exceeded."""
        if not self.summarizer:
            return

        turns_key = self._turns_key(sid)
        raw_turns = self.client.lrange(turns_key, 0, -1)
        if not raw_turns:
            return

        turns: list[dict[str, Any]] = [json.loads(t) for t in raw_turns]
        total_tokens = sum(estimate_tokens(t.get("content", "")) for t in turns)

        if len(turns) <= self.max_turns and total_tokens <= self.token_budget:
            return

        # Acquire lock to ensure only one compressor per session
        lock_key = self._lock_key(sid)
        lock_ms = int(self.lock_timeout_s * 1000)
        acquired = self.client.set(lock_key, "1", nx=True, px=lock_ms)
        if not acquired:
            logger.debug(f"Compression lock held for session {sid}; skipping")
            return

        try:
            # Re-fetch under lock
            raw_turns = self.client.lrange(turns_key, 0, -1)
            turns = [json.loads(t) for t in raw_turns]
            if len(turns) <= self.max_turns and total_tokens <= self.token_budget:
                return

            keep_count = max(2, self.max_turns // 2)
            if len(turns) <= keep_count:
                return

            compress_count = len(turns) - keep_count
            to_compress = turns[:compress_count]

            summary_key = self._summary_key(sid)
            prev_summary_raw = self.client.get(summary_key)
            prev_summary = prev_summary_raw.decode("utf-8") if isinstance(prev_summary_raw, bytes) else (prev_summary_raw or "")

            try:
                new_summary = self.summarizer(prev_summary, to_compress)
            except Exception as e:
                logger.warning(f"Summarizer failed for session {sid} ({e}); keeping raw turns.")
                return

            # Compute summary embedding if embedder available
            summary_vec: list[float] | None = None
            if self.embedder and new_summary:
                try:
                    summary_vec = self.embedder.embed_query(new_summary)
                except Exception as e:
                    logger.warning(f"Summary embedding failed: {e}")

            # Atomic update via pipeline: trim turns, update summary and TTLs
            pipe = self.client.pipeline()
            # LTRIM to keep from compress_count to end
            pipe.ltrim(turns_key, compress_count, -1)
            pipe.expire(turns_key, self.ttl_s)
            pipe.set(summary_key, new_summary, ex=self.ttl_s)
            if summary_vec:
                vec_key = self._summary_vec_key(sid)
                pipe.set(vec_key, json.dumps(summary_vec), ex=self.ttl_s)
            pipe.execute()

        except Exception as e:
            logger.warning(f"Error during conversation compression for session {sid}: {e}")
        finally:
            try:
                self.client.delete(lock_key)
            except Exception:
                pass

    def context(self, sid: str) -> list[dict[str, str]]:
        """Return list of active conversation turns."""
        turns_key = self._turns_key(sid)
        raw_turns = self.client.lrange(turns_key, 0, -1)
        if not raw_turns:
            return []
        result = []
        for r in raw_turns:
            t = json.loads(r)
            result.append({"role": t["role"], "content": t["content"]})
        return result

    def get_summary(self, sid: str) -> str:
        """Get the current summary text, if any."""
        summary_key = self._summary_key(sid)
        val = self.client.get(summary_key)
        if not val:
            return ""
        return val.decode("utf-8") if isinstance(val, bytes) else str(val)

    def render(self, sid: str) -> str:
        """Render the full conversation context including summary and turns."""
        summary = self.get_summary(sid)
        turns = self.context(sid)

        parts = []
        if summary:
            parts.append(f"[Summary of earlier conversation]\n{summary}\n")

        for t in turns:
            parts.append(f"{t['role'].capitalize()}: {t['content']}")

        return "\n".join(parts)

    def clear(self, sid: str) -> None:
        """Clear all session data."""
        self.client.delete(
            self._turns_key(sid),
            self._summary_key(sid),
            self._summary_vec_key(sid),
            self._lock_key(sid),
        )
