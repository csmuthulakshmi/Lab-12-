import fakeredis
import pytest

from genai_platform.embeddings.embedder import HashEmbedder
from genai_platform.memory.redis_memory import RedisMemory


def sample_summarizer(prev_summary: str, turns: list[dict]) -> str:
    texts = [t["content"] for t in turns]
    return f"Summary of: {', '.join(texts)}"


def test_redis_memory_add_turns_and_ping():
    client = fakeredis.FakeRedis()
    mem = RedisMemory(client=client, summarizer=sample_summarizer, max_turns=6)

    assert mem.ping() is True

    mem.add_turn("sess1", "user", "Hello there")
    mem.add_turn("sess1", "assistant", "Hi, how can I help?")

    ctx = mem.context("sess1")
    assert len(ctx) == 2
    assert ctx[0]["role"] == "user"
    assert ctx[1]["role"] == "assistant"

    # Verify TTL set on keys
    ttl = client.ttl("chat:sess1:turns")
    assert ttl > 0


def test_redis_memory_compression_under_lock():
    client = fakeredis.FakeRedis()
    embedder = HashEmbedder(dim=256)

    # Set max_turns=4, keep = max(2, 4//2) = 2
    mem = RedisMemory(
        client=client,
        summarizer=sample_summarizer,
        embedder=embedder,
        max_turns=4,
        token_budget=1000,
        lock_timeout_s=0.2,
    )

    sid = "sess_compress"

    # Add 4 turns -> no compression yet
    for i in range(4):
        mem.add_turn(sid, "user" if i % 2 == 0 else "assistant", f"Turn content {i}")
    assert len(mem.context(sid)) == 4
    assert mem.get_summary(sid) == ""

    # Add 5th turn -> triggers compression
    mem.add_turn(sid, "user", "Turn content 4")

    # Compressed: 5 - 2 = 3 oldest turns folded into summary, leaving 2 active turns
    assert len(mem.context(sid)) == 2
    assert "Summary of: Turn content 0" in mem.get_summary(sid)

    # Render includes summary
    rendered = mem.render(sid)
    assert "[Summary of earlier conversation]" in rendered
    assert "User: Turn content 4" in rendered


def test_redis_memory_clear():
    client = fakeredis.FakeRedis()
    mem = RedisMemory(client=client, max_turns=5)
    sid = "sess_clear"

    mem.add_turn(sid, "user", "To be cleared")
    assert len(mem.context(sid)) == 1

    mem.clear(sid)
    assert len(mem.context(sid)) == 0
    assert mem.get_summary(sid) == ""
