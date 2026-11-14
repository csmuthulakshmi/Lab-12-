#!/usr/bin/env python3
"""Step 07: Conversation Memory and Auto-Summarization Demo."""

import sys
from pathlib import Path
import fakeredis

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genai_platform.embeddings.embedder import HashEmbedder
from genai_platform.memory.redis_memory import RedisMemory


def mock_summarizer(prev_summary: str, turns: list[dict]) -> str:
    topics = [t["content"][:30] for t in turns]
    return f"User discussed topics: {'; '.join(topics)}"


def main():
    print("=== Step 07: Redis Conversation Memory & Compression Demo ===")

    fake_client = fakeredis.FakeRedis()
    embedder = HashEmbedder(dim=256)

    # Set tight max_turns=4 to demonstrate compression
    memory = RedisMemory(
        client=fake_client,
        summarizer=mock_summarizer,
        embedder=embedder,
        max_turns=4,
        token_budget=100,
        lock_timeout_s=0.2,
    )

    sid = "demo-session-123"
    print(f"Session ID: {sid}")

    print("\nAdding 3 turns...")
    memory.add_turn(sid, "user", "Hello, my favorite language is Python.")
    memory.add_turn(sid, "assistant", "That is great! Python is very versatile.")
    memory.add_turn(sid, "user", "I am building a RAG platform.")

    print(f"Active turns count: {len(memory.context(sid))}")
    print(f"Summary: '{memory.get_summary(sid)}'")

    print("\nAdding 2 more turns (exceeding max_turns=4)...")
    memory.add_turn(sid, "assistant", "RAG platforms require robust retrieval and memory.")
    memory.add_turn(sid, "user", "Can you explain how compression works?")

    print(f"\nAfter automatic compression:")
    print(f"Active turns count : {len(memory.context(sid))}")
    print(f"Generated Summary  : '{memory.get_summary(sid)}'")

    print("\nRendered Conversation Context:")
    print("---")
    print(memory.render(sid))
    print("---")

    print("\nStep 07 completed successfully!")


if __name__ == "__main__":
    main()
