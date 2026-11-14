#!/usr/bin/env python3
"""Step 01: LLM Gateway, Retry, and Automatic Fallback Demo."""

import sys
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genai_platform.llm.gateway import (
    BaseBackend,
    Chunk,
    LLMGateway,
    LLMResult,
    PermanentError,
    RateLimitError,
    RetryConfig,
    TransientError,
    Usage,
)


class FlakyPrimaryBackend(BaseBackend):
    name = "flaky_primary"
    model = "gemini-2.5-flash"

    def __init__(self):
        self.attempts = 0

    def chat(self, messages, temperature=0.7, max_tokens=None, stop=None):
        self.attempts += 1
        print(f"  [Primary] Attempt {self.attempts} triggered...")
        if self.attempts == 1:
            print("  [Primary] Simulated TransientError (503 Service Unavailable)")
            raise TransientError("Simulated 503")
        if self.attempts == 2:
            print("  [Primary] Simulated RateLimitError (429 Quota Exceeded, retry_after=0.2s)")
            raise RateLimitError("Simulated 429", retry_after=0.2)
        print("  [Primary] Final permanent failure to demonstrate fallback!")
        raise PermanentError("Simulated Quota Depleted")

    def stream(self, messages, temperature=0.7, max_tokens=None, stop=None):
        raise TransientError("Primary stream failed before token")


class StableFallbackBackend(BaseBackend):
    name = "stable_fallback"
    model = "llama3.2"

    def chat(self, messages, temperature=0.7, max_tokens=None, stop=None):
        print("  [Fallback] Answering request successfully via Ollama!")
        return LLMResult(
            text="Hello! This response comes from the fallback provider.",
            usage=Usage(prompt_tokens=12, completion_tokens=8, total=20),
            provider=self.name,
            model=self.model,
            latency_s=0.15,
        )

    def stream(self, messages, temperature=0.7, max_tokens=None, stop=None):
        yield Chunk(text="Hello ")
        yield Chunk(text="from ")
        yield Chunk(text="fallback!")


def main():
    print("=== Step 01: Gateway Retry & Fallback Demo ===")
    cfg = RetryConfig(max_retries=2, base_delay=0.1, cap_delay=0.5)
    primary = FlakyPrimaryBackend()
    fallback = StableFallbackBackend()

    gateway = LLMGateway(primary=primary, fallback=fallback, retry_config=cfg)

    print("\nSending chat message to gateway...")
    res = gateway.chat([{"role": "user", "content": "Hello world!"}])

    print("\nResult:")
    print(f"  Provider : {res.provider}")
    print(f"  Model    : {res.model}")
    print(f"  Answer   : {res.text}")
    print(f"  Gateway Stats: {dict(gateway.stats)}")
    print("\nStep 01 completed successfully!")


if __name__ == "__main__":
    main()
