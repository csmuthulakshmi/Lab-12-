import time
from collections import Counter
import pytest
from unittest.mock import MagicMock

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
    backoff_delay,
    call_with_retry,
    estimate_tokens,
    parse_retry_after,
    stream_with_retry,
)
from genai_platform.llm.lc_model import GatewayChatModel
from langchain_core.messages import HumanMessage, SystemMessage


# ---------------------------------------------------------------------------
# Test Retry and Backoff Maths
# ---------------------------------------------------------------------------

def test_backoff_delay_bounds():
    for attempt in range(5):
        delay = backoff_delay(attempt, base=1.0, cap=30.0)
        max_possible = min(30.0, 1.0 * (2 ** attempt))
        assert 0.0 <= delay <= max_possible

    # Cap respected
    for _ in range(20):
        delay = backoff_delay(10, base=1.0, cap=15.0)
        assert 0.0 <= delay <= 15.0


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hi") >= 1
    assert estimate_tokens("a" * 40) == 10


def test_parse_retry_after():
    # Header numeric seconds
    assert parse_retry_after({"Retry-After": "12"}) == 12.0
    assert parse_retry_after({"retry-after": "5.5"}) == 5.5

    # Header x-ratelimit-reset
    assert parse_retry_after({"x-ratelimit-reset-requests": "4.2"}) == 4.2

    # Gemini string hints
    assert parse_retry_after(None, 'Quota exceeded. Please retry in 17.5s') == 17.5
    assert parse_retry_after(None, '{"error": {"details": [{"retryDelay": "20s"}]}}') == 20.0
    assert parse_retry_after(None, 'Please retry after 3s') == 3.0

    # No hint
    assert parse_retry_after(None, 'Generic bad request') is None


# ---------------------------------------------------------------------------
# Test Call With Retry & Stream With Retry
# ---------------------------------------------------------------------------

def test_call_with_retry_permanent_fails_fast():
    stats = Counter()
    cfg = RetryConfig(max_retries=3, base_delay=0.01, cap_delay=0.05)
    mock_fn = MagicMock(side_effect=PermanentError("Invalid API key"))

    with pytest.raises(PermanentError):
        call_with_retry(mock_fn, cfg, stats, "test")

    assert mock_fn.call_count == 1
    assert stats["test_permanent_error"] == 1
    assert stats["test_retry"] == 0


def test_call_with_retry_recovers():
    stats = Counter()
    cfg = RetryConfig(max_retries=3, base_delay=0.01, cap_delay=0.02)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise TransientError("network hiccup")
        return "success"

    result = call_with_retry(flaky, cfg, stats, "test")
    assert result == "success"
    assert len(calls) == 3
    assert stats["test_transient"] == 2
    assert stats["test_retry"] == 2


def test_call_with_retry_exhaustion():
    stats = Counter()
    cfg = RetryConfig(max_retries=2, base_delay=0.01, cap_delay=0.02)
    mock_fn = MagicMock(side_effect=RateLimitError("quota", retry_after=0.01))

    with pytest.raises(RateLimitError):
        call_with_retry(mock_fn, cfg, stats, "test")

    # attempt 0, 1, 2 = 3 calls
    assert mock_fn.call_count == 3
    assert stats["test_ratelimit"] == 3


def test_stream_no_retry_after_first_token():
    stats = Counter()
    cfg = RetryConfig(max_retries=3, base_delay=0.01, cap_delay=0.02)

    def failing_stream():
        yield Chunk(text="Hello")
        raise TransientError("network dropped mid-stream")

    stream_gen = stream_with_retry(failing_stream, cfg, stats, "stream_test")

    # First token yielded fine
    first = next(stream_gen)
    assert first.text == "Hello"

    # Mid-stream error must NOT be retried, but raised directly
    with pytest.raises(TransientError):
        list(stream_gen)

    assert stats["stream_test_retry"] == 0


def test_stream_retries_before_first_token():
    stats = Counter()
    cfg = RetryConfig(max_retries=2, base_delay=0.01, cap_delay=0.02)
    attempts = [0]

    def flaky_stream():
        attempts[0] += 1
        if attempts[0] == 1:
            raise TransientError("failed before token")
        yield Chunk(text="Recovered")

    chunks = list(stream_with_retry(flaky_stream, cfg, stats, "stream_test"))
    assert len(chunks) == 1
    assert chunks[0].text == "Recovered"
    assert attempts[0] == 2
    assert stats["stream_test_retry"] == 1


# ---------------------------------------------------------------------------
# Fake Backends for Gateway Tests
# ---------------------------------------------------------------------------

class FakeBackend(BaseBackend):
    def __init__(self, name: str, should_fail: bool = False, fail_type: str = "transient"):
        self.name = name
        self.model = f"{name}-model"
        self.should_fail = should_fail
        self.fail_type = fail_type
        self.chat_calls = 0

    def chat(self, messages, temperature=0.7, max_tokens=None, stop=None):
        self.chat_calls += 1
        if self.should_fail:
            if self.fail_type == "permanent":
                raise PermanentError(f"{self.name} permanent failure")
            raise TransientError(f"{self.name} transient failure")
        return LLMResult(
            text=f"Response from {self.name}",
            usage=Usage(10, 5, 15),
            provider=self.name,
            model=self.model,
            latency_s=0.05,
        )

    def stream(self, messages, temperature=0.7, max_tokens=None, stop=None):
        if self.should_fail:
            if self.fail_type == "permanent":
                raise PermanentError(f"{self.name} stream failure")
            raise TransientError(f"{self.name} stream transient")
        yield Chunk(text=f"Response from {self.name}")
        yield Chunk(text="", usage=Usage(10, 5, 15))


def test_gateway_fallback_switch():
    primary = FakeBackend("primary", should_fail=True, fail_type="permanent")
    fallback = FakeBackend("fallback", should_fail=False)

    gw = LLMGateway(primary=primary, fallback=fallback)
    res = gw.chat([{"role": "user", "content": "hi"}])

    assert res.text == "Response from fallback"
    assert res.provider == "fallback"
    assert gw.stats["fallback_used"] == 1
    assert fallback.chat_calls == 1


def test_gateway_stream_fallback():
    primary = FakeBackend("primary", should_fail=True, fail_type="permanent")
    fallback = FakeBackend("fallback", should_fail=False)

    gw = LLMGateway(primary=primary, fallback=fallback)
    chunks = list(gw.stream([{"role": "user", "content": "hi"}]))

    assert any("Response from fallback" in c.text for c in chunks)
    assert gw.stats["fallback_used"] == 1


def test_gateway_no_fallback_raises():
    primary = FakeBackend("primary", should_fail=True, fail_type="permanent")
    gw = LLMGateway(primary=primary, fallback=None)

    with pytest.raises(PermanentError):
        gw.chat([{"role": "user", "content": "hi"}])


def test_langchain_gateway_chat_model():
    primary = FakeBackend("primary", should_fail=False)
    gw = LLMGateway(primary=primary)
    model = GatewayChatModel(gateway=gw)

    # Invocation
    response = model.invoke([
        SystemMessage(content="You are helpful."),
        HumanMessage(content="Hello"),
    ])
    assert response.content == "Response from primary"
    assert response.usage_metadata is not None
    assert response.usage_metadata["total_tokens"] == 15

    # Streaming
    stream_chunks = list(model.stream([HumanMessage(content="Hello")]))
    text = "".join(c.content for c in stream_chunks)
    assert text == "Response from primary"
