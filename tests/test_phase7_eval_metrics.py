import pytest

from genai_platform.evaluation.ragas_eval import run_ragas_compare
from genai_platform.observability.metrics import (
    TelemetryHandler,
    render_metrics,
)
from genai_platform.observability.tracing import (
    flush,
    lc_config,
    observe_fn,
    tracing_enabled,
)


def test_render_metrics():
    body, content_type = render_metrics()
    assert isinstance(body, bytes)
    assert "text/plain" in content_type
    assert b"http_requests_total" in body or b"llm_tokens_total" in body


def test_telemetry_handler():
    handler = TelemetryHandler()
    assert handler.first_token_time is None

    handler.on_llm_new_token("first_token")
    assert handler.first_token_time is not None

    handler.record_usage(prompt=50, completion=20)
    summary = handler.summary()

    assert summary["prompt_tokens"] == 50
    assert summary["completion_tokens"] == 20
    assert summary["total_tokens"] == 70
    assert summary["ttft_s"] is not None
    assert summary["total_latency_s"] >= 0.0


def test_tracing_degrades_gracefully_without_keys():
    # Without keys, tracing_enabled is False
    assert tracing_enabled() is False

    # lc_config returns empty dict
    cfg = lc_config(session_id="123")
    assert cfg == {}

    # observe_fn is a pass-through decorator
    @observe_fn(name="test")
    def add(a, b):
        return a + b

    assert add(3, 4) == 7

    # flush does not raise
    flush()


def test_ragas_compare_delta_computation():
    class MockPipeline:
        def __init__(self, answer_text):
            self.answer_text = answer_text
        def answer(self, q, k=5):
            from unittest.mock import MagicMock
            res = MagicMock()
            res.answer = self.answer_text
            res.hits = []
            return res

    p_base = MockPipeline("Base answer")
    p_rerank = MockPipeline("Reranked answer")

    res = run_ragas_compare(
        pipeline_base=p_base,
        pipeline_rerank=p_rerank,
        cases=[{"question": "Q1", "ground_truth": "GT1"}],
        k=2,
    )

    assert "baseline_without_reranking" in res
    assert "with_reranking" in res
    assert "delta" in res
