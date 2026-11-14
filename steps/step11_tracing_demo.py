#!/usr/bin/env python3
"""Step 11: Langfuse Observability and Tracing Helpers Demo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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


@observe_fn(name="demo_task")
def sample_observed_operation(x: int) -> int:
    return x * 42


def main():
    print("=== Step 11: Observability & Tracing Demo ===")

    print(f"Tracing Enabled (Langfuse keys present): {tracing_enabled()}")

    config = lc_config(session_id="session-demo-trace", tags=["demo", "cli"])
    print(f"Generated LangChain Callback Config: {config}")

    print("\nExecuting observed function with graceful fallback...")
    res = sample_observed_operation(2)
    print(f"Result: {res}")
    flush()

    print("\nSimulating TelemetryHandler for TTFT and Token accounting:")
    telemetry = TelemetryHandler()
    telemetry.on_llm_new_token("First")
    telemetry.record_usage(prompt=120, completion=45)
    summary = telemetry.summary()
    print(f"Telemetry Summary: {summary}")

    print("\nExposing Prometheus metrics snapshot (first 10 lines):")
    raw_metrics, content_type = render_metrics()
    lines = raw_metrics.decode("utf-8").splitlines()[:10]
    for l in lines:
        print(f"  {l}")

    print("\nStep 11 completed successfully!")


if __name__ == "__main__":
    main()
