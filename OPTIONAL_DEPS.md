# Optional Heavy Dependencies

The packages below are **not required** to run the platform. The app auto-detects
their absence and degrades gracefully — no crashes, no errors.

| Package | Used for | Degraded behaviour when absent |
|---------|----------|---------------------------------|
| `crewai` | `POST /crew/report` multi-agent report | Returns a plain stub report JSON |
| `ragas` | `POST /eval/ragas` evaluation suite | Returns mock baseline metrics (0.85–0.92) |
| `datasets` | Ragas `Dataset` wrapper | Same as above |

## Install (whenever you're ready)

```bash
# From the project root, once your venv is active:
uv pip install -r requirements-optional.txt --python .venv/bin/python

# — or plain pip —
pip install -r requirements-optional.txt
```

The packages will download in the background. **As soon as they land in your
venv the full features snap back in automatically** — no restart required for
one-off CLI usage; for the running API server, restart once after install:

```bash
uvicorn genai_platform.api.main:app --reload
```

## Why they're split out

| Package | Approx download |
|---------|-----------------|
| `crewai` | ~50 MB + transitive deps |
| `ragas` | ~30 MB |
| `datasets` (HuggingFace) | ~25 MB |

Total: **~500 MB** including transitive deps (torch model weights, etc.)

## Pytest behaviour

Tests that need these packages are automatically **skipped** when the
packages are absent (via `pytest.importorskip`). They turn green the moment
you install the optional deps — no changes to the test files needed.
