#!/usr/bin/env python3
"""Step 12: Start API Server Helper."""

import os
import sys
from pathlib import Path
import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main():
    print("=== Step 12: Starting GenAI Platform Plus API Server ===")
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")

    print(f"Server URL: http://{host}:{port}")
    print(f"Web UI    : http://localhost:{port}/")
    print(f"Health    : http://localhost:{port}/health")
    print(f"Metrics   : http://localhost:{port}/metrics")
    print("\nPress Ctrl+C to stop.")

    uvicorn.run(
        "genai_platform.api.main:app",
        host=host,
        port=port,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
