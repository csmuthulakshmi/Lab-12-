"""Locust load test for GenAI Platform Plus: chat streaming and health endpoint."""

import json
import uuid
from locust import HttpUser, between, task


class GenAIPlatformUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        self.api_key = "test-key-123"
        self.session_id = f"loadtest-{uuid.uuid4().hex[:8]}"

    @task(5)
    def chat_stream(self):
        """Simulate user asking questions and reading SSE stream to completion."""
        payload = {
            "message": "What is the architecture of the vector search pipeline?",
            "session_id": self.session_id,
            "k": 4,
            "hybrid": True,
            "rerank": True,
        }
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
        }

        with self.client.post(
            "/chat/stream",
            json=payload,
            headers=headers,
            stream=True,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text}")
                return

            done_received = False
            for line in response.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8") if isinstance(line, bytes) else line
                if "event: done" in decoded:
                    done_received = True
                    break

            if done_received:
                response.success()
            else:
                response.failure("Stream terminated without 'event: done'")

    @task(2)
    def health_check(self):
        """Monitor platform health status."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
