import io
from pathlib import Path
import fakeredis
import pytest
from fastapi.testclient import TestClient

from genai_platform.api.deps import (
    get_memory,
    get_rag_pipeline,
    get_redis_client,
)
from genai_platform.api.main import app
from genai_platform.config import DATA_RAW, ROOT
from genai_platform.embeddings.embedder import HashEmbedder
from genai_platform.llm.gateway import BaseBackend, Chunk, LLMGateway, LLMResult, Usage
from genai_platform.rag.pipeline import RagPipeline
from genai_platform.rag.retrievers import DenseRetriever
from genai_platform.vectorstore.qdrant_store import Hit, QdrantStore


class FakeApiLLM(BaseBackend):
    name = "fake_api_llm"
    model = "fake-model"

    def chat(self, messages, temperature=0.7, max_tokens=None, stop=None):
        return LLMResult(
            text="This is an answer grounded in the uploaded document [1].",
            usage=Usage(prompt_tokens=40, completion_tokens=15, total=55),
            provider=self.name,
            model=self.model,
            latency_s=0.05,
        )

    def stream(self, messages, temperature=0.7, max_tokens=None, stop=None):
        yield Chunk(text="This is an answer grounded in the uploaded document [1].")
        yield Chunk(text="", usage=Usage(prompt_tokens=40, completion_tokens=15, total=55))


@pytest.fixture
def client_with_fakes():
    # In-memory store and fake redis
    dim = 256
    embedder = HashEmbedder(dim=dim)
    store = QdrantStore(url=":memory:", collection=f"api_test_{dim}", dim=dim, recreate=True)
    retriever = DenseRetriever(store=store, embedder=embedder)
    gw = LLMGateway(primary=FakeApiLLM())
    pipeline = RagPipeline(retriever=retriever, llm=gw)

    fake_redis = fakeredis.FakeRedis()
    from genai_platform.memory.redis_memory import RedisMemory
    mem = RedisMemory(client=fake_redis)

    app.dependency_overrides[get_rag_pipeline] = lambda hybrid=True, rerank=True: pipeline
    app.dependency_overrides[get_memory] = lambda: mem
    app.dependency_overrides[get_redis_client] = lambda: fake_redis

    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()


def test_auth_rejection_401(client_with_fakes):
    # No key provided
    res = client_with_fakes.get("/documents")
    assert res.status_code == 401

    # Bad key provided
    res = client_with_fakes.get("/documents", headers={"X-API-Key": "wrong-key"})
    assert res.status_code == 401


def test_public_health_and_metrics(client_with_fakes):
    res_health = client_with_fakes.get("/health")
    assert res_health.status_code == 200
    data = res_health.json()
    assert "status" in data
    assert "components" in data

    res_metrics = client_with_fakes.get("/metrics")
    assert res_metrics.status_code == 200
    assert "text/plain" in res_metrics.headers["content-type"]


def test_ingest_path_traversal_protection(client_with_fakes):
    headers = {"X-API-Key": "test-key-123"}
    # Outside data/ directory
    payload = {"path": "/etc/passwd"}
    res = client_with_fakes.post("/ingest", json=payload, headers=headers)
    assert res.status_code == 400
    assert "Path traversal rejected" in res.json()["detail"]


def test_feature_a_document_lifecycle(client_with_fakes):
    headers = {"X-API-Key": "test-key-123"}

    # 1. Reject unsupported extension (415)
    bad_file = ("bad.exe", io.BytesIO(b"binary data"), "application/octet-stream")
    res_bad = client_with_fakes.post(
        "/documents/upload",
        files=[("files", bad_file)],
        headers=headers,
    )
    assert res_bad.status_code == 415

    # 2. Upload valid document
    doc_content = b"Knowledge base document describing deployment steps and cluster setup."
    good_file = ("deployment_guide.txt", io.BytesIO(doc_content), "text/plain")
    res_upload = client_with_fakes.post(
        "/documents/upload",
        files=[("files", good_file)],
        headers=headers,
    )
    assert res_upload.status_code == 200
    upload_stats = res_upload.json()
    assert len(upload_stats) == 1
    assert upload_stats[0]["status"] == "success"
    assert upload_stats[0]["chunks"] >= 1

    # 3. List documents
    res_list = client_with_fakes.get("/documents", headers=headers)
    assert res_list.status_code == 200
    docs = res_list.json()
    assert len(docs) >= 1
    target_source = docs[0]["source"]

    # 4. Chat stream verifies ordered events: sources -> metrics -> token -> citations -> done
    chat_payload = {
        "message": "What does the deployment guide describe?",
        "session_id": "api-sess-1",
        "k": 2,
    }
    res_chat = client_with_fakes.post("/chat/stream", json=chat_payload, headers=headers)
    assert res_chat.status_code == 200
    body = res_chat.text

    assert "event: sources" in body
    assert "event: token" in body
    assert "event: citations" in body
    assert "event: done" in body

    # Verify sources event precedes token events
    idx_sources = body.find("event: sources")
    idx_token = body.find("event: token")
    assert idx_sources < idx_token

    # 5. Delete document
    res_del = client_with_fakes.delete(f"/documents/{target_source}", headers=headers)
    assert res_del.status_code == 200
    del_data = res_del.json()
    assert del_data["status"] == "deleted"

    # 6. Delete unknown document returns 404
    res_del_404 = client_with_fakes.delete("/documents/non_existent_source.pdf", headers=headers)
    assert res_del_404.status_code == 404

    # 7. Delete session memory
    res_sess = client_with_fakes.delete("/sessions/api-sess-1", headers=headers)
    assert res_sess.status_code == 200
    assert res_sess.json()["cleared"] is True
