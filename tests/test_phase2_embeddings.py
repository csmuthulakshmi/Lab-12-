import math
import numpy as np
import pytest

from genai_platform.embeddings.embedder import HashEmbedder
from genai_platform.embeddings.math_ops import (
    cosine_loop,
    cosine_matrix,
    dot_loop,
    euclid_loop,
    l2_normalize,
    norm,
    verify_against_numpy,
)
from genai_platform.ingestion.schemas import Chunk
from genai_platform.vectorstore.qdrant_store import QdrantStore


def test_math_ops_plain_loops_and_numpy_verification():
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]

    assert dot_loop(a, b) == 32.0
    assert norm([3.0, 4.0]) == 5.0
    assert euclid_loop([0.0, 0.0], [3.0, 4.0]) == 5.0

    # Verify against numpy
    assert verify_against_numpy(a, b)

    # Unit vector identity: ||A - B||^2 = 2 - 2 * cos(A, B)
    norm_a = [x / norm(a) for x in a]
    norm_b = [x / norm(b) for x in b]

    dist_sq = euclid_loop(norm_a, norm_b) ** 2
    cos_val = cosine_loop(norm_a, norm_b)
    rhs = 2.0 - 2.0 * cos_val
    assert abs(dist_sq - rhs) < 1e-6


def test_l2_normalize_and_cosine_matrix():
    mat = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    normed = l2_normalize(mat)
    # Check rows have unit norm
    assert np.allclose(np.linalg.norm(normed, axis=-1), [1.0, 1.0])

    sim_mat = cosine_matrix(normed, normed)
    assert sim_mat.shape == (2, 2)
    assert np.isclose(sim_mat[0, 0], 1.0)
    assert np.isclose(sim_mat[1, 1], 1.0)


def test_hash_embedder_deterministic_and_unit_norm():
    embedder = HashEmbedder(dim=256)
    v1 = embedder.embed_query("machine learning rag pipeline")
    v2 = embedder.embed_query("machine learning rag pipeline")

    assert len(v1) == 256
    assert v1 == v2  # strictly deterministic
    assert np.isclose(np.linalg.norm(v1), 1.0, atol=1e-5)

    docs = ["doc one", "doc two"]
    doc_vecs = embedder.embed_documents(docs)
    assert len(doc_vecs) == 2
    for dv in doc_vecs:
        assert np.isclose(np.linalg.norm(dv), 1.0, atol=1e-5)


def test_qdrant_store_memory_upsert_idempotent_and_stats():
    dim = 256
    embedder = HashEmbedder(dim=dim)
    store = QdrantStore(url=":memory:", collection=f"test_store_{dim}", dim=dim, recreate=True)

    chunks = [
        Chunk(
            text="First chunk about Python programming language.",
            source="python_docs.txt",
            doc_type="txt",
            page=1,
            index=0,
            strategy="recursive",
        ),
        Chunk(
            text="Second chunk about vector indexing in Qdrant.",
            source="qdrant_docs.pdf",
            doc_type="pdf",
            page=2,
            index=1,
            strategy="recursive",
        ),
    ]

    vecs = embedder.embed_documents([c.text for c in chunks])
    store.upsert(chunks, vecs)
    assert store.count() == 2

    # Idempotent re-upsert: point IDs based on uuid5(chunk.id)
    store.upsert(chunks, vecs)
    assert store.count() == 2

    # Search with source filter
    q_vec = embedder.embed_query("Python")
    hits_filtered = store.search(q_vec, k=2, source="python_docs.txt")
    assert len(hits_filtered) == 1
    assert hits_filtered[0].source == "python_docs.txt"
    assert "Python" in hits_filtered[0].text

    # Source stats
    stats = store.source_stats()
    assert len(stats) == 2
    sources = {s["source"] for s in stats}
    assert "python_docs.txt" in sources
    assert "qdrant_docs.pdf" in sources

    # Delete source
    deleted = store.delete_source("python_docs.txt")
    assert deleted == 1
    assert store.count() == 1
    assert "python_docs.txt" not in store.sources()
