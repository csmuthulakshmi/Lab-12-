"""Embeddings module exports."""

from genai_platform.embeddings.embedder import (
    Embedder,
    GeminiEmbedder,
    HashEmbedder,
    LocalEmbedder,
    get_embedder,
)
from genai_platform.embeddings.math_ops import (
    cosine_loop,
    cosine_matrix,
    dot_loop,
    euclid_loop,
    l2_normalize,
    norm,
    verify_against_numpy,
)

__all__ = [
    "Embedder",
    "GeminiEmbedder",
    "HashEmbedder",
    "LocalEmbedder",
    "get_embedder",
    "cosine_loop",
    "cosine_matrix",
    "dot_loop",
    "euclid_loop",
    "l2_normalize",
    "norm",
    "verify_against_numpy",
]
