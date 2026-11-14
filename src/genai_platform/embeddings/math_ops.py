"""Vector and embedding mathematical operations.

Unit vector geometry:
For two unit-normalized vectors A and B with ||A||_2 = ||B||_2 = 1:
    ||A - B||^2 = (A - B) . (A - B)
                = ||A||^2 + ||B||^2 - 2 (A . B)
                = 1 + 1 - 2 cos(A, B)
                = 2 - 2 cos(A, B)

Therefore:
    ||A - B|| = sqrt(2 - 2 cos(A, B))

Maximizing cosine similarity is strictly equivalent to minimizing Euclidean
distance when vectors are L2-normalized.
"""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np


def dot_loop(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute dot product using a hand-written loop."""
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions do not match: {len(a)} vs {len(b)}")
    total = 0.0
    for x, y in zip(a, b):
        total += float(x) * float(y)
    return total


def norm(a: Sequence[float]) -> float:
    """Compute L2 norm using a hand-written loop."""
    total = 0.0
    for x in a:
        total += float(x) * float(x)
    return math.sqrt(total)


def cosine_loop(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity using hand-written loops."""
    norm_a = norm(a)
    norm_b = norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_loop(a, b) / (norm_a * norm_b)


def euclid_loop(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute Euclidean distance using a hand-written loop."""
    if len(a) != len(b):
        raise ValueError(f"Vector dimensions do not match: {len(a)} vs {len(b)}")
    total = 0.0
    for x, y in zip(a, b):
        diff = float(x) - float(y)
        total += diff * diff
    return math.sqrt(total)


def l2_normalize(vecs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize a 1D or 2D NumPy array along the last axis.
    Returns float32 array.
    """
    vecs = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    norms = np.maximum(norms, eps)
    return vecs / norms


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix between 2D arrays A (N x D) and B (M x D).
    Returns (N x M) array of cosine similarities.
    """
    a_norm = l2_normalize(a)
    b_norm = l2_normalize(b)
    return np.matmul(a_norm, b_norm.T)


def verify_against_numpy(a: Sequence[float], b: Sequence[float], tol: float = 1e-5) -> bool:
    """
    Verify hand-written cosine_loop and euclid_loop against NumPy implementations.
    Returns True if both match within tolerance.
    """
    loop_cos = cosine_loop(a, b)
    loop_euc = euclid_loop(a, b)

    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)

    norm_a = np.linalg.norm(arr_a)
    norm_b = np.linalg.norm(arr_b)
    np_cos = 0.0 if (norm_a == 0 or norm_b == 0) else float(np.dot(arr_a, arr_b) / (norm_a * norm_b))
    np_euc = float(np.linalg.norm(arr_a - arr_b))

    cos_match = abs(loop_cos - np_cos) < tol
    euc_match = abs(loop_euc - np_euc) < tol
    return cos_match and euc_match
