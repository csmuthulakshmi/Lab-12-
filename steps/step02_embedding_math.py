#!/usr/bin/env python3
"""Step 02: Embedding Mathematics, Plain Loops, and NumPy Verification."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genai_platform.embeddings.math_ops import (
    cosine_loop,
    cosine_matrix,
    dot_loop,
    euclid_loop,
    l2_normalize,
    norm,
    verify_against_numpy,
)


def main():
    print("=== Step 02: Embedding Mathematics Demo ===")

    a = [0.2, 0.5, 0.8, -0.1]
    b = [0.1, 0.4, 0.9, 0.0]

    print(f"Vector A: {a}")
    print(f"Vector B: {b}")

    # Plain loop calculations
    dot_val = dot_loop(a, b)
    norm_a = norm(a)
    norm_b = norm(b)
    cos_val = cosine_loop(a, b)
    euc_val = euclid_loop(a, b)

    print(f"\nHand-written Loop Metrics:")
    print(f"  Dot Product        : {dot_val:.6f}")
    print(f"  Norm A             : {norm_a:.6f}")
    print(f"  Norm B             : {norm_b:.6f}")
    print(f"  Cosine Similarity  : {cos_val:.6f}")
    print(f"  Euclidean Distance : {euc_val:.6f}")

    # Verify against numpy
    matches = verify_against_numpy(a, b)
    print(f"\nVerification against NumPy (tolerance 1e-5): {matches}")
    assert matches, "Loop implementation differed from NumPy!"

    # Unit vector geometry demonstration: ||A - B||^2 == 2 - 2*cos(A, B)
    normed_a = a / np.linalg.norm(a)
    normed_b = b / np.linalg.norm(b)

    dist_sq = float(np.sum((normed_a - normed_b) ** 2))
    cos_geom = float(np.dot(normed_a, normed_b))
    geom_rhs = 2.0 - 2.0 * cos_geom

    print(f"\nUnit Vector Distance Geometry:")
    print(f"  ||A - B||^2        : {dist_sq:.6f}")
    print(f"  2 - 2*cos(A, B)    : {geom_rhs:.6f}")
    print(f"  Identity matches   : {abs(dist_sq - geom_rhs) < 1e-6}")

    # Matrix cosine
    arr_a = np.array([normed_a, normed_b], dtype=np.float32)
    matrix = cosine_matrix(arr_a, arr_a)
    print(f"\nCosine Matrix (2x2):\n{matrix}")

    print("\nStep 02 completed successfully!")


if __name__ == "__main__":
    main()
