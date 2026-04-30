"""coherence() tests — median, IQR, bimodal detection."""

from __future__ import annotations

import numpy as np

from rag_lib.coherence import coherence


def test_single_vector_returns_nan_median():
    r = coherence([[1.0, 0.0]])
    assert r["n"] == 1
    assert np.isnan(r["median"])
    assert r["bimodal"] is False


def test_identical_vectors_yield_median_one():
    vecs = [[1.0, 0.0, 0.0]] * 5
    r = coherence(vecs)
    assert r["n"] == 5
    assert abs(r["median"] - 1.0) < 1e-9
    assert abs(r["iqr"]) < 1e-9
    assert r["bimodal"] is False


def test_orthogonal_vectors_yield_median_zero():
    vecs = np.eye(4).tolist()  # 4 orthonormal basis vectors
    r = coherence(vecs)
    assert abs(r["median"]) < 1e-9
    assert r["bimodal"] is False


def test_two_tight_clusters_flagged_bimodal():
    rng = np.random.default_rng(0)
    # Two clusters well-separated in 8-dim space.
    c1 = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=float)
    c2 = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    vecs = []
    for c in (c1, c2):
        for _ in range(8):
            v = c + 0.02 * rng.standard_normal(8)
            v = v / np.linalg.norm(v)
            vecs.append(v)
    r = coherence(vecs)
    assert r["bimodal"] is True


def test_uniform_cluster_not_flagged_bimodal():
    rng = np.random.default_rng(1)
    base = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    vecs = []
    for _ in range(12):
        v = base + 0.1 * rng.standard_normal(8)
        v = v / np.linalg.norm(v)
        vecs.append(v)
    r = coherence(vecs)
    assert r["bimodal"] is False


def test_rejects_1d_input():
    import pytest
    with pytest.raises(ValueError):
        coherence([1.0, 0.0, 1.0])
