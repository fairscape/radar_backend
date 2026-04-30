"""Embedder tests. Phase 1A has just the hash-based placeholder; Phase 1B
will add specter2_embed and its own test file."""

from __future__ import annotations

import math

from rag_lib.embedders import placeholder_embed


def test_deterministic():
    assert placeholder_embed("some text") == placeholder_embed("some text")


def test_different_input_different_vector():
    assert placeholder_embed("alpha") != placeholder_embed("beta")


def test_unit_norm():
    v = placeholder_embed("hrv neonatal sepsis")
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-6


def test_respects_dim():
    assert len(placeholder_embed("x", dim=64)) == 64
    assert len(placeholder_embed("x", dim=256)) == 256
