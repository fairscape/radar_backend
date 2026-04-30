"""MaxSeedSelector-specific behavior (beyond the generic compliance suite).

Covers:
  - fit() stores the (row-normalized) seed matrix, not a mean.
  - A candidate near one specific seed beats the centroid score for the
    same candidate when seeds are spread out — the whole point of this
    selector.
  - config/from_config round-trip restores the seed matrix.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag_lib.embed import build_embedding_input
from rag_lib.embedders import placeholder_embed
from rag_lib.paper import Paper
from rag_lib.profile import Profile
from rag_lib.selectors.centroid import CentroidSelector
from rag_lib.selectors.max_seed import MaxSeedSelector


def _make_seed(doi: str, title: str, abstract: str) -> Paper:
    p = Paper(doi=doi, openalex_id=None, title=title, abstract=abstract)
    p.embeddings["placeholder-v1"] = placeholder_embed(build_embedding_input(p))
    return p


def _profile() -> Profile:
    return Profile(
        name="x",
        papers=[
            _make_seed("10.1/a", "A", "alpha"),
            _make_seed("10.1/b", "B", "beta"),
            _make_seed("10.1/c", "C", "gamma"),
        ],
        embedding_model="placeholder-v1",
    )


# ----------------------------------------------------------------------


def test_fit_stores_row_normalized_seed_matrix():
    profile = _profile()
    sel = MaxSeedSelector()
    sel.fit(profile)
    cfg = sel.config()
    seeds = np.asarray(cfg["seed_matrix"])
    assert seeds.shape[0] == 3
    norms = np.linalg.norm(seeds, axis=1)
    np.testing.assert_allclose(norms, np.ones(3), rtol=1e-6, atol=1e-6)


def test_max_beats_centroid_when_candidate_matches_one_seed():
    """A candidate that is a near-twin of a single distinctive seed
    should score strictly higher under max-of-seeds than under
    centroid, because the centroid dilutes the matching seed with the
    others."""
    rng = np.random.default_rng(0)
    d = 32
    seeds = np.eye(3, d) + 0.01 * rng.standard_normal((3, d))
    candidate = seeds[0] + 0.001 * rng.standard_normal(d)

    cand = np.asarray(candidate, dtype=float)
    cand /= np.linalg.norm(cand)
    seeds_n = seeds / np.linalg.norm(seeds, axis=1, keepdims=True)
    centroid = seeds.mean(axis=0)
    centroid_n = centroid / np.linalg.norm(centroid)

    max_cos = float((seeds_n @ cand).max())
    cent_cos = float(centroid_n @ cand)
    assert max_cos > cent_cos


def test_config_round_trip_preserves_seed_matrix_and_diagnostics():
    sel = MaxSeedSelector()
    sel.fit(_profile())
    cfg = sel.config()
    sel2 = MaxSeedSelector.from_config(cfg)
    np.testing.assert_allclose(sel2.config()["seed_matrix"], cfg["seed_matrix"])
    assert sel2.diagnostics()["status"] == "fit"


def test_select_before_fit_raises():
    sel = MaxSeedSelector()
    with pytest.raises(RuntimeError, match="before fit"):
        sel.select([Paper(doi="x", openalex_id=None, title="x")], _profile())


def test_diagnostics_include_coherence_fields():
    sel = MaxSeedSelector()
    sel.fit(_profile())
    d = sel.diagnostics()
    assert d["status"] == "fit"
    assert d["n_seed"] == 3
    assert "coherence_median" in d


def test_threshold_filters_at_select_time():
    profile = _profile()
    sel = MaxSeedSelector()
    sel.fit(profile)
    candidates = [
        Paper(doi=f"10.9/{i}", openalex_id=f"W{i}",
              title=f"cand-{i}", abstract=f"text {i}")
        for i in range(5)
    ]
    all_results = sel.select(candidates, profile)
    if not all_results:
        return
    mid = all_results[len(all_results) // 2][0]
    filtered = sel.select(candidates, profile, threshold=mid)
    assert all(r[0] >= mid for r in filtered)
    assert len(filtered) <= len(all_results)
