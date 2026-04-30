"""CentroidSelector-specific behavior (beyond the generic compliance suite).

Covers:
  - Centroid is actually the mean of seed embeddings.
  - Candidates that carry a stored embedding get reused (no re-embed).
  - Candidates without an embedding are computed via the embedder registry.
  - Diagnostics carry coherence fields.
  - config/from_config round-trip includes the centroid.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag_lib.embed import build_embedding_input
from rag_lib.embedders import placeholder_embed, register_embedder
from rag_lib.paper import Paper
from rag_lib.profile import Profile
from rag_lib.selectors.centroid import CentroidSelector


def _make_seed_paper(doi: str, title: str, abstract: str) -> Paper:
    p = Paper(doi=doi, openalex_id=None, title=title, abstract=abstract)
    p.embeddings["placeholder-v1"] = placeholder_embed(build_embedding_input(p))
    return p


def _profile() -> Profile:
    return Profile(
        name="x",
        papers=[
            _make_seed_paper("10.1/a", "A", "alpha"),
            _make_seed_paper("10.1/b", "B", "beta"),
            _make_seed_paper("10.1/c", "C", "gamma"),
        ],
        embedding_model="placeholder-v1",
    )


# ----------------------------------------------------------------------


def test_fit_sets_centroid_to_mean_of_seed_vectors():
    profile = _profile()
    expected = np.mean(
        [p.embeddings["placeholder-v1"] for p in profile.papers], axis=0
    )
    sel = CentroidSelector()
    sel.fit(profile)
    cfg = sel.config()
    assert cfg["centroid"] is not None
    np.testing.assert_allclose(cfg["centroid"], expected, rtol=1e-9)


def test_diagnostics_include_coherence_fields():
    sel = CentroidSelector()
    sel.fit(_profile())
    d = sel.diagnostics()
    assert d["status"] == "fit"
    assert "coherence_median" in d
    assert "coherence_iqr" in d
    assert "coherence_bimodal" in d
    assert d["n_seed"] == 3


def test_select_reuses_stored_candidate_embeddings():
    """Candidates pre-embedded should not trigger the embedder registry."""
    profile = _profile()
    candidates = [Paper(doi="10.9/x", openalex_id="WX", title="X",
                        abstract="alpha")]
    candidates[0].embeddings["placeholder-v1"] = placeholder_embed(
        build_embedding_input(candidates[0])
    )

    call_count = {"n": 0}

    def counting_embed(text: str) -> list[float]:
        call_count["n"] += 1
        return placeholder_embed(text)

    register_embedder("counting-v1", counting_embed)
    # Force selector to use 'counting-v1' key, but candidate carries a
    # different key. This means the selector WOULD embed — except here
    # we'll swap the candidate's key to match.
    candidates[0].embeddings["counting-v1"] = candidates[0].embeddings.pop("placeholder-v1")
    profile.embedding_model = "counting-v1"
    # Re-embed the seeds under the same key so fit has consistent space.
    for p in profile.papers:
        p.embeddings["counting-v1"] = counting_embed(build_embedding_input(p))
    baseline = call_count["n"]

    sel = CentroidSelector(embedding_model="counting-v1")
    sel.fit(profile)
    sel.select(candidates, profile)
    # Only fit-time seed encoding should have hit the embedder; select
    # should have reused the stored candidate embedding.
    assert call_count["n"] == baseline  # no additional calls during select


def test_select_computes_missing_candidate_embedding():
    """Candidates without a stored vector trigger the embedder."""
    profile = _profile()
    sel = CentroidSelector()
    sel.fit(profile)
    candidates = [Paper(doi="10.9/x", openalex_id="WX", title="X", abstract="alpha")]
    results = sel.select(candidates, profile)
    assert len(results) == 1
    score, _, breakdown = results[0]
    assert -1.0 <= score <= 1.0
    assert "score_raw" in breakdown
    assert "score_max_seed" in breakdown
    assert "score_median_seed" in breakdown
    assert "score_pct" in breakdown
    # Primary tuple slot is now the raw cosine, not the percentile.
    assert score == breakdown["score_raw"]


def test_select_before_fit_raises():
    sel = CentroidSelector()
    with pytest.raises(RuntimeError, match="before fit"):
        sel.select([Paper(doi="x", openalex_id=None, title="x")], _profile())


def test_config_round_trip_preserves_centroid_and_diagnostics():
    sel = CentroidSelector()
    sel.fit(_profile())
    cfg = sel.config()
    sel2 = CentroidSelector.from_config(cfg)
    np.testing.assert_allclose(sel2.config()["centroid"], cfg["centroid"])
    assert sel2.diagnostics()["status"] == "fit"


def test_threshold_filters_at_select_time():
    profile = _profile()
    sel = CentroidSelector()
    sel.fit(profile)
    candidates = [Paper(doi=f"10.9/{i}", openalex_id=f"W{i}",
                        title=f"cand-{i}", abstract=f"text {i}")
                  for i in range(5)]
    all_results = sel.select(candidates, profile)
    mid = all_results[len(all_results) // 2][0]
    filtered = sel.select(candidates, profile, threshold=mid)
    assert all(r[0] >= mid for r in filtered)
    assert len(filtered) <= len(all_results)
