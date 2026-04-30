"""Phase 3 score-tuning behavior on top of CentroidSelector.

Beyond the protocol-compliance test, these assert the contract:
  - select() returns 3-tuples whose primary score is the raw cosine to
    centroid; breakdown carries score_raw, score_max_seed,
    score_median_seed, and score_pct.
  - score_raw is unmapped cosine (can be negative on adversarial inputs).
  - score_pct is still attached for the daily-feed UI (uniform by rank
    across a batch).
  - max-seed spread exceeds centroid spread on a synthetic seed set
    where one seed sits far from the others (the original
    score-compression hypothesis).
  - config()/from_config() persists the seed_matrix so reloaded
    selectors can still emit score_max_seed.
"""

from __future__ import annotations

import numpy as np

from rag_lib.paper import Paper
from rag_lib.profile import Profile
from rag_lib.selectors.centroid import CentroidSelector


def _seed_paper(oa_id: str, vec: np.ndarray) -> Paper:
    p = Paper(doi=None, openalex_id=oa_id, title=oa_id, abstract="abs",
              source="openalex_gatherer")
    p.embeddings["test"] = vec.tolist()
    return p


def _candidate(oa_id: str, vec: np.ndarray) -> Paper:
    p = Paper(doi=None, openalex_id=oa_id, title=oa_id, abstract="abs",
              source="openalex_gatherer")
    p.embeddings["test"] = vec.tolist()
    return p


def _profile(seeds: list[np.ndarray]) -> Profile:
    return Profile(
        name="p",
        papers=[_seed_paper(f"S{i}", v) for i, v in enumerate(seeds)],
        embedding_model="test",
    )


# ----------------------------------------------------------------------


def test_breakdown_has_expected_keys():
    seeds = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    profile = _profile(seeds)
    sel = CentroidSelector(embedding_model="test")
    sel.fit(profile)
    cands = [_candidate("W1", np.array([1.0, 0.5])),
             _candidate("W2", np.array([0.0, 1.0]))]
    out = sel.select(cands, profile)
    for entry in out:
        breakdown = entry[2]
        assert {
            "score_raw", "score_max_seed", "score_median_seed", "score_pct",
        } <= breakdown.keys()
        # Primary tuple slot mirrors the raw cosine, not the percentile.
        assert entry[0] == breakdown["score_raw"]


def test_score_pct_is_uniform_by_rank():
    seeds = [np.array([1.0, 0.0])]
    profile = _profile(seeds)
    sel = CentroidSelector(embedding_model="test")
    sel.fit(profile)
    n = 5
    cands = [_candidate(f"W{i}", np.array([1.0 - 0.1 * i, 0.05 * i])) for i in range(n)]
    out = sel.select(cands, profile)
    pcts = [e[2]["score_pct"] for e in out]
    assert pcts[0] == 1.0
    assert pcts[-1] == 0.0
    # Strictly decreasing across the batch.
    assert all(pcts[i] > pcts[i + 1] for i in range(n - 1))


def test_score_raw_is_unmapped_cosine_can_be_negative():
    """Drop-the-(c+1)/2 contract: opposing vectors get negative raw scores."""
    seeds = [np.array([1.0, 0.0])]
    profile = _profile(seeds)
    sel = CentroidSelector(embedding_model="test")
    sel.fit(profile)
    cands = [_candidate("near", np.array([1.0, 0.0])),
             _candidate("far", np.array([-1.0, 0.0]))]
    out = sel.select(cands, profile)
    raws = {e[1].openalex_id: e[2]["score_raw"] for e in out}
    assert raws["near"] > 0.99
    assert raws["far"] < -0.99


def test_max_seed_spread_exceeds_centroid_spread_on_distinctive_seed():
    """A candidate near a single distinctive seed is dragged down by the
    centroid average but stays high under max-seed; the breakdown
    captures both signals so consumers can choose."""
    rng = np.random.default_rng(0)
    d = 32
    seeds = [
        np.eye(1, d).ravel(),                       # distinctive direction 1
        np.eye(1, d, k=1).ravel(),                  # near-orthogonal direction 2
        np.eye(1, d, k=2).ravel(),                  # near-orthogonal direction 3
    ]
    profile = _profile(seeds)
    sel = CentroidSelector(embedding_model="test")
    sel.fit(profile)

    # A candidate that's a near-twin of seed 0 only.
    cand = _candidate("W1", seeds[0] + 0.01 * rng.standard_normal(d))
    out = sel.select([cand], profile)
    breakdown = out[0][2]
    assert breakdown["score_max_seed"] > breakdown["score_raw"]


def test_config_round_trip_carries_seed_matrix():
    seeds = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    profile = _profile(seeds)
    sel = CentroidSelector(embedding_model="test")
    sel.fit(profile)
    cfg = sel.config()
    assert cfg["seed_matrix"] is not None
    assert len(cfg["seed_matrix"]) == 2
    sel2 = CentroidSelector.from_config(cfg)
    out = sel2.select([_candidate("W1", np.array([1.0, 0.5]))], profile)
    # Re-loaded selector still emits score_max_seed.
    assert "score_max_seed" in out[0][2]


def test_threshold_filters_against_raw_cosine():
    """Threshold semantics: the primary score is the raw cosine, so
    threshold=0.5 keeps everything with cos(centroid, candidate) >= 0.5."""
    seeds = [np.array([1.0, 0.0])]
    profile = _profile(seeds)
    sel = CentroidSelector(embedding_model="test")
    sel.fit(profile)
    n = 10
    cands = [_candidate(f"W{i}", np.array([1.0 - 0.1 * i, 0.05 * i])) for i in range(n)]
    full = sel.select(cands, profile)
    filtered = sel.select(cands, profile, threshold=0.5)
    assert len(filtered) <= len(full)
    assert all(r[0] >= 0.5 for r in filtered)
    # Sanity: at least one candidate should clear 0.5 against the +x seed.
    assert len(filtered) > 0
