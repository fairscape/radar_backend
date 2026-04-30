"""attach_percentile (rag_lib/scoring/percentile.py)."""

from __future__ import annotations

from rag_lib.paper import Paper
from rag_lib.scoring import attach_percentile
from rag_lib.scoring.percentile import percentile_of


def _entry(score: float, oa_id: str, **breakdown) -> tuple[float, Paper, dict]:
    return (score, Paper(doi=None, openalex_id=oa_id, title=oa_id), dict(breakdown))


def test_single_entry_collapses_to_one():
    ranked = [_entry(0.7, "W1")]
    out = attach_percentile(ranked)
    assert out[0][2]["score_pct"] == 1.0


def test_top_to_bottom_spans_one_to_zero():
    ranked = [_entry(0.9, "W1"), _entry(0.5, "W2"), _entry(0.1, "W3")]
    attach_percentile(ranked)
    pcts = [r[2]["score_pct"] for r in ranked]
    assert pcts == [1.0, 0.5, 0.0]


def test_uniform_spread_for_n_entries():
    ranked = [_entry(1.0 - i * 0.1, f"W{i}") for i in range(11)]
    attach_percentile(ranked)
    pcts = [r[2]["score_pct"] for r in ranked]
    assert pcts[0] == 1.0
    assert pcts[-1] == 0.0
    # Strictly decreasing.
    assert all(pcts[i] > pcts[i + 1] for i in range(len(pcts) - 1))


def test_mutates_in_place():
    ranked = [_entry(0.9, "W1"), _entry(0.5, "W2")]
    breakdown_before = ranked[0][2]
    attach_percentile(ranked)
    # Same dict object — write was in-place rather than producing a copy.
    assert ranked[0][2] is breakdown_before
    assert "score_pct" in breakdown_before


def test_preserves_existing_breakdown_fields():
    ranked = [
        _entry(0.9, "W1", score_raw=0.81, score_max_seed=0.94),
        _entry(0.5, "W2", score_raw=0.45, score_max_seed=0.61),
    ]
    attach_percentile(ranked)
    assert ranked[0][2]["score_raw"] == 0.81
    assert ranked[0][2]["score_max_seed"] == 0.94
    assert ranked[0][2]["score_pct"] == 1.0


def test_percentile_of_helper_matches_attach():
    # rank-by-rank cross-check
    n = 7
    ranked = [_entry(1.0 - i * 0.1, f"W{i}") for i in range(n)]
    attach_percentile(ranked)
    for i, entry in enumerate(ranked):
        assert entry[2]["score_pct"] == percentile_of(i, n)


def test_empty_input_returns_empty():
    assert attach_percentile([]) == []
