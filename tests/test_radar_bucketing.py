"""Cards are bucketed on the percentile they display, not on the score.

BUCKET_HIGH and BUCKET_MEDIUM are 0.95 and 0.925 — top 5% and top 7.5%.
That reading only holds against a rank percentile. Against ``score``,
whose range depends on how the selector and reranker are normalised,
nothing reached either threshold and every card came out "low".
"""

from __future__ import annotations

import pytest

from rag_lib.api.mappers import _bucket_for
from rag_lib.api.schemas import BUCKET_HIGH, BUCKET_MEDIUM


def _bucket_of(score_pct, score):
    """The rule radar.daily applies, in isolation."""
    pct = score_pct
    return _bucket_for(float(pct if pct is not None else score))


def test_the_top_card_is_not_grey():
    """Measured live: rank one had score 0.909 and percentile 1.0 — it
    displayed "100%" in grey, because 0.909 is below 0.925."""
    assert _bucket_of(1.0, 0.909) == "high"


def test_a_blended_score_never_reaches_the_thresholds():
    """Both stages min-max to [0, 1] and the top paper is rarely top on
    both, so the blend peaks well under 0.925. Bucketing on it made 874
    of 875 live cards "low"."""
    assert _bucket_for(0.909) == "low"
    assert _bucket_for(0.846) == "low"


@pytest.mark.parametrize("pct,expected", [
    (1.0, "high"),
    (BUCKET_HIGH, "high"),
    (0.94, "medium"),
    (BUCKET_MEDIUM, "medium"),
    (0.9, "low"),
    (0.0, "low"),
])
def test_percentiles_land_in_the_intended_bands(pct, expected):
    assert _bucket_of(pct, 0.5) == expected


def test_rows_written_before_the_percentile_existed_fall_back():
    """score_pct is NULL on candidates persisted by older gathers; those
    keep the previous behaviour rather than crashing."""
    assert _bucket_of(None, 0.96) == "high"
    assert _bucket_of(None, 0.5) == "low"
