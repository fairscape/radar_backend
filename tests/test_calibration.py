"""Calibration constants behave the way the measurements say they should."""

from __future__ import annotations

import numpy as np
import pytest

from rag_lib import calibration as cal


def test_bands_follow_the_measured_scale():
    assert cal.coherence_label(0.915, 0.026, 4) == "focused"   # real FAIR set
    assert cal.coherence_label(0.926, 0.022, 4) == "focused"   # real vitals set
    assert cal.coherence_label(0.878, 0.073, 8) == "broad"     # the two mixed
    assert cal.coherence_label(0.847, 0.03, 4) == "mixed"      # random same-field
    assert cal.coherence_label(0.92, 0.08, 6) == "broad"       # high median, two groups
    assert cal.coherence_label(None, None, 1) == "single"
    assert cal.coherence_label(None, None, 0) == "none"


def test_agreement_is_readable_and_bounded():
    assert cal.agreement_score(0.915) == 73
    assert cal.agreement_score(0.847) == 21
    assert cal.agreement_score(0.5) == 0
    assert cal.agreement_score(0.99) == 100
    assert cal.agreement_score(None) is None


def test_health_maps_labels():
    assert cal.health_for(0.92, 4) == "ok"
    assert cal.health_for(0.88, 4) == "warn"
    assert cal.health_for(0.84, 4) == "err"
    assert cal.health_for(None, 1) == "warn"


def test_seed_band_is_leave_one_out():
    rng = np.random.default_rng(0)
    base = rng.normal(size=8)
    vecs = [base + rng.normal(scale=0.2, size=8) for _ in range(5)]
    band = cal.seed_similarity_band(vecs)
    assert band is not None
    assert len(band["values"]) == 5
    assert band["min"] <= band["median"] <= band["max"] <= 1.0
    assert cal.seed_similarity_band([base]) is None


def test_least_similar_pair_finds_the_outlier():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.99, 0.1, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    i, j, cos = cal.least_similar_pair([a, b, c])
    assert {i, j} == {0, 2}
    assert cos == pytest.approx(0.0, abs=1e-9)


def test_threshold_suggestion_sits_below_the_seeds_inside_the_pool():
    scores = list(np.linspace(0.80, 0.95, 200))
    thr = cal.suggest_threshold(0.927, scores)
    assert thr == pytest.approx(0.917, abs=1e-3)
    # Seeds far above the pool: clamp to the 98th percentile so something passes.
    assert cal.suggest_threshold(0.99, scores) <= np.percentile(scores, 98) + 1e-3
    # Seeds inside the pool's bulk: clamp so the top quarter is the floor.
    assert cal.suggest_threshold(0.80, scores) >= np.percentile(scores, 75) - 1e-3
    # No seeds: 90th percentile of the pool.
    assert cal.suggest_threshold(None, scores) == pytest.approx(np.percentile(scores, 90), abs=1e-3)
    # Nothing at all: fallback.
    assert cal.suggest_threshold(None, []) == cal.THRESHOLD_FALLBACK


def test_score_range_covers_scores_and_seed_band():
    lo, hi = cal.score_range([0.85, 0.90], {"min": 0.93, "max": 0.96})
    assert lo < 0.85 and hi > 0.96 and hi <= 1.0
    assert cal.score_range([], None) is None
