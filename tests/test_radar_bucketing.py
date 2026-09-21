"""Cards are coloured by what their similarity means for the profile.

``high`` — at least as similar to the centroid as the least typical
seed ("like your own papers"). ``medium`` — above the profile's
threshold. ``low`` — below it. Rows without a raw similarity fall back
to the rank percentile with cut-points at top 10% / top third.

Measured scale (docs/CALIBRATION.md): candidates sit in 0.81–0.96, the
seeds' own leave-one-out similarity in 0.93–0.96. The old cut-points of
0.95 / 0.925 on a raw score made 874 of 875 live cards grey.

These go through ``candidate_row_to_card``, the function that actually
builds the card.
"""

from __future__ import annotations

import pytest

from rag_lib.api.mappers import candidate_row_to_card
from rag_lib.api.schemas import BUCKET_HIGH, BUCKET_MEDIUM


def _row(*, score, score_pct, score_raw=None):
    """The subset of a ``profile_candidates JOIN papers`` row it reads."""
    return {
        "openalex_id": "https://openalex.org/W1",
        "title": "A paper", "doi": None, "abstract": "", "venue": "",
        "year": 2026, "publication_date": "2026-01-01", "topics_json": None,
        "score": score, "score_pct": score_pct, "score_raw": score_raw,
    }


def _card(score, score_pct, score_raw=None, **kw):
    return candidate_row_to_card(
        _row(score=score, score_pct=score_pct, score_raw=score_raw),
        profile_slug="p", active_topic_ids=[], **kw,
    )


def test_similarity_is_exposed_and_the_figure_stays_the_percentile():
    card = _card(score=0.42, score_pct=0.97, score_raw=0.931)
    assert card.score == 0.97
    assert card.similarity == 0.931
    assert card.centroidCos == 0.931


@pytest.mark.parametrize("raw,expected", [
    (0.96, "high"),     # above the seed band floor
    (0.927, "high"),
    (0.92, "medium"),   # above threshold, below the seeds
    (0.917, "medium"),
    (0.90, "low"),      # below the threshold the user set
])
def test_similarity_is_read_against_threshold_and_seed_band(raw, expected):
    card = _card(score=0.5, score_pct=0.5, score_raw=raw, threshold=0.917, seed_sim_min=0.927)
    assert card.bucket == expected


def test_without_a_seed_band_high_is_unreachable_but_medium_is_not():
    assert _card(score=0.5, score_pct=1.0, score_raw=0.95, threshold=0.9).bucket == "medium"
    assert _card(score=0.5, score_pct=1.0, score_raw=0.85, threshold=0.9).bucket == "low"


@pytest.mark.parametrize("pct,expected", [
    (1.0, "high"),
    (BUCKET_HIGH, "high"),
    (0.8, "medium"),
    (BUCKET_MEDIUM, "medium"),
    (0.5, "low"),
    (0.0, "low"),
])
def test_without_a_threshold_the_percentile_decides(pct, expected):
    assert _card(score=0.5, score_pct=pct, score_raw=0.93).bucket == expected


def test_rows_written_before_the_percentile_existed_fall_back():
    """score_pct and score_raw are NULL on candidates persisted by older gathers."""
    assert _card(score=0.96, score_pct=None).bucket == "high"
    assert _card(score=0.50, score_pct=None).bucket == "low"
    assert _card(score=0.50, score_pct=None).similarity is None


def test_a_blended_score_never_leaks_into_the_similarity():
    """With a reranker on, ``score`` is a batch-relative blend; the
    similarity must come from ``score_raw`` or be absent."""
    card = _card(score=0.42, score_pct=0.97)
    assert card.similarity is None
