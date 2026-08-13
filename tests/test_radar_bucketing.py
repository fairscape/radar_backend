"""A card's figure and colour both come from its rank percentile.

BUCKET_HIGH and BUCKET_MEDIUM are 0.95 and 0.925 — top 5% and top 7.5%.
That reading only holds against a percentile. Against ``score``, whose
range depends on how the selector and reranker are normalised, nothing
reached either threshold: 874 of 875 live cards came out "low" and the
top-ranked paper displayed in grey.

These go through ``candidate_row_to_card``, the function that actually
builds the card. An earlier version of this file re-implemented the
rule locally and passed while the real path was still wrong.
"""

from __future__ import annotations

import pytest

from rag_lib.api.mappers import candidate_row_to_card
from rag_lib.api.schemas import BUCKET_HIGH, BUCKET_MEDIUM


def _row(*, score, score_pct):
    """The subset of a ``profile_candidates JOIN papers`` row it reads."""
    return {
        "openalex_id": "https://openalex.org/W1",
        "title": "A paper", "doi": None, "abstract": "", "venue": "",
        "year": 2026, "publication_date": "2026-01-01", "topics_json": None,
        "score": score, "score_pct": score_pct,
    }


def _card(score, score_pct):
    return candidate_row_to_card(
        _row(score=score, score_pct=score_pct),
        profile_slug="p", active_topic_ids=[],
    )


def test_the_top_ranked_card_is_not_grey():
    """Measured live: rank one had score 0.909 and percentile 1.0, and
    displayed "100%" in grey because 0.909 is below 0.925."""
    card = _card(score=0.909, score_pct=1.0)
    assert card.bucket == "high"
    assert card.score == 1.0


def test_the_figure_and_the_colour_come_from_the_same_value():
    """They were consistent before — both read ``score`` — and a partial
    fix that moved only the filter made them disagree instead."""
    card = _card(score=0.42, score_pct=0.97)
    assert card.score == 0.97
    assert card.bucket == "high"


@pytest.mark.parametrize("pct,expected", [
    (1.0, "high"),
    (BUCKET_HIGH, "high"),
    (0.94, "medium"),
    (BUCKET_MEDIUM, "medium"),
    (0.9, "low"),
    (0.0, "low"),
])
def test_percentiles_land_in_the_intended_bands(pct, expected):
    assert _card(score=0.5, score_pct=pct).bucket == expected


def test_a_blend_of_two_min_maxed_stages_would_never_qualify():
    """Both stages normalise to [0, 1] and the top paper is rarely top on
    both, so the blend peaks well under 0.925 — which is why bucketing on
    it produced no high or medium cards at all."""
    assert _card(score=0.909, score_pct=0.5).bucket == "low"


def test_rows_written_before_the_percentile_existed_fall_back():
    """score_pct is NULL on candidates persisted by older gathers."""
    assert _card(score=0.96, score_pct=None).bucket == "high"
    assert _card(score=0.50, score_pct=None).bucket == "low"
