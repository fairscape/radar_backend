"""Review 6 (d): "All interests" capped the list at 200 cards shared out
fairly, and nothing said more papers were waiting."""

from __future__ import annotations

from rag_lib.api.services import radar as radar_service
from tests.test_review_round12_journeys import (  # noqa: F401
    env, app, _feed, _go_live, _request, _scan,
)


def test_the_feed_says_how_many_papers_the_cap_left_out(app, env, monkeypatch):
    a, _ = _go_live(app, env, name="One", threshold=0.0)
    b, _ = _go_live(app, env, name="Two", seeds_from_researcher=False, threshold=0.0)
    _scan(app, a)
    _scan(app, b)
    full = _feed(app)
    total = len(full["cards"])
    assert total > 4 and full["untriagedNotShown"] == 0
    monkeypatch.setattr(radar_service, "_DAILY_LIMIT", 4)
    capped = _feed(app)
    assert len(capped["cards"]) == 4
    # At least: each interest is read up to the cap before it is shared out.
    assert 0 < capped["untriagedNotShown"] <= total - 4
