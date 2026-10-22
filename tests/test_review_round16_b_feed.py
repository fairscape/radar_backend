"""Review 16 (b): the feed's untriagedNotShown against what the cap hid."""

from __future__ import annotations

from rag_lib.api.services import radar as radar_service
from tests.test_review_round12_journeys import (  # noqa: F401
    env, app, _feed, _go_live, _request, _scan,
)


def test_one_interest_over_the_cap_says_nothing_is_hidden(app, env, monkeypatch):
    a, _ = _go_live(app, env, name="One", threshold=0.0)
    _scan(app, a)
    full = _feed(app, a)
    total = sum(1 for c in full["cards"] if c.get("state") is None)
    assert total > 4
    monkeypatch.setattr(radar_service, "_DAILY_LIMIT", 4)
    for view in (_feed(app, a), _feed(app)):
        fresh = [c for c in view["cards"] if c.get("state") is None]
        assert len(fresh) == 4
        # total - 4 untriaged papers are not in the list; the count says 0.
        assert view["untriagedNotShown"] >= total - 4, (view["untriagedNotShown"], total)


def test_hidden_papers_are_counted_by_bucket(app, env, monkeypatch):
    # The feed marks only the buckets with papers left out as "N+".
    a, _ = _go_live(app, env, name="Buckets", threshold=0.0)
    _scan(app, a)
    full = [c for c in _feed(app, a)["cards"] if c.get("state") is None]
    assert len(full) > 2
    monkeypatch.setattr(radar_service, "_DAILY_LIMIT", 2)
    view = _feed(app, a)
    by = view["untriagedNotShownByBucket"]
    assert view["untriagedNotShown"] > 0
    assert sum(by.values()) == view["untriagedNotShown"], by
    shown = {c["id"] for c in view["cards"]}
    hidden = [c for c in full if c["id"] not in shown]
    for b, n in by.items():
        assert sum(1 for c in hidden if c["bucket"] == b) >= n, (b, by)
