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
