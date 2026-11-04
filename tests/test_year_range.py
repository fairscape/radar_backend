"""Publication-year ranges: OpenAlex queries, the year-by-year fetch, the
backfill job and route, and the feed's year rules (migration 0018)."""

from __future__ import annotations

import pytest

from rag_lib import years as years_rules
from rag_lib.gatherers.openalex import OpenAlexGatherer
from rag_lib.openalex_client import OpenAlexClient
from rag_lib.openalex_tiers import build_filter_string
from rag_lib.paper import Paper
from rag_lib.profile import Profile


def test_a_filter_can_be_bounded_on_both_sides():
    s = build_filter_string([("topics.id", ["T1"], "or")], "2019-01-01", until="2019-12-31")
    assert s.startswith("from_publication_date:2019-01-01,to_publication_date:2019-12-31,")
    assert "to_publication_date" not in build_filter_string([], "2026-01-01")   # the daily scans stay open


def test_paginate_sends_the_sort(monkeypatch):
    sent = []
    client = OpenAlexClient(mailto="x@example.com")
    monkeypatch.setattr(client, "_get", lambda path, params: sent.append(dict(params)) or {"results": [], "meta": {}})
    client.paginate_filter("type:article", limit=5, sort="cited_by_count:desc")
    client.paginate_filter("type:article", limit=5)
    assert sent[0]["sort"] == "cited_by_count:desc" and "sort" not in sent[1]


class _RecordingClient:
    def __init__(self):
        self.calls = []
        self.api_calls = 0

    def paginate_filter(self, filter_str, *, limit=None, per_page=200, sort=None):
        self.calls.append((filter_str, limit, sort))
        self.api_calls += 1
        year = filter_str.split("from_publication_date:")[1][:4]
        return [{"id": f"https://openalex.org/W{year}{i}", "title": f"Paper {year}-{i}"} for i in range(3)]

    def paper_from_work(self, w, source=None):
        return Paper(doi=None, openalex_id=w["id"], title=w["title"], year=int(w["id"][-5:-1]))


def _gatherer(client):
    g = OpenAlexGatherer(mailto="x@example.com")
    g._resolve_client = lambda: client
    return g


def test_each_year_is_one_query_over_the_enabled_topics_most_cited_first():
    client = _RecordingClient()
    profile = Profile(name="p", topic_filters={"topics": [
        {"id": "https://openalex.org/T1", "on": True}, {"id": "https://openalex.org/T2", "on": True},
        {"id": "https://openalex.org/T3", "on": False}]})
    papers = _gatherer(client).fetch_years(profile, 2019, 2021, per_year=50)
    assert [c[0].split(",")[:2] for c in client.calls] == [
        [f"from_publication_date:{y}-01-01", f"to_publication_date:{y}-12-31"] for y in (2019, 2020, 2021)]
    assert all("topics.id:T1|T2" in c[0] and "T3" not in c[0] for c in client.calls)
    assert all(c[1] == 50 and c[2] == "cited_by_count:desc" for c in client.calls)
    assert len(papers) == 9


def test_all_topics_switched_off_fetches_nothing():
    client = _RecordingClient()
    profile = Profile(name="p", topic_filters={"topics": [{"id": "T1", "on": False}]})
    assert _gatherer(client).fetch_years(profile, 2019, 2021, per_year=50) == []
    assert client.calls == []


def test_ranges_are_checked_and_a_backfill_is_capped():
    with pytest.raises(ValueError):
        years_rules.check_range(2022, 2019)
    with pytest.raises(ValueError):
        years_rules.check_range(1800, None)
    with pytest.raises(ValueError):
        years_rules.check_range(None, years_rules.this_year() + 1)
    now = years_rules.this_year()
    assert years_rules.fetch_span(1990, None) == (now - years_rules.MAX_BACKFILL_YEARS + 1, now)
    assert years_rules.fetch_span(2018, 2020) == (2018, 2020)


def _profile_with(n_topics: int) -> Profile:
    return Profile(name="p", topic_filters={"topics": [
        {"id": f"https://openalex.org/T{i}", "on": True} for i in range(n_topics)]})


def test_a_backfill_over_more_than_ten_topics_sends_wide_queries():
    """OpenAlex holds a filter ORing more than 10 topics to one a second;
    the client paces those for the whole process (CR.11 L8)."""
    from rag_lib.openalex_client import is_wide

    for n, wide in ((11, True), (10, False)):
        client = _RecordingClient()
        _gatherer(client).fetch_years(_profile_with(n), 2019, 2021, per_year=50)
        assert len(client.calls) == 3
        assert all(is_wide(f) is wide for f, _, _ in client.calls), (n, client.calls[0][0])


def test_wide_queries_are_a_second_apart_across_clients_and_threads(monkeypatch):
    """Review SF-04: two users' backfills, each its own client, in two
    threads. Every two wide requests are let through at least a second
    apart; a narrow one is not held.

    Measured where the gate lets a request through (B2-2): the time it is
    then sent also holds the thread's own scheduling delay, which on a busy
    machine took a few milliseconds off a gap now and then."""
    import threading
    import time

    from rag_lib import openalex_client as oc
    from rag_lib.openalex_client import OpenAlexClient

    slots = []
    real_wait = oc._wait_for_wide_slot

    def wait():
        real_wait()
        slots.append(oc._wide_next - oc.WIDE_OR_SECONDS)   # set under the gate's lock

    monkeypatch.setattr(oc, "_wait_for_wide_slot", wait)

    sent = []

    class Session:
        headers: dict = {}

        def get(self, url, params=None, timeout=None):
            sent.append((time.monotonic(), params["filter"]))
            time.sleep(0.05)

            class R:
                status_code = 200
                headers: dict = {}

                def raise_for_status(self):
                    pass

                def json(self):
                    return {"results": [], "meta": {"count": 0}}
            return R()

    wide = "topics.id:" + "|".join(f"T{i}" for i in range(11)) + ",from_publication_date:2020-01-01"
    narrow = "topics.id:T1|T2,from_publication_date:2020-01-01"

    def user():
        c = OpenAlexClient(mailto="t@example.com", session=Session(), rate_limit_sleep=0)
        for _ in range(3):
            c.paginate_filter(wide, per_page=10)

    threads = [threading.Thread(target=user) for _ in range(2)]
    for t in threads:
        t.start()
    time.sleep(0.2)
    t0 = time.monotonic()
    OpenAlexClient(mailto="t@example.com", session=Session(), rate_limit_sleep=0).paginate_filter(narrow, per_page=10)
    narrow_took = time.monotonic() - t0
    for t in threads:
        t.join()
    times = sorted(at for at, f in sent if f == wide)
    assert len(times) == 6 and len(slots) == 6
    slots.sort()
    assert min(b - a for a, b in zip(slots, slots[1:])) >= oc.WIDE_OR_SECONDS - 1e-6
    # And sent about a second apart: loose, for the scheduling delay above.
    assert min(b - a for a, b in zip(times, times[1:])) >= 0.8
    assert narrow_took < 0.9
