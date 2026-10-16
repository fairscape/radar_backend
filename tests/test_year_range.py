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


class _SleepRecordingClient(_RecordingClient):
    def __init__(self):
        super().__init__()
        self.rate_limit_sleep = 0.15
        self.sleeps_at_call = []

    def paginate_filter(self, filter_str, **kw):
        self.sleeps_at_call.append(self.rate_limit_sleep)
        return super().paginate_filter(filter_str, **kw)


def _profile_with(n_topics: int) -> Profile:
    return Profile(name="p", topic_filters={"topics": [
        {"id": f"https://openalex.org/T{i}", "on": True} for i in range(n_topics)]})


def test_a_wide_or_list_is_queried_once_a_second():
    """OpenAlex refuses the second of two back-to-back queries that OR more
    than 10 topics; the client's pause after each request is raised to a
    second for the fetch and put back after."""
    client = _SleepRecordingClient()
    _gatherer(client).fetch_years(_profile_with(11), 2019, 2021, per_year=50)
    assert client.sleeps_at_call == [1.0, 1.0, 1.0]
    assert client.rate_limit_sleep == 0.15


def test_ten_topics_keep_the_usual_pace():
    client = _SleepRecordingClient()
    _gatherer(client).fetch_years(_profile_with(10), 2019, 2021, per_year=50)
    assert client.sleeps_at_call == [0.15, 0.15, 0.15]
