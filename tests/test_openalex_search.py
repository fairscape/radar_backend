"""OpenAlexClient.search_works tests. Uses a fake HTTP session to
verify filter-string construction and cursor pagination without
network."""

from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

import pytest

from rag_lib.openalex_client import OpenAlexClient


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    """Records every GET; returns canned JSON pages in order."""

    def __init__(self, pages: list[dict]):
        self._pages = list(pages)
        self.calls: list[dict] = []
        self.headers: dict = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {})})
        if not self._pages:
            return _FakeResponse({"results": [], "meta": {"next_cursor": None}})
        return _FakeResponse(self._pages.pop(0))


def _client(pages: list[dict]) -> OpenAlexClient:
    session = _FakeSession(pages)
    c = OpenAlexClient(mailto="test@example.com", rate_limit_sleep=0, session=session)
    # Expose the session for call inspection.
    c._fake_session = session  # type: ignore[attr-defined]
    return c


# ----------------------------------------------------------------------
# Filter-string construction
# ----------------------------------------------------------------------


def test_filter_string_includes_from_publication_date():
    c = _client([{"results": [], "meta": {"next_cursor": None}}])
    c.search_works({}, since="2026-03-01")
    params = c._fake_session.calls[0]["params"]
    assert "from_publication_date:2026-03-01" in params["filter"]


def test_filter_string_or_joins_topic_ids():
    c = _client([{"results": [], "meta": {"next_cursor": None}}])
    c.search_works({"topics": ["T1", "T2", "T3"]}, since="2026-03-01")
    f = c._fake_session.calls[0]["params"]["filter"]
    assert "topics.id:T1|T2|T3" in f


def test_filter_string_accepts_dict_shape_from_aggregate():
    """Profile.aggregate_topic_filters returns {id,display_name,count} dicts."""
    c = _client([{"results": [], "meta": {"next_cursor": None}}])
    c.search_works(
        {"topics": [{"id": "T1", "display_name": "x", "count": 5}]},
        since="2026-03-01",
    )
    f = c._fake_session.calls[0]["params"]["filter"]
    assert "topics.id:T1" in f


def test_filter_string_includes_every_hierarchy_level():
    c = _client([{"results": [], "meta": {"next_cursor": None}}])
    c.search_works(
        {
            "topics": ["T1"], "subfields": ["S1"],
            "fields": ["F1"], "domains": ["D1"],
            "primary_topic": ["PT1"],
        },
        since="2026-03-01",
    )
    f = c._fake_session.calls[0]["params"]["filter"]
    assert "topics.id:T1" in f
    assert "topics.subfield.id:S1" in f
    assert "topics.field.id:F1" in f
    assert "topics.domain.id:D1" in f
    assert "primary_topic.id:PT1" in f


def test_filter_string_strips_openalex_url_prefix():
    """OpenAlex /works silently returns 0 when multiple URL-form ID
    clauses are ANDed; we always emit bare IDs."""
    c = _client([{"results": [], "meta": {"next_cursor": None}}])
    c.search_works(
        {
            "topics": [{"id": "https://openalex.org/T11289"}],
            "subfields": [{"id": "https://openalex.org/subfields/1312"}],
            "fields": [{"id": "https://openalex.org/fields/13"}],
            "domains": [{"id": "https://openalex.org/domains/1"}],
        },
        since="2026-03-01",
    )
    f = c._fake_session.calls[0]["params"]["filter"]
    assert "https://" not in f
    assert "topics.id:T11289" in f
    assert "topics.subfield.id:1312" in f
    assert "topics.field.id:13" in f
    assert "topics.domain.id:1" in f


def test_search_falls_back_when_full_filter_returns_zero():
    """If the full topic_filters query returns nothing, retry with one
    hierarchy level at a time (narrowest first)."""
    pages = [
        # Full filter (topics+subfields+fields+domains) -> empty
        {"results": [], "meta": {"next_cursor": None}},
        # Fallback 1: topics only -> empty
        {"results": [], "meta": {"next_cursor": None}},
        # Fallback 2: subfields only -> hit
        {"results": [{"id": "W42"}], "meta": {"next_cursor": None}},
    ]
    c = _client(pages)
    out = c.search_works(
        {
            "topics": ["T1"], "subfields": ["S1"],
            "fields": ["F1"], "domains": ["D1"],
        },
        since="2026-03-01",
    )
    assert [w["id"] for w in out] == ["W42"]
    # Three queries total: full, then topics-only, then subfields-only.
    assert len(c._fake_session.calls) == 3
    f0 = c._fake_session.calls[0]["params"]["filter"]
    f1 = c._fake_session.calls[1]["params"]["filter"]
    f2 = c._fake_session.calls[2]["params"]["filter"]
    assert "topics.id:T1" in f0 and "topics.subfield.id:S1" in f0
    assert "topics.id:T1" in f1 and "topics.subfield.id" not in f1
    assert "topics.subfield.id:S1" in f2 and "topics.id:" not in f2


def test_search_does_not_fall_back_when_full_filter_succeeds():
    pages = [{"results": [{"id": "W1"}], "meta": {"next_cursor": None}}]
    c = _client(pages)
    out = c.search_works(
        {"topics": ["T1"], "fields": ["F1"]},
        since="2026-03-01",
    )
    assert [w["id"] for w in out] == ["W1"]
    assert len(c._fake_session.calls) == 1


def test_mailto_always_attached():
    c = _client([{"results": [], "meta": {"next_cursor": None}}])
    c.search_works({"topics": ["T1"]}, since="2026-03-01")
    params = c._fake_session.calls[0]["params"]
    assert params["mailto"] == "test@example.com"


# ----------------------------------------------------------------------
# Cursor pagination
# ----------------------------------------------------------------------


def test_paginates_until_next_cursor_null():
    pages = [
        {"results": [{"id": "W1"}, {"id": "W2"}], "meta": {"next_cursor": "c2"}},
        {"results": [{"id": "W3"}], "meta": {"next_cursor": "c3"}},
        {"results": [{"id": "W4"}], "meta": {"next_cursor": None}},
    ]
    c = _client(pages)
    out = c.search_works({"topics": ["T1"]}, since="2026-03-01")
    assert [w["id"] for w in out] == ["W1", "W2", "W3", "W4"]
    assert len(c._fake_session.calls) == 3
    # First call uses cursor=*, subsequent use the returned cursors.
    assert c._fake_session.calls[0]["params"]["cursor"] == "*"
    assert c._fake_session.calls[1]["params"]["cursor"] == "c2"
    assert c._fake_session.calls[2]["params"]["cursor"] == "c3"


def test_limit_stops_pagination_early():
    pages = [
        {"results": [{"id": "W1"}, {"id": "W2"}], "meta": {"next_cursor": "c2"}},
        {"results": [{"id": "W3"}, {"id": "W4"}], "meta": {"next_cursor": "c3"}},
    ]
    c = _client(pages)
    out = c.search_works({"topics": ["T1"]}, since="2026-03-01", limit=3)
    assert len(out) == 3
    # Should have hit at most 2 pages.
    assert len(c._fake_session.calls) == 2


def test_per_page_param_plumbed_through():
    c = _client([{"results": [], "meta": {"next_cursor": None}}])
    c.search_works({}, since="2026-03-01", per_page=50)
    assert c._fake_session.calls[0]["params"]["per-page"] == 50


def test_api_calls_counter_increments():
    pages = [
        {"results": [], "meta": {"next_cursor": "c2"}},
        {"results": [], "meta": {"next_cursor": None}},
    ]
    c = _client(pages)
    assert c.api_calls == 0
    c.search_works({}, since="2026-03-01")
    assert c.api_calls == 2
