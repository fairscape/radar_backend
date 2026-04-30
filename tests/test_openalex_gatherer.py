"""OpenAlexGatherer.fetch behavior beyond the generic compliance suite."""

from __future__ import annotations

import pytest

from rag_lib.gatherers.openalex import OpenAlexGatherer
from rag_lib.paper import Paper
from rag_lib.profile import Profile
from tests.fake_openalex_client import FakeOpenAlexClient, canned_openalex_work


def _profile_with_filters() -> Profile:
    return Profile(
        name="p",
        topic_filters={
            "topics": [{"id": "T1", "display_name": "t", "count": 2}],
            "subfields": [], "fields": [], "domains": [],
        },
    )


def test_fetch_maps_hits_through_paper_from_work():
    client = FakeOpenAlexClient(search_results=[
        canned_openalex_work(doi="10.x/1", openalex_id="WX1", title="Cand A"),
        canned_openalex_work(doi="10.x/2", openalex_id="WX2", title="Cand B"),
    ])
    g = OpenAlexGatherer(mailto="test@example.com", client=client)
    papers = g.fetch(_profile_with_filters(), since="2026-01-01")
    assert len(papers) == 2
    assert all(isinstance(p, Paper) for p in papers)
    assert papers[0].title == "Cand A"
    assert papers[0].source == "openalex_gatherer"


def test_fetch_passes_topic_filters_through():
    client = FakeOpenAlexClient(search_results=[])
    g = OpenAlexGatherer(mailto="test@example.com", client=client)
    profile = _profile_with_filters()
    g.fetch(profile, since="2026-01-01")
    assert len(client.searches) == 1
    assert client.searches[0]["topic_filters"] == profile.topic_filters
    assert client.searches[0]["since"] == "2026-01-01"


def test_fetch_records_cost():
    client = FakeOpenAlexClient(search_results=[
        canned_openalex_work(doi="10.x/1", openalex_id="WX1", title="X"),
    ])
    g = OpenAlexGatherer(mailto="test@example.com", client=client)
    g.fetch(_profile_with_filters(), since="2026-01-01")
    cost = g.cost()
    assert cost["api_calls"] >= 1
    assert cost["wall_seconds"] >= 0


def test_fetch_requires_mailto_if_no_client_injected():
    """Without mailto AND without an injected client, fetch() must refuse
    rather than hit the real OpenAlex with an empty polite-pool header."""
    g = OpenAlexGatherer()  # no mailto, no client
    with pytest.raises(ValueError, match="mailto"):
        g.fetch(_profile_with_filters(), since="2026-01-01")
