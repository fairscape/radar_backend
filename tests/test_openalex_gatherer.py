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


def test_fetch_renders_topic_filters_into_filter_string():
    """The gatherer walks tier_specs and calls paginate_filter with a
    rendered filter string. With ``min_results`` defaulting to 50 and
    paginate_filter returning [] for every tier, the gatherer keeps
    walking; we just verify the first attempt carries the seed's topic
    id in the rendered filter and the date floor is present."""
    client = FakeOpenAlexClient(search_results=[])
    g = OpenAlexGatherer(mailto="test@example.com", client=client)
    profile = _profile_with_filters()
    g.fetch(profile, since="2026-01-01")
    assert len(client.paginate_calls) >= 1
    first_filter = client.paginate_calls[0]["filter_str"]
    assert "topics.id:T1" in first_filter
    assert "from_publication_date:2026-01-01" in first_filter
    # last_filter_str carries whatever tier the gatherer last attempted,
    # so the scheduler / wizard can persist or display it.
    assert g.last_filter_str == client.paginate_calls[-1]["filter_str"]


def test_fetch_walks_tiers_and_stops_at_first_meeting_min_results():
    """With min_results=2 and a profile whose seeds carry a topic that
    appears in 100% of papers (must-have prevalence), tier 1 fires
    first. Returning 2 hits clears min_results and the walk stops
    there — tier 2 / tier 3 should NOT be queried."""
    client = FakeOpenAlexClient(tier_results=[
        # tier 1 (must-have-AND) — returns 2 hits, meets min_results.
        [
            canned_openalex_work(doi="10.x/1", openalex_id="WX1", title="A"),
            canned_openalex_work(doi="10.x/2", openalex_id="WX2", title="B"),
        ],
        # would be tier 2 — must NOT be reached.
        [canned_openalex_work(doi="10.x/3", openalex_id="WX3", title="C")],
    ])
    profile = Profile(
        name="p",
        papers=[
            Paper.from_dict({
                "doi": "10.1/a", "openalex_id": "W1", "title": "A", "abstract": "x",
                "primary_topic": {"id": "T11", "display_name": "core"},
            }),
            Paper.from_dict({
                "doi": "10.1/b", "openalex_id": "W2", "title": "B", "abstract": "y",
                "primary_topic": {"id": "T11", "display_name": "core"},
            }),
        ],
        # No topics in topic_filters -> the profile was never
        # aggregated, so fetch() falls back to the tier walk.
        topic_filters={
            "topics": [], "subfields": [], "fields": [], "domains": [],
        },
    )
    g = OpenAlexGatherer(
        mailto="test@example.com", client=client, min_results=2,
    )
    papers = g.fetch(profile, since="2026-01-01")
    assert len(papers) == 2
    assert g.last_tier_used == "must-have-AND"
    # Only one paginate_filter call — tier 2 was never queried.
    assert len(client.paginate_calls) == 1
    # Must-have AND emits topics.id:T11 (single repeated key).
    assert "topics.id:T11" in client.paginate_calls[0]["filter_str"]


def test_fetch_falls_through_when_tier_under_min_results():
    """Tier 1 returns 0; gatherer must walk to tier 2 and report that
    tier's filter as ``last_tier_used``."""
    client = FakeOpenAlexClient(tier_results=[
        [],  # tier 1: empty
        # tier 2: 3 hits
        [
            canned_openalex_work(doi="10.x/1", openalex_id="WX1", title="A"),
            canned_openalex_work(doi="10.x/2", openalex_id="WX2", title="B"),
            canned_openalex_work(doi="10.x/3", openalex_id="WX3", title="C"),
        ],
    ])
    profile = Profile(
        name="p",
        papers=[
            Paper.from_dict({
                "doi": "10.1/a", "openalex_id": "W1", "title": "A", "abstract": "x",
                "primary_topic": {"id": "T11", "display_name": "core"},
            }),
        ],
        # No topics in topic_filters -> tier-walk fallback.
        topic_filters={
            "topics": [],
            "subfields": [{"id": "SF1", "display_name": "sf", "count": 1}],
            "fields": [], "domains": [],
        },
    )
    g = OpenAlexGatherer(
        mailto="test@example.com", client=client, min_results=2,
    )
    papers = g.fetch(profile, since="2026-01-01")
    assert len(papers) == 3
    assert g.last_tier_used == "top-topics-OR-subfields-AND"
    assert len(client.paginate_calls) == 2


def test_fetch_unions_and_dedupes_across_tiers():
    """Tier 1 and tier 2 are each below min_results individually but
    their dedup'd union clears it. Papers from earlier (tighter) tiers
    must survive into the returned pile, and a paper appearing in both
    tiers is counted once. With cumulative-stop, tier 3 is not queried
    once the union meets the floor."""
    client = FakeOpenAlexClient(tier_results=[
        # tier 1 (must-have-AND): 2 hits, both unique to this tier.
        [
            canned_openalex_work(doi="10.x/1", openalex_id="WX1", title="A"),
            canned_openalex_work(doi="10.x/2", openalex_id="WX2", title="B"),
        ],
        # tier 2: 3 hits — WX2 overlaps with tier 1, WX3/WX4 are new.
        [
            canned_openalex_work(doi="10.x/2", openalex_id="WX2", title="B"),
            canned_openalex_work(doi="10.x/3", openalex_id="WX3", title="C"),
            canned_openalex_work(doi="10.x/4", openalex_id="WX4", title="D"),
        ],
        # tier 3: must NOT be reached — union after tier 2 is 4 ≥ min=4.
        [canned_openalex_work(doi="10.x/9", openalex_id="WX9", title="Z")],
    ])
    profile = Profile(
        name="p",
        papers=[
            Paper.from_dict({
                "doi": "10.1/a", "openalex_id": "W1", "title": "A", "abstract": "x",
                "primary_topic": {"id": "T11", "display_name": "core"},
            }),
        ],
        # No topics in topic_filters -> tier-walk fallback.
        topic_filters={
            "topics": [],
            "subfields": [{"id": "SF1", "display_name": "sf", "count": 1}],
            "fields": [], "domains": [],
        },
    )
    g = OpenAlexGatherer(
        mailto="test@example.com", client=client, min_results=4,
    )
    papers = g.fetch(profile, since="2026-01-01")
    ids = [p.openalex_id for p in papers]
    assert ids == ["WX1", "WX2", "WX3", "WX4"]
    assert g.last_tier_used == "top-topics-OR-subfields-AND"
    assert len(client.paginate_calls) == 2


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


# ---------------------------------------------------------------------------
# Per-topic quota gathering
# ---------------------------------------------------------------------------


def _quota_profile(*topic_specs) -> Profile:
    """Profile whose topic_filters carry (id, on) pairs."""
    return Profile(
        name="p",
        topic_filters={
            "topics": [
                {"id": tid, "display_name": tid, "count": 1, "on": on}
                for tid, on in topic_specs
            ],
            "subfields": [], "fields": [], "domains": [],
        },
    )


def test_per_topic_quota_queries_each_enabled_topic_once():
    client = FakeOpenAlexClient(tier_results=[
        [canned_openalex_work(doi="10.x/1", openalex_id="WA1", title="A1")],
        [canned_openalex_work(doi="10.x/2", openalex_id="WB1", title="B1")],
    ])
    g = OpenAlexGatherer(mailto="t@example.com", client=client, per_topic_quota=10)
    papers = g.fetch(_quota_profile(("T1", True), ("T2", True)), since="2026-01-01")

    assert len(client.paginate_calls) == 2
    assert "topics.id:T1" in client.paginate_calls[0]["filter_str"]
    assert "topics.id:T2" in client.paginate_calls[1]["filter_str"]
    assert g.last_tier_used == "per-topic-quota"
    assert {p.openalex_id for p in papers} == {"WA1", "WB1"}


def test_per_topic_quota_skips_disabled_topics():
    """A topic the user switched off must not be queried at all."""
    client = FakeOpenAlexClient(tier_results=[
        [canned_openalex_work(doi="10.x/1", openalex_id="WA1", title="A1")],
    ])
    g = OpenAlexGatherer(mailto="t@example.com", client=client)
    g.fetch(_quota_profile(("T1", True), ("T2", False)), since="2026-01-01")

    assert len(client.paginate_calls) == 1
    assert "topics.id:T1" in client.paginate_calls[0]["filter_str"]


def test_all_topics_disabled_gathers_nothing():
    """Switching every topic off means "none of these", not "use the
    seed metadata instead" — the tier walk must not run."""
    client = FakeOpenAlexClient(tier_results=[
        [canned_openalex_work(doi="10.x/1", openalex_id="WA1", title="A1")],
    ])
    g = OpenAlexGatherer(mailto="t@example.com", client=client)
    papers = g.fetch(_quota_profile(("T1", False), ("T2", False)), since="2026-01-01")

    assert papers == []
    assert client.paginate_calls == []
    assert g.last_tier_used == "no-enabled-topics"


def test_quota_caps_each_topic_contribution():
    """A prolific topic can't spend more than its quota."""
    client = FakeOpenAlexClient(tier_results=[
        [canned_openalex_work(doi=f"10.a/{i}", openalex_id=f"WA{i}", title=f"A{i}")
         for i in range(20)],
        [canned_openalex_work(doi=f"10.b/{i}", openalex_id=f"WB{i}", title=f"B{i}")
         for i in range(20)],
    ])
    g = OpenAlexGatherer(mailto="t@example.com", client=client, per_topic_quota=3)
    papers = g.fetch(_quota_profile(("T1", True), ("T2", True)), since="2026-01-01")

    ids = [p.openalex_id for p in papers]
    assert len(ids) == 6
    assert sum(1 for i in ids if i.startswith("WA")) == 3
    assert sum(1 for i in ids if i.startswith("WB")) == 3


def test_shared_papers_do_not_starve_later_topics():
    """The whole point of the round-robin.

    Both topics return the same first two papers. Draining T1 fully
    first would let it bank both and leave T2 with only its unique
    third paper. Round-robin deals them out alternately, so each topic
    still lands its quota.
    """
    shared = [
        canned_openalex_work(doi="10.s/1", openalex_id="WS1", title="S1"),
        canned_openalex_work(doi="10.s/2", openalex_id="WS2", title="S2"),
    ]
    client = FakeOpenAlexClient(tier_results=[
        shared + [canned_openalex_work(doi="10.a/9", openalex_id="WA9", title="A9")],
        shared + [canned_openalex_work(doi="10.b/9", openalex_id="WB9", title="B9")],
    ])
    g = OpenAlexGatherer(mailto="t@example.com", client=client, per_topic_quota=2)
    papers = g.fetch(_quota_profile(("T1", True), ("T2", True)), since="2026-01-01")

    ids = [p.openalex_id for p in papers]
    assert len(ids) == len(set(ids)), "deduplication failed"
    # T1 takes WS1, T2 takes WS2 (WS1 already claimed), then one each more.
    assert g.last_source_topics["WS1"] == "T1"
    assert g.last_source_topics["WS2"] == "T2"
    assert len([t for t in g.last_source_topics.values() if t == "T1"]) == 2
    assert len([t for t in g.last_source_topics.values() if t == "T2"]) == 2


def test_source_topics_recorded_for_every_returned_paper():
    client = FakeOpenAlexClient(tier_results=[
        [canned_openalex_work(doi="10.x/1", openalex_id="WA1", title="A1")],
        [canned_openalex_work(doi="10.x/2", openalex_id="WB1", title="B1")],
    ])
    g = OpenAlexGatherer(mailto="t@example.com", client=client)
    papers = g.fetch(_quota_profile(("T1", True), ("T2", True)), since="2026-01-01")

    assert set(g.last_source_topics) == {p.openalex_id for p in papers}
    assert g.last_source_topics["WA1"] == "T1"
    assert g.last_source_topics["WB1"] == "T2"


def test_limit_truncation_keeps_attribution_consistent():
    """A round can overshoot ``limit``; the surplus must not linger in
    last_source_topics, or the UI would credit topics for papers that
    were never returned."""
    client = FakeOpenAlexClient(tier_results=[
        [canned_openalex_work(doi=f"10.a/{i}", openalex_id=f"WA{i}", title=f"A{i}")
         for i in range(10)],
        [canned_openalex_work(doi=f"10.b/{i}", openalex_id=f"WB{i}", title=f"B{i}")
         for i in range(10)],
        [canned_openalex_work(doi=f"10.c/{i}", openalex_id=f"WC{i}", title=f"C{i}")
         for i in range(10)],
    ])
    g = OpenAlexGatherer(mailto="t@example.com", client=client, per_topic_quota=10)
    papers = g.fetch(
        _quota_profile(("T1", True), ("T2", True), ("T3", True)),
        since="2026-01-01", limit=5,
    )

    assert len(papers) == 5
    assert set(g.last_source_topics) == {p.openalex_id for p in papers}
