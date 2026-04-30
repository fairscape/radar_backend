"""Gatherer protocol compliance test.

Phase 1A shipped the FixtureGatherer in FULL and OpenAlexGatherer in
STUB. Phase 1B promotes OpenAlexGatherer to FULL by wiring a
FakeOpenAlexClient behind it so the compliance suite stays offline.

Three parametrizations:

- FULL_GATHERERS — real behaviour required.
- STUB_GATHERERS — empty in 1B (kept for future plugin gatherers).
- TestStubsRaise — enforces the NotImplementedError contract on stubs.
"""

from __future__ import annotations

import pytest

from rag_lib.gatherer import Gatherer
from rag_lib.gatherers.openalex import OpenAlexGatherer
from rag_lib.paper import Paper
from rag_lib.profile import Profile
from tests.fake_openalex_client import FakeOpenAlexClient, canned_openalex_work
from tests.fixture_gatherer import FixtureGatherer


def _sample_profile() -> Profile:
    return Profile(
        name="neonatal_vitals",
        papers=[
            Paper(doi="10.1/a", openalex_id="W1", title="A", abstract="alpha"),
            Paper(doi="10.1/b", openalex_id="W2", title="B", abstract="beta"),
        ],
        topic_filters={
            "topics": [
                {"id": "T10123", "display_name": "HRV", "count": 2},
                {"id": "T10456", "display_name": "Sepsis", "count": 1},
            ],
            "subfields": [], "fields": [], "domains": [],
        },
        embedding_model="placeholder-v1",
    )


def _make_openalex_with_fake_client() -> OpenAlexGatherer:
    """Factory: OpenAlexGatherer pre-wired with a fake client that
    returns two canned works from search_works. Used by the FULL
    parametrization so compliance tests run offline."""
    fake = FakeOpenAlexClient(search_results=[
        canned_openalex_work(doi="10.a/1", openalex_id="WA1", title="Cand A"),
        canned_openalex_work(doi="10.a/2", openalex_id="WA2", title="Cand B"),
    ])
    return OpenAlexGatherer(mailto="test@example.com", client=fake)


FULL_GATHERERS = [FixtureGatherer, _make_openalex_with_fake_client]
STUB_GATHERERS: list = []


def _instance(entry):
    """Entries in FULL_GATHERERS may be either a zero-arg class or a
    factory returning a preconfigured instance. Normalize here."""
    if callable(entry) and not isinstance(entry, type):
        return entry()
    return entry()


# ---------------------------------------------------------------------------
# Non-invoking subset — every gatherer (full impl and stub) must pass these.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gatherer_factory", FULL_GATHERERS + STUB_GATHERERS)
class TestNonInvoking:
    def test_instantiates(self, gatherer_factory):
        g = _instance(gatherer_factory)
        assert hasattr(g, "name") and isinstance(g.name, str)

    def test_runtime_protocol_check(self, gatherer_factory):
        g = _instance(gatherer_factory)
        assert isinstance(g, Gatherer)

    def test_diagnostics_is_dict(self, gatherer_factory):
        g = _instance(gatherer_factory)
        assert isinstance(g.diagnostics(), dict)

    def test_config_round_trip(self, gatherer_factory):
        g = _instance(gatherer_factory)
        cfg = g.config()
        assert isinstance(cfg, dict)
        cls = type(g)
        g2 = cls.from_config(cfg)
        assert g2.name == g.name

    def test_cost_shape(self, gatherer_factory):
        g = _instance(gatherer_factory)
        cost = g.cost()
        assert isinstance(cost, dict)
        assert "wall_seconds" in cost
        assert isinstance(cost["wall_seconds"], float)


# ---------------------------------------------------------------------------
# Full suite — gatherers with working fetch must pass these.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gatherer_factory", FULL_GATHERERS)
class TestFull:
    def test_fetch_returns_papers(self, gatherer_factory):
        g = _instance(gatherer_factory)
        results = g.fetch(_sample_profile(), since="2026-01-01")
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, Paper)

    def test_limit_respected(self, gatherer_factory):
        g = _instance(gatherer_factory)
        profile = _sample_profile()
        all_results = g.fetch(profile, since="2026-01-01")
        if len(all_results) < 2:
            return
        capped = g.fetch(profile, since="2026-01-01", limit=1)
        assert len(capped) <= 1

    def test_cost_after_fetch(self, gatherer_factory):
        g = _instance(gatherer_factory)
        g.fetch(_sample_profile(), since="2026-01-01")
        cost = g.cost()
        assert set(cost.keys()) >= {"wall_seconds"}
        assert cost["wall_seconds"] >= 0.0


# ---------------------------------------------------------------------------
# Stub sentinel — stubs (none in 1B) must raise NotImplementedError.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gatherer_factory", STUB_GATHERERS)
class TestStubsRaise:
    def test_fetch_raises(self, gatherer_factory):
        g = _instance(gatherer_factory)
        with pytest.raises(NotImplementedError):
            g.fetch(_sample_profile(), since="2026-01-01")
