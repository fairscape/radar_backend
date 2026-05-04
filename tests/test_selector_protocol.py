"""Protocol compliance test — the unambiguous definition of "satisfies
the Selector protocol."

Three parametrizations:

- FULL_SELECTORS runs every test. CentroidSelector joined here in
  Phase 1B; FixtureSelector has been here since 1A.
- STUB_SELECTORS runs only the non-invoking subset. LLMRetrainSelector
  is the remaining stub — Track 2 owns the body.
- TestStubsRaise ensures stubs actively raise NotImplementedError.
"""

from __future__ import annotations

import pytest

from rag_lib.embed import build_embedding_input
from rag_lib.embedders import placeholder_embed
from rag_lib.paper import Paper
from rag_lib.profile import Profile
from rag_lib.selector import Selector
from rag_lib.selectors.centroid import CentroidSelector
from rag_lib.selectors.full_text import FullTextSelector
from rag_lib.selectors.llm_retrain import LLMRetrainSelector
from rag_lib.selectors.max_seed import MaxSeedSelector
from tests.fixture_selector import FixtureSelector


def _embed(p: Paper) -> list[float]:
    return placeholder_embed(build_embedding_input(p))


def _sample_profile() -> Profile:
    papers = [
        Paper(doi="10.1/a", openalex_id="W1", title="Neonatal HRV A",
              abstract="alpha beta gamma"),
        Paper(doi="10.1/b", openalex_id="W2", title="Neonatal HRV B",
              abstract="beta gamma delta"),
        Paper(doi="10.1/c", openalex_id="W3", title="Neonatal sepsis A",
              abstract="gamma delta epsilon"),
        Paper(doi="10.1/d", openalex_id="W4", title="Neonatal sepsis B",
              abstract="delta epsilon zeta"),
    ]
    # Give each seed paper a real placeholder embedding so CentroidSelector
    # has consistent-dim vectors to average.
    for p in papers:
        p.embeddings["placeholder-v1"] = _embed(p)
    return Profile(name="neonatal", papers=papers,
                   embedding_model="placeholder-v1")


def _sample_candidates() -> list[Paper]:
    return [
        Paper(doi="10.1/e", openalex_id="W5", title="E", abstract="alpha gamma"),
        Paper(doi="10.1/f", openalex_id="W6", title="F", abstract="epsilon"),
        Paper(doi="10.1/g", openalex_id="W7", title="G", abstract="alpha beta delta"),
        Paper(doi="10.1/h", openalex_id="W8", title="H", abstract="zeta eta"),
    ]


FULL_SELECTORS = [FixtureSelector, CentroidSelector, MaxSeedSelector,
                  FullTextSelector]
STUB_SELECTORS = [LLMRetrainSelector]


# ---------------------------------------------------------------------------
# Non-invoking subset — every selector (full impl and stub) must pass these.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("selector_cls", FULL_SELECTORS + STUB_SELECTORS)
class TestNonInvoking:
    def test_instantiates(self, selector_cls):
        s = selector_cls()
        assert hasattr(s, "name") and isinstance(s.name, str)

    def test_runtime_protocol_check(self, selector_cls):
        s = selector_cls()
        assert isinstance(s, Selector)

    def test_diagnostics_is_dict(self, selector_cls):
        s = selector_cls()
        assert isinstance(s.diagnostics(), dict)

    def test_config_round_trip(self, selector_cls):
        s = selector_cls()
        cfg = s.config()
        assert isinstance(cfg, dict)
        s2 = selector_cls.from_config(cfg)
        assert s2.name == s.name

    def test_cost_shape(self, selector_cls):
        s = selector_cls()
        cost = s.cost()
        assert isinstance(cost, dict)
        assert "wall_seconds" in cost
        assert isinstance(cost["wall_seconds"], float)


# ---------------------------------------------------------------------------
# Full suite — selectors with working fit/select must pass these.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("selector_cls", FULL_SELECTORS)
class TestFull:
    def test_fit_then_select(self, selector_cls):
        s = selector_cls()
        profile = _sample_profile()
        s.fit(profile)
        candidates = _sample_candidates()
        results = s.select(candidates, profile)
        assert len(results) == len(candidates)
        assert all(isinstance(r[0], float) for r in results)
        scores = [r[0] for r in results]
        assert scores == sorted(scores, reverse=True)
        # Phase 3: every entry now carries a (possibly empty) breakdown dict.
        for entry in results:
            assert len(entry) == 3
            assert isinstance(entry[2], dict)

    def test_scores_in_unit_interval(self, selector_cls):
        s = selector_cls()
        profile = _sample_profile()
        s.fit(profile)
        results = s.select(_sample_candidates(), profile)
        tol = 1e-6
        for entry in results:
            score = entry[0]
            assert -tol <= score <= 1.0 + tol

    def test_threshold_filters(self, selector_cls):
        s = selector_cls()
        profile = _sample_profile()
        s.fit(profile)
        all_results = s.select(_sample_candidates(), profile)
        if not all_results:
            return
        thresh = all_results[len(all_results) // 2][0]
        filtered = s.select(_sample_candidates(), profile, threshold=thresh)
        assert all(r[0] >= thresh for r in filtered)
        assert len(filtered) <= len(all_results)

    def test_diagnostics_after_fit(self, selector_cls):
        s = selector_cls()
        profile = _sample_profile()
        s.fit(profile)
        assert isinstance(s.diagnostics(), dict)

    def test_cost_after_fit_and_select(self, selector_cls):
        s = selector_cls()
        profile = _sample_profile()
        s.fit(profile)
        s.select(_sample_candidates(), profile)
        cost = s.cost()
        assert set(cost.keys()) >= {"wall_seconds"}
        assert cost["wall_seconds"] >= 0.0

    def test_select_returns_paper_objects(self, selector_cls):
        s = selector_cls()
        profile = _sample_profile()
        s.fit(profile)
        results = s.select(_sample_candidates(), profile)
        for entry in results:
            assert isinstance(entry[0], float)
            assert isinstance(entry[1], Paper)
            assert isinstance(entry[2], dict)


# ---------------------------------------------------------------------------
# Stub sentinel — stubs must raise NotImplementedError.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("selector_cls", STUB_SELECTORS)
class TestStubsRaise:
    def test_fit_raises(self, selector_cls):
        s = selector_cls()
        profile = _sample_profile()
        with pytest.raises(NotImplementedError):
            s.fit(profile)

    def test_select_raises(self, selector_cls):
        s = selector_cls()
        profile = _sample_profile()
        with pytest.raises(NotImplementedError):
            s.select(_sample_candidates(), profile)
