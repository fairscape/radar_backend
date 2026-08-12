"""Every registered reranker must satisfy the Reranker protocol.

Mirrors tests/test_selector_protocol.py — the protocol only earns its
keep if something checks that implementations actually conform.
"""

from __future__ import annotations

import pytest

from rag_lib.paper import Paper
from rag_lib.profile import Profile
from rag_lib.reranker import Reranker
from rag_lib.rerankers import RERANKERS, get_reranker, reranker_from_config


def _paper(oid: str) -> Paper:
    return Paper(doi=None, openalex_id=oid, title=f"Paper {oid}",
                 abstract="some abstract text")


def _ranked(n: int = 3):
    return [
        (0.9 - i / 100, _paper(f"W{i}"), {"score_raw": 0.9 - i / 100})
        for i in range(n)
    ]


@pytest.mark.parametrize("name", sorted(RERANKERS))
def test_registered_reranker_satisfies_protocol(name):
    cls = get_reranker(name)
    assert isinstance(cls(), Reranker), f"{name} does not satisfy Reranker"


@pytest.mark.parametrize("name", sorted(RERANKERS))
def test_config_round_trips_through_from_config(name):
    rr = get_reranker(name)()
    cfg = rr.config()
    assert cfg.get("type") == name, "config must carry its registry key"
    clone = reranker_from_config(cfg)
    assert clone.name == rr.name
    assert clone.config() == cfg


@pytest.mark.parametrize("name", sorted(RERANKERS))
def test_diagnostics_and_cost_shapes(name):
    rr = get_reranker(name)()
    assert "status" in rr.diagnostics()
    assert isinstance(rr.cost().get("wall_seconds"), float)


def test_noop_reranker_is_order_preserving_identity():
    from rag_lib.rerankers import NoopReranker
    ranked = _ranked()
    out = NoopReranker().rerank(ranked, Profile(name="p"))
    assert out == ranked


def test_unknown_reranker_key_raises():
    with pytest.raises(ValueError, match="Unknown reranker"):
        get_reranker("does-not-exist")


def test_reranker_from_config_requires_type():
    with pytest.raises(ValueError, match="missing 'type'"):
        reranker_from_config({"alpha": 0.5})


# ---------------------------------------------------------------------------
# Topic-mode query construction
# ---------------------------------------------------------------------------


def _profile_with_topics(*specs) -> Profile:
    return Profile(
        name="p",
        topic_filters={"topics": [
            {"id": f"T{i}", "display_name": name, "count": 1, "on": on}
            for i, (name, on) in enumerate(specs)
        ]},
    )


def test_topic_mode_skips_disabled_topics():
    """A topic the user switched off must not become a reranker query.

    Deselecting used to delete the entry outright, so this filtering was
    implicit. It is now a flag, which makes the check explicit — and
    load-bearing.
    """
    from rag_lib.rerankers import MedCPTReranker

    rr = MedCPTReranker(query_mode="topic")
    profile = _profile_with_topics(
        ("Diabetes Management", True),
        ("Proteins in Food Systems", False),
        ("Renal Diseases", True),
    )
    assert rr._load_queries(profile) == ["Diabetes Management", "Renal Diseases"]


def test_topic_mode_falls_back_to_titles_when_all_disabled():
    from rag_lib.rerankers import MedCPTReranker

    rr = MedCPTReranker(query_mode="topic")
    profile = _profile_with_topics(("A", False), ("B", False))
    profile.papers = [_paper("W1")]
    assert rr._load_queries(profile) == ["Paper W1"]


def test_topic_mode_respects_max_queries():
    from rag_lib.rerankers import MedCPTReranker

    rr = MedCPTReranker(query_mode="topic", max_queries=2)
    profile = _profile_with_topics(("A", True), ("B", True), ("C", True))
    assert rr._load_queries(profile) == ["A", "B"]
