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


# ---------------------------------------------------------------------------
# Score blending — how much each stage actually moves the ranking
# ---------------------------------------------------------------------------


def _stub_scored(monkeypatch, rr, logits):
    """Run rerank() without the cross-encoder.

    ``_batch_score`` is the only thing in ``rerank`` that needs the model,
    so replacing it exercises aggregation, normalisation and blending —
    none of which had any coverage, and all of which decide the order the
    daily feed is served in.
    """
    monkeypatch.setattr(rr, "_batch_score", lambda queries, articles: list(logits))


def test_batch_span_maps_a_batch_onto_the_unit_interval():
    from rag_lib.rerankers.medcpt import _batch_span

    lo, span = _batch_span([-16.0, -12.0, -14.0])
    assert lo == -16.0 and span == 4.0
    assert [(v - lo) / span for v in (-16.0, -12.0)] == [0.0, 1.0]


def test_batch_span_of_a_flat_batch_does_not_divide_by_zero():
    """A stage with no spread has expressed no preference; every entry
    normalises to 0.0 and the term drops out of the blend."""
    from rag_lib.rerankers.medcpt import _batch_span

    lo, span = _batch_span([0.7, 0.7, 0.7])
    assert span == 1.0
    assert [(v - lo) / span for v in (0.7, 0.7)] == [0.0, 0.0]


def test_both_stages_are_normalised_over_the_same_batch(monkeypatch):
    """The fix this test exists for.

    The selector used to be mapped with a fixed (cos + 1) / 2 while the
    reranker was min-maxed. Real cosines within a batch span a fraction
    of a point, min-max spans exactly one, so alpha and beta weighted
    quantities of wildly different size — a nominal 40:60 measured as an
    effective 4:96. Both stages now span [0, 1].
    """
    from rag_lib.rerankers import MedCPTReranker

    rr = MedCPTReranker(query_mode="topic", alpha=0.4, beta=0.6)
    _stub_scored(monkeypatch, rr, [-16.0, -14.0, -12.0])
    ranked = [
        (0.90, _paper("W0"), {"score_raw": 0.90}),
        (0.88, _paper("W1"), {"score_raw": 0.88}),
        (0.86, _paper("W2"), {"score_raw": 0.86}),
    ]
    out = rr.rerank(ranked, _profile_with_topics(("Diabetes", True)))

    sel = [b["score_selector_norm"] for _, _, b in out]
    rrn = [b["score_reranker_norm"] for _, _, b in out]
    assert (min(sel), max(sel)) == (0.0, 1.0)
    assert (min(rrn), max(rrn)) == (0.0, 1.0)


def test_the_weights_now_split_influence_as_written(monkeypatch):
    """Weight times spread is what moves a ranking. With equal spreads
    that reduces to the weights themselves."""
    from rag_lib.rerankers import MedCPTReranker

    rr = MedCPTReranker(query_mode="topic", alpha=0.4, beta=0.6)
    _stub_scored(monkeypatch, rr, [-16.0, -14.0, -12.0])
    ranked = [
        (0.90, _paper("W0"), {"score_raw": 0.90}),
        (0.88, _paper("W1"), {"score_raw": 0.88}),
        (0.86, _paper("W2"), {"score_raw": 0.86}),
    ]
    out = rr.rerank(ranked, _profile_with_topics(("Diabetes", True)))

    sel = [b["score_selector_norm"] for _, _, b in out]
    rrn = [b["score_reranker_norm"] for _, _, b in out]
    sel_influence = 0.4 * (max(sel) - min(sel))
    rr_influence = 0.6 * (max(rrn) - min(rrn))
    assert sel_influence / (sel_influence + rr_influence) == pytest.approx(0.4)


def test_blended_score_is_the_weighted_sum_of_the_two_norms(monkeypatch):
    from rag_lib.rerankers import MedCPTReranker

    rr = MedCPTReranker(query_mode="topic", alpha=0.4, beta=0.6)
    _stub_scored(monkeypatch, rr, [-16.0, -14.0, -12.0])
    ranked = [
        (0.90, _paper("W0"), {"score_raw": 0.90}),
        (0.88, _paper("W1"), {"score_raw": 0.88}),
        (0.86, _paper("W2"), {"score_raw": 0.86}),
    ]
    out = rr.rerank(ranked, _profile_with_topics(("Diabetes", True)))

    for score, _, b in out:
        expected = 0.4 * b["score_selector_norm"] + 0.6 * b["score_reranker_norm"]
        assert score == pytest.approx(expected)
        assert b["score_blended"] == pytest.approx(expected)


def test_a_selector_with_no_spread_cannot_move_the_ranking(monkeypatch):
    """Identical selector scores must leave the order to the reranker
    rather than blowing up on a zero divisor."""
    from rag_lib.rerankers import MedCPTReranker

    rr = MedCPTReranker(query_mode="topic", alpha=0.4, beta=0.6)
    _stub_scored(monkeypatch, rr, [-16.0, -12.0, -14.0])
    ranked = [
        (0.9, _paper("W0"), {"score_raw": 0.9}),
        (0.9, _paper("W1"), {"score_raw": 0.9}),
        (0.9, _paper("W2"), {"score_raw": 0.9}),
    ]
    out = rr.rerank(ranked, _profile_with_topics(("Diabetes", True)))

    assert all(b["score_selector_norm"] == 0.0 for _, _, b in out)
    assert [p.openalex_id for _, p, _ in out] == ["W1", "W2", "W0"]


def test_title_mode_uses_seed_titles():
    """Titles are a mode of their own, not only the fallback.

    A topic name is a taxonomy label — "Diabetes Management and
    Research" — and several of them arrive as separate queries that get
    averaged, so a candidate matching one scores like a candidate
    matching all. A seed title states the whole request in one string.
    """
    from rag_lib.rerankers import MedCPTReranker

    rr = MedCPTReranker(query_mode="title")
    profile = _profile_with_topics(("Diabetes Management", True))
    profile.papers = [_paper("W1"), _paper("W2")]
    assert rr._load_queries(profile) == ["Paper W1", "Paper W2"]


def test_title_mode_ignores_topics_entirely():
    """Including ones the user left switched on — that is the point of
    asking for titles."""
    from rag_lib.rerankers import MedCPTReranker

    rr = MedCPTReranker(query_mode="title")
    profile = _profile_with_topics(("Plant Reproductive Biology", True))
    profile.papers = [_paper("W1")]
    assert rr._load_queries(profile) == ["Paper W1"]


def test_title_mode_respects_max_queries():
    from rag_lib.rerankers import MedCPTReranker

    rr = MedCPTReranker(query_mode="title", max_queries=2)
    profile = _profile_with_topics(("T", True))
    profile.papers = [_paper("W1"), _paper("W2"), _paper("W3")]
    assert rr._load_queries(profile) == ["Paper W1", "Paper W2"]


def test_title_mode_round_trips_through_config():
    from rag_lib.rerankers import MedCPTReranker, reranker_from_config

    rr = MedCPTReranker(query_mode="title")
    assert reranker_from_config(rr.config()).query_mode == "title"


def test_title_mode_pairs_titles_with_candidate_title_and_abstract(monkeypatch):
    """The candidate side must stay short too — only "article" mode
    hands the cross-encoder a whole document."""
    from rag_lib.rerankers import MedCPTReranker

    seen: dict = {}
    rr = MedCPTReranker(query_mode="title")

    def _capture(queries, articles):
        seen["queries"], seen["articles"] = queries, articles
        return [0.0] * (len(queries) * len(articles))

    monkeypatch.setattr(rr, "_batch_score", _capture)
    profile = _profile_with_topics(("Diabetes", True))
    profile.papers = [_paper("SEED")]
    rr.rerank([(0.9, _paper("W0"), {"score_raw": 0.9})], profile)

    assert seen["queries"] == ["Paper SEED"]
    assert seen["articles"] == ["Paper W0. some abstract text"]
