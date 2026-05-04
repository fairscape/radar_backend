"""FullTextSelector — corpus-aware full-text scoring behavior.

Covers:
  - fit produces a unit-norm weighted centroid from seed chunks.
  - Seeds with empty body fall back to a single abstract chunk and
    diagnostics report the coverage gap.
  - select with fetch_missing_pdfs=False NEVER calls fetch_pdf_text;
    empty-body candidates are tagged fallback="abstract_only".
  - select with fetch_missing_pdfs=True calls fetch_pdf_text only for
    candidates that have an empty body AND a pdf_url.
  - config / from_config round-trips fetch_missing_pdfs and fitted
    state; the rehydrated selector produces identical scores without
    re-fitting.
"""

from __future__ import annotations

import numpy as np
import pytest

from rag_lib.embed import build_embedding_input
from rag_lib.embedders import placeholder_embed
from rag_lib.paper import Paper
from rag_lib.profile import Profile
from rag_lib.selectors.full_text import FullTextSelector


def _seed_paper(doi: str, title: str, abstract: str, body_text: str = "") -> Paper:
    p = Paper(doi=doi, openalex_id=None, title=title, abstract=abstract,
              body_text=body_text)
    p.embeddings["placeholder-v1"] = placeholder_embed(build_embedding_input(p))
    return p


def _profile_with_bodies() -> Profile:
    body_a = "alpha alpha methods results conclusions one two three " * 30
    body_b = "beta beta methods results conclusions four five six " * 30
    body_c = "gamma gamma methods results conclusions seven eight nine " * 30
    return Profile(
        name="x",
        papers=[
            _seed_paper("10.1/a", "A", "alpha", body_text=body_a),
            _seed_paper("10.1/b", "B", "beta", body_text=body_b),
            _seed_paper("10.1/c", "C", "gamma", body_text=body_c),
        ],
        embedding_model="placeholder-v1",
    )


def _profile_no_bodies() -> Profile:
    return Profile(
        name="x",
        papers=[
            _seed_paper("10.1/a", "A", "alpha"),
            _seed_paper("10.1/b", "B", "beta"),
            _seed_paper("10.1/c", "C", "gamma"),
        ],
        embedding_model="placeholder-v1",
    )


# ----------------------------------------------------------------------


def test_fit_produces_unit_norm_weighted_centroid():
    sel = FullTextSelector()
    sel.fit(_profile_with_bodies())
    cfg = sel.config()
    assert cfg["weighted_centroid"] is not None
    centroid = np.asarray(cfg["weighted_centroid"], dtype=float)
    assert np.linalg.norm(centroid) == pytest.approx(1.0, abs=1e-6)
    # Each long seed body chunks into multiple windows, plus one chunk
    # per shorter seed → at minimum one chunk per seed paper.
    assert len(cfg["seed_chunk_matrix"]) >= 3


def test_fit_falls_back_to_abstract_when_body_empty():
    sel = FullTextSelector()
    sel.fit(_profile_no_bodies())
    diag = sel.diagnostics()
    assert diag["status"] == "fit"
    assert diag["n_seed_papers"] == 3
    assert diag["n_seed_papers_no_body"] == 3
    assert diag["n_seed_chunks"] == 3


def test_select_without_fetch_flag_never_fetches(monkeypatch):
    calls = {"n": 0}

    def fake_fetch(url, *, timeout=30.0):
        calls["n"] += 1
        return "should not be called"

    monkeypatch.setattr(
        "rag_lib.selectors.full_text.fetch_pdf_text", fake_fetch
    )

    profile = _profile_with_bodies()
    sel = FullTextSelector(fetch_missing_pdfs=False)
    sel.fit(profile)
    candidates = [Paper(
        doi="10.9/x", openalex_id="WX", title="X",
        abstract="alpha methods",
        pdf_url="https://example.org/paper.pdf",
    )]
    results = sel.select(candidates, profile)

    assert calls["n"] == 0
    _, _, breakdown = results[0]
    assert breakdown["fallback"] == "abstract_only"
    assert breakdown["fetched_pdf"] is False


def test_select_with_fetch_flag_fetches_only_when_needed(monkeypatch):
    calls = {"urls": []}

    def fake_fetch(url, *, timeout=30.0):
        calls["urls"].append(url)
        return "fetched body content with words " * 60

    monkeypatch.setattr(
        "rag_lib.selectors.full_text.fetch_pdf_text", fake_fetch
    )

    profile = _profile_with_bodies()
    sel = FullTextSelector(fetch_missing_pdfs=True)
    sel.fit(profile)
    candidates = [
        Paper(doi="10.9/a", openalex_id="WA", title="A", abstract="x",
              pdf_url="https://example.org/a.pdf"),
        Paper(doi="10.9/b", openalex_id="WB", title="B", abstract="y",
              body_text="local body text " * 60,
              pdf_url="https://example.org/b.pdf"),
        Paper(doi="10.9/c", openalex_id="WC", title="C", abstract="z"),
    ]
    results = sel.select(candidates, profile)

    assert calls["urls"] == ["https://example.org/a.pdf"]
    by_doi = {r[1].doi: r[2] for r in results}
    assert by_doi["10.9/a"]["fetched_pdf"] is True
    assert by_doi["10.9/a"]["fallback"] is None
    assert by_doi["10.9/b"]["fetched_pdf"] is False
    assert by_doi["10.9/b"]["fallback"] is None
    assert by_doi["10.9/c"]["fetched_pdf"] is False
    assert by_doi["10.9/c"]["fallback"] == "abstract_only"


def test_fetch_pdf_text_failure_falls_back_gracefully(monkeypatch):
    def boom(url, *, timeout=30.0):
        raise RuntimeError("network down")

    monkeypatch.setattr(
        "rag_lib.selectors.full_text.fetch_pdf_text", boom
    )

    profile = _profile_with_bodies()
    sel = FullTextSelector(fetch_missing_pdfs=True)
    sel.fit(profile)
    candidates = [Paper(
        doi="10.9/x", openalex_id="WX", title="X", abstract="alpha",
        pdf_url="https://example.org/p.pdf",
    )]
    results = sel.select(candidates, profile)

    _, _, breakdown = results[0]
    assert breakdown["fetched_pdf"] is False
    assert breakdown["fallback"] == "abstract_only"


def test_select_before_fit_raises():
    sel = FullTextSelector()
    with pytest.raises(RuntimeError, match="before fit"):
        sel.select(
            [Paper(doi="x", openalex_id=None, title="x")],
            _profile_with_bodies(),
        )


def test_config_round_trip_preserves_state_and_scores():
    profile = _profile_with_bodies()
    sel = FullTextSelector(fetch_missing_pdfs=True, alpha=0.6, top_k_maxsim=2)
    sel.fit(profile)
    candidates = [
        Paper(doi=f"10.9/{i}", openalex_id=f"W{i}", title=f"T{i}",
              abstract=f"alpha methods results {i}")
        for i in range(3)
    ]
    expected = sel.select(candidates, profile)

    cfg = sel.config()
    sel2 = FullTextSelector.from_config(cfg)
    assert sel2.fetch_missing_pdfs is True
    assert sel2.alpha == 0.6
    assert sel2.top_k_maxsim == 2
    assert sel2.diagnostics()["status"] == "fit"

    actual = sel2.select(candidates, profile)
    assert len(actual) == len(expected)
    for (s_a, _, b_a), (s_e, _, b_e) in zip(actual, expected):
        assert s_a == pytest.approx(s_e, abs=1e-9)
        assert b_a["score_pooled"] == pytest.approx(b_e["score_pooled"], abs=1e-9)
        assert b_a["score_maxsim"] == pytest.approx(b_e["score_maxsim"], abs=1e-9)


def test_score_in_unit_interval_and_breakdown_shape():
    profile = _profile_with_bodies()
    sel = FullTextSelector()
    sel.fit(profile)
    candidates = [
        Paper(doi="10.9/x", openalex_id="WX", title="X", abstract="alpha"),
    ]
    score, _, breakdown = sel.select(candidates, profile)[0]
    assert -1.0 <= score <= 1.0
    expected_keys = {
        "score_raw", "score_pooled", "score_maxsim", "n_chunks",
        "fetched_pdf", "fallback", "score_pct",
    }
    assert expected_keys <= set(breakdown.keys())


def test_threshold_filters_at_select_time():
    profile = _profile_with_bodies()
    sel = FullTextSelector()
    sel.fit(profile)
    candidates = [Paper(doi=f"10.9/{i}", openalex_id=f"W{i}",
                        title=f"cand-{i}", abstract=f"text {i}")
                  for i in range(5)]
    all_results = sel.select(candidates, profile)
    mid = all_results[len(all_results) // 2][0]
    filtered = sel.select(candidates, profile, threshold=mid)
    assert all(r[0] >= mid for r in filtered)
    assert len(filtered) <= len(all_results)


def test_cost_reports_pdf_fetches(monkeypatch):
    calls = []

    def fake_fetch(url, *, timeout=30.0):
        calls.append(url)
        return "fetched body words " * 60

    monkeypatch.setattr(
        "rag_lib.selectors.full_text.fetch_pdf_text", fake_fetch
    )

    profile = _profile_with_bodies()
    sel = FullTextSelector(fetch_missing_pdfs=True)
    sel.fit(profile)
    candidates = [
        Paper(doi=f"10.9/{i}", openalex_id=f"W{i}", title=f"T{i}",
              abstract="x", pdf_url=f"https://example.org/{i}.pdf")
        for i in range(2)
    ]
    sel.select(candidates, profile)
    cost = sel.cost()
    assert cost["pdf_fetches"] == 2
    assert cost["chunks_embedded"] > 0
    assert cost["wall_seconds"] >= 0.0


def test_registry_round_trip():
    """Selector registers under "full_text" and rehydrates via
    selector_from_config."""
    from rag_lib.selectors import selector_from_config, SELECTORS

    assert SELECTORS["full_text"] is FullTextSelector

    sel = FullTextSelector(fetch_missing_pdfs=True, alpha=0.7)
    sel.fit(_profile_with_bodies())
    cfg = sel.config()
    assert cfg["type"] == "full_text"
    sel2 = selector_from_config(cfg)
    assert isinstance(sel2, FullTextSelector)
    assert sel2.fetch_missing_pdfs is True
    assert sel2.alpha == 0.7
