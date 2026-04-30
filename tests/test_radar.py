"""radar.dry_run and centroid_drift tests."""

from __future__ import annotations

import numpy as np

from rag_lib.embed import build_embedding_input
from rag_lib.embedders import placeholder_embed
from rag_lib.paper import Paper
from rag_lib.profile import Profile
from rag_lib.radar import centroid_drift, dry_run
from rag_lib.selectors.centroid import CentroidSelector
from tests.fixture_gatherer import FixtureGatherer


def _profile() -> Profile:
    papers = [
        Paper(doi="10.1/a", openalex_id="W1", title="A", abstract="alpha"),
        Paper(doi="10.1/b", openalex_id="W2", title="B", abstract="beta"),
    ]
    for p in papers:
        p.embeddings["placeholder-v1"] = placeholder_embed(build_embedding_input(p))
    return Profile(
        name="x", papers=papers,
        topic_filters={"topics": [{"id": "T1", "display_name": "t", "count": 2}],
                       "subfields": [], "fields": [], "domains": []},
        embedding_model="placeholder-v1",
    )


def test_dry_run_returns_expected_shape():
    profile = _profile()
    sel = CentroidSelector()
    sel.fit(profile)
    gatherer = FixtureGatherer(seed=42, n_candidates=6)
    report = dry_run(sel, profile, gatherer, thresholds=[0.3, 0.5, 0.9])

    assert "since" in report
    assert report["fetched"] == 6
    assert "0.30" in report["results"]
    assert "0.50" in report["results"]
    assert "0.90" in report["results"]
    for r in report["results"].values():
        assert "count" in r and isinstance(r["count"], int)
        assert "top_titles" in r and isinstance(r["top_titles"], list)


def test_dry_run_threshold_counts_are_monotone():
    """Higher threshold must never pass more candidates than a lower one."""
    profile = _profile()
    sel = CentroidSelector()
    sel.fit(profile)
    gatherer = FixtureGatherer(seed=7, n_candidates=20)
    report = dry_run(sel, profile, gatherer, thresholds=[0.10, 0.30, 0.60, 0.90])
    counts = [report["results"][k]["count"] for k in sorted(report["results"].keys())]
    assert counts == sorted(counts, reverse=True)


def test_dry_run_records_both_costs():
    profile = _profile()
    sel = CentroidSelector()
    sel.fit(profile)
    gatherer = FixtureGatherer(seed=3, n_candidates=4)
    report = dry_run(sel, profile, gatherer)
    assert "wall_seconds" in report["selector_cost"]
    assert "wall_seconds" in report["gatherer_cost"]
    assert "api_calls" in report["gatherer_cost"]


def test_centroid_drift_identical_vectors_is_one():
    v = [0.1, 0.2, 0.3, 0.4]
    assert abs(centroid_drift(v, v) - 1.0) < 1e-9


def test_centroid_drift_antipodal_vectors_is_zero():
    v1 = [1.0, 0.0]
    v2 = [-1.0, 0.0]
    assert abs(centroid_drift(v1, v2) - 0.0) < 1e-9


def test_centroid_drift_orthogonal_vectors_is_half():
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    assert abs(centroid_drift(v1, v2) - 0.5) < 1e-9
