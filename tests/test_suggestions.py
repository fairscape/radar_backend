"""Suggested interests: groups found in a researcher's stored embeddings.

The vectors are built by hand — tight clusters around two or three
random directions (within-topic cosine ~0.92, across ~0, which is
wider apart than the measured SPECTER2 scale but tests the cut, not
the scale) — because the placeholder embedder hashes titles
to noise and the point of these tests is the cut, not the embedder.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import numpy as np
import pytest

from rag_lib.api import settings as settings_module
from rag_lib.api.services.suggestions import (
    average_linkage_tree,
    choose_groups,
    cut_tree,
    suggest_interests,
)
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import (
    embeddings as embeddings_repo,
    papers as papers_repo,
    researchers as researchers_repo,
    users as users_repo,
)


MODEL = "placeholder-v1"


def _clustered(rng: np.random.Generator, sizes: list[int], *, dim: int = 256, spread: float = 0.3) -> tuple[np.ndarray, list[int]]:
    """Unit vectors in ``len(sizes)`` groups; returns them with group labels."""
    centres = rng.standard_normal((len(sizes), dim))
    vecs, labels = [], []
    for g, n in enumerate(sizes):
        for _ in range(n):
            v = centres[g] + spread * rng.standard_normal(dim)
            vecs.append(v / np.linalg.norm(v))
            labels.append(g)
    return np.array(vecs), labels


# ---------------------------------------------------------------------------
# The clustering itself
# ---------------------------------------------------------------------------


def test_tree_and_cut_recover_three_planted_groups():
    rng = np.random.default_rng(1)
    vecs, labels = _clustered(rng, [6, 5, 4])
    sim = vecs @ vecs.T
    merges = average_linkage_tree(sim)
    assert len(merges) == len(vecs) - 1
    groups = cut_tree(merges, len(vecs), 3)
    assert sorted(len(g) for g in groups) == [4, 5, 6]
    for g in groups:
        assert len({labels[i] for i in g}) == 1


def test_choose_groups_stops_at_three_and_leaves_the_rest_out():
    rng = np.random.default_rng(2)
    vecs, labels = _clustered(rng, [6, 5, 4, 3])
    sim = vecs @ vecs.T
    merges = average_linkage_tree(sim)
    groups, note = choose_groups(sim, merges)
    # Four planted groups, but at most three suggestions, largest first;
    # the smallest planted group is left out rather than merged in.
    assert note is None
    assert [len(g) for g in groups] == [6, 5, 4]
    for g in groups:
        assert len({labels[i] for i in g}) == 1


def test_choose_groups_skips_outliers():
    """Three one-off papers in other fields: two groups come back, and
    any stray that a cut still attaches to a group is at most one per
    group (the service then reports it as loose)."""
    rng = np.random.default_rng(5)
    vecs, labels = _clustered(rng, [6, 6, 1, 1, 1])
    sim = vecs @ vecs.T
    groups, note = choose_groups(sim, average_linkage_tree(sim))
    assert note is None
    assert len(groups) == 2
    majorities = set()
    strays = 0
    for g in groups:
        counts = {lab: sum(1 for i in g if labels[i] == lab) for lab in set(labels[i] for i in g)}
        major = max(counts, key=counts.get)  # type: ignore[arg-type]
        majorities.add(major)
        assert counts[major] == 6
        strays += len(g) - 6
    assert majorities == {0, 1}
    # The median is robust, so a cut can still carry a stray or two; the
    # service reports those as loose (next test) rather than as seeds.
    assert strays <= 3


def test_one_focused_topic_is_one_suggestion():
    rng = np.random.default_rng(2)
    one_topic, _ = _clustered(rng, [12], spread=0.2)
    sim1 = one_topic @ one_topic.T
    groups, note = choose_groups(sim1, average_linkage_tree(sim1))
    assert len(groups) == 1 and len(groups[0]) == 12
    assert "one focused topic" in note


def test_choose_groups_keeps_small_sets_whole():
    rng = np.random.default_rng(3)
    vecs, _ = _clustered(rng, [3, 3])
    sim = vecs @ vecs.T
    groups, note = choose_groups(sim, average_linkage_tree(sim))
    assert len(groups) == 1 and "Fewer than" in note


def test_tree_runs_fast_enough_on_the_cap():
    import time
    rng = np.random.default_rng(4)
    vecs, _ = _clustered(rng, [200, 200])
    sim = vecs @ vecs.T
    t0 = time.monotonic()
    average_linkage_tree(sim)
    assert time.monotonic() - t0 < 10.0


# ---------------------------------------------------------------------------
# Through the service and the route
# ---------------------------------------------------------------------------


@pytest.fixture()
def env(tmp_path, monkeypatch):
    db = tmp_path / "radar.db"
    monkeypatch.setenv("RADAR_DB_PATH", str(db))
    monkeypatch.setenv("RADAR_VAULT_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("RADAR_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("RADAR_LOG_JSON", "false")
    monkeypatch.setenv("RADAR_SCHEDULER_ENABLED", "false")
    monkeypatch.setenv("RADAR_DEFAULT_EMBEDDING_MODEL", MODEL)
    monkeypatch.setenv("RADAR_UMLS_ENABLED", "false")
    settings_module.get_settings.cache_clear()
    conn = connect(db)
    apply_migrations(conn)
    users_repo.upsert(conn, "demo@example.com")
    conn.close()
    yield db
    settings_module.get_settings.cache_clear()


def _seed_researcher(env, sizes: list[int], *, topics=True, embed=True) -> int:
    rng = np.random.default_rng(7)
    vecs, labels = _clustered(rng, sizes)
    names = ["Neonatal monitoring", "Genomic intervals", "Data provenance", "Stray topic A", "Stray topic B"]
    conn = connect(env)
    try:
        rid = researchers_repo.upsert(conn, user_id=1, source="orcid", key="0000-0002-1825-0097", name="Test Person")
        for i, (v, g) in enumerate(zip(vecs, labels)):
            oid = f"https://openalex.org/W{i}"
            paper = {"openalex_id": oid, "title": f"{names[g]} paper {i}", "source": "prosopia", "year": 2020 + (i % 5)}
            if topics:
                paper["primary_topic"] = {"id": f"T{g}", "display_name": names[g]}
                paper["topics"] = [{"id": "T-shared", "display_name": "Biomedical informatics"}]
            papers_repo.upsert(conn, paper)
            if embed:
                embeddings_repo.upsert(conn, oid, MODEL, v)
            researchers_repo.attach_paper(conn, rid, oid, resolved_by="work_id")
        researchers_repo.mark_imported(conn, rid, run_id=None)
        return rid
    finally:
        conn.close()


def _request(method, path, **kwargs) -> httpx.Response:
    from rag_lib.api.app import create_app

    app = create_app()

    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(method, path, **kwargs)

    return asyncio.run(_run())


def test_suggestions_name_groups_by_their_distinctive_topic(env):
    rid = _seed_researcher(env, [6, 5, 4])
    resp = _request("GET", f"/api/researchers/{rid}/suggestions")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["researcher_id"] == rid
    assert body["n_papers"] == 15 and body["n_embedded"] == 15
    assert body["note"] is None
    names = [s["name"] for s in body["suggestions"]]
    assert sorted(names) == ["Data provenance", "Genomic intervals", "Neonatal monitoring"]
    # The shared topic never names a group, and sizes match the plant.
    assert [len(s["paper_ids"]) + len(s["loose_ids"]) for s in body["suggestions"]] == [6, 5, 4]
    assert body["n_grouped"] == 15
    for s in body["suggestions"]:
        assert s["label"] in ("focused", "broad", "mixed")
        assert s["agreement"] is not None
        assert s["seed_titles"]
        assert all(t["name"] != "Biomedical informatics" for t in s["topics"][:1])


def test_strays_are_reported_loose_not_hidden(env):
    rid = _seed_researcher(env, [6, 6, 1, 1, 1])
    body = _request("GET", f"/api/researchers/{rid}/suggestions").json()
    assert len(body["suggestions"]) == 2
    for s in body["suggestions"]:
        assert len(s["paper_ids"]) == 6
    strays = {f"https://openalex.org/W{i}" for i in (12, 13, 14)}
    loose = {i for s in body["suggestions"] for i in s["loose_ids"]}
    assert loose <= strays
    assert body["n_grouped"] == 12 + len(loose)


def test_a_small_profile_gets_one_suggestion_with_a_reason(env):
    rid = _seed_researcher(env, [3, 3])
    body = _request("GET", f"/api/researchers/{rid}/suggestions").json()
    assert len(body["suggestions"]) == 1
    assert len(body["suggestions"][0]["paper_ids"]) + len(body["suggestions"][0]["loose_ids"]) == 6
    assert "Fewer than" in body["note"]


def test_nothing_embedded_is_a_note_not_an_error(env):
    rid = _seed_researcher(env, [4, 4], embed=False)
    body = _request("GET", f"/api/researchers/{rid}/suggestions").json()
    assert body["suggestions"] == []
    assert body["n_embedded"] == 0
    assert "embedded" in body["note"]


def test_untitled_topics_fall_back_to_a_title(env):
    rid = _seed_researcher(env, [5, 5], topics=False)
    body = _request("GET", f"/api/researchers/{rid}/suggestions").json()
    assert len(body["suggestions"]) == 2
    assert all(s["name"].endswith(tuple(f"paper {i}" for i in range(10))) for s in body["suggestions"])


def test_suggestions_are_per_user(env):
    rid = _seed_researcher(env, [4, 4])
    resp = _request("GET", f"/api/researchers/{rid}/suggestions", headers={"X-User-Email": "other@example.com"})
    assert resp.status_code == 404


def test_suggested_ids_build_an_interest(env):
    rid = _seed_researcher(env, [6, 5, 4])
    body = _request("GET", f"/api/researchers/{rid}/suggestions").json()
    first = body["suggestions"][0]
    resp = _request("POST", f"/api/researchers/{rid}/interests", json={"name": first["name"], "openalex_ids": first["paper_ids"]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["n_seeds"] == len(first["paper_ids"])
