"""Wizard endpoints — Phase 11 end-to-end.

Walks the four-step wizard against a tmp SQLite seeded with 8 fake
seed papers (their topics overlap so ``aggregate_topic_filters`` has
something to return). The dry-run step injects a ``FixtureGatherer``
via ``app.dependency_overrides`` so no network call is made.
"""

from __future__ import annotations

import asyncio
import sqlite3

import httpx
import numpy as np
import pytest

from rag_lib.api import settings as settings_module
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import (
    embeddings as embeddings_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    users as users_repo,
)


def _seed_user_papers(db_path) -> int:
    """Create one user + 8 papers (all topic T100) with embeddings.

    Returns the user_id. The embeddings cluster around a base direction
    so coherence comes out tight; the topic overlap is total so the
    aggregator returns a single dominant topic the wizard can show.
    """
    conn = connect(db_path)
    apply_migrations(conn)
    user = users_repo.upsert(conn, "demo@example.com")
    user_id = int(user["id"])

    # Match the placeholder embedder's 128-dim output so the dry-run
    # selector (which embeds candidates fresh via placeholder_embed) gets
    # vectors of the same dimensionality as the seeds.
    rng = np.random.default_rng(7)
    base = rng.standard_normal(128).astype("float32")
    base /= np.linalg.norm(base) + 1e-9

    for i in range(8):
        oa = f"W{3000000 + i}"
        papers_repo.upsert(conn, {
            "openalex_id": oa,
            "title": f"Neonatal vitals study {i}",
            "abstract": "Abstract describing neonatal vitals work.",
            "year": 2024,
            "venue": "Pediatrics",
            "source": "user_pdf",
            "primary_topic": {
                "id": "T100",
                "display_name": "Neonatal Vitals Monitoring",
                "subfield": {"id": "SF1", "display_name": "Pediatrics"},
                "field": {"id": "F1", "display_name": "Medicine"},
                "domain": {"id": "D1", "display_name": "Health Sciences"},
            },
            "topics": [{
                "id": "T101",
                "display_name": "Heart Rate Variability",
                "subfield": {"id": "SF1", "display_name": "Pediatrics"},
                "field": {"id": "F1", "display_name": "Medicine"},
                "domain": {"id": "D1", "display_name": "Health Sciences"},
            }],
        })
        # Tight cluster: small jitter around `base` so coherence stays high.
        v = base + 0.05 * rng.standard_normal(128).astype("float32")
        v /= np.linalg.norm(v) + 1e-9
        embeddings_repo.upsert(conn, oa, "placeholder-v1", v)
    conn.close()
    return user_id


@pytest.fixture()
def app(tmp_path, monkeypatch):
    db = tmp_path / "radar.db"
    monkeypatch.setenv("RADAR_DB_PATH", str(db))
    monkeypatch.setenv("RADAR_VAULT_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("RADAR_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("RADAR_LOG_JSON", "false")
    monkeypatch.setenv("RADAR_SCHEDULER_ENABLED", "false")
    settings_module.get_settings.cache_clear()

    _seed_user_papers(db)

    from rag_lib.api.app import create_app
    from rag_lib.api.routers.profiles import get_wizard_gatherer
    from tests.fixture_gatherer import FixtureGatherer

    app = create_app()
    app.dependency_overrides[get_wizard_gatherer] = lambda: FixtureGatherer(
        seed=42, n_candidates=12,
    )
    yield app

    settings_module.get_settings.cache_clear()


def _attach_seeds(db_path: str, user_id: int, slug: str) -> int:
    """Attach all 8 seeded papers to the given draft. Returns profile_id."""
    conn = connect(db_path)
    try:
        row = profiles_repo.get_by_slug(conn, user_id, slug)
        assert row is not None
        profile_id = int(row["id"])
        for i in range(8):
            profiles_repo.attach_seed(conn, profile_id, f"W{3000000 + i}")
        return profile_id
    finally:
        conn.close()


def _request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(method, path, **kwargs)
    return asyncio.run(_run())


def test_create_draft_returns_slug(app, tmp_path):
    resp = _request(app, "POST", "/api/profiles/draft", json={"name": "Neonatal Vitals"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["slug"] == "neonatal-vitals"
    assert body["name"] == "Neonatal Vitals"

    # Draft should not appear in list_profiles output.
    resp = _request(app, "GET", "/api/profiles")
    assert resp.json() == []


def test_create_draft_slug_collision_suffixes(app):
    _request(app, "POST", "/api/profiles/draft", json={"name": "alpha"})
    resp = _request(app, "POST", "/api/profiles/draft", json={"name": "alpha"})
    assert resp.status_code == 200
    assert resp.json()["slug"] == "alpha-2"


def test_coherence_endpoint(app, tmp_path):
    db = str(tmp_path / "radar.db")
    resp = _request(app, "POST", "/api/profiles/draft", json={"name": "Neonatal"})
    slug = resp.json()["slug"]
    _attach_seeds(db, user_id=1, slug=slug)

    resp = _request(app, "POST", f"/api/profiles/draft/{slug}/coherence")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body.keys()) == {"bins", "median", "iqr", "bimodal", "n"}
    assert body["n"] == 8
    assert len(body["bins"]) == 16
    # Tight cluster → median > 0.7.
    assert body["median"] > 0.7
    assert body["bimodal"] is False


def test_topics_endpoint_returns_aggregated_topic(app, tmp_path):
    db = str(tmp_path / "radar.db")
    resp = _request(app, "POST", "/api/profiles/draft", json={"name": "Neonatal"})
    slug = resp.json()["slug"]
    _attach_seeds(db, user_id=1, slug=slug)

    resp = _request(app, "GET", f"/api/profiles/draft/{slug}/topics")
    assert resp.status_code == 200, resp.text
    topics = resp.json()
    ids = {t["id"] for t in topics}
    assert "T100" in ids
    assert "T101" in ids
    by_id = {t["id"]: t for t in topics}
    assert by_id["T100"]["count"] == 8
    assert by_id["T100"]["on"] is True


def test_dry_run_with_fixture_gatherer(app, tmp_path):
    db = str(tmp_path / "radar.db")
    resp = _request(app, "POST", "/api/profiles/draft", json={"name": "Neonatal"})
    slug = resp.json()["slug"]
    _attach_seeds(db, user_id=1, slug=slug)
    # Step 3 must run before Step 4 so topic_filters are populated.
    _request(app, "GET", f"/api/profiles/draft/{slug}/topics")

    resp = _request(
        app, "POST", f"/api/profiles/draft/{slug}/dry-run",
        json={"days": 7, "thresholds": [0.50, 0.75, 0.95]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert {"sweep", "preview", "scores"} <= set(body.keys())
    # ``scores`` carries one raw cosine per fetched candidate so the
    # wizard can render a slider-driven histogram. Length matches the
    # number of ranked candidates.
    assert isinstance(body["scores"], list)
    sweep = body["sweep"]
    assert [round(s["th"], 2) for s in sweep] == [0.50, 0.75, 0.95]
    # Selector primary score is raw cosine; sweep counts how many papers
    # clear each cosine threshold. Lower thresholds always admit at least
    # as many candidates as higher ones.
    counts = [s["n"] for s in sweep]
    assert counts[0] >= counts[-1]
    # Preview cards have the wizard's slug as the profile field.
    if body["preview"]:
        assert body["preview"][0]["profile"] == slug


def test_commit_draft_flips_is_draft_and_lists_profile(app, tmp_path):
    db = str(tmp_path / "radar.db")
    resp = _request(app, "POST", "/api/profiles/draft", json={"name": "Neonatal"})
    slug = resp.json()["slug"]
    _attach_seeds(db, user_id=1, slug=slug)
    _request(app, "GET", f"/api/profiles/draft/{slug}/topics")

    resp = _request(
        app, "POST", "/api/profiles",
        json={
            "slug": slug,
            "threshold": 0.85,
            "selected_topic_ids": ["T100"],
            "cron": "0 4 * * *",
            "tz": "UTC",
        },
    )
    assert resp.status_code == 200, resp.text
    profile = resp.json()
    assert profile["key"] == slug
    assert profile["threshold"] == 0.85
    assert profile["seeds"] == 8

    # Now appears in list_profiles.
    resp = _request(app, "GET", "/api/profiles")
    keys = [p["key"] for p in resp.json()]
    assert slug in keys

    # is_draft should be 0 in the DB; topic_filters pruned to T100 only.
    conn = connect(db)
    try:
        row = profiles_repo.get_by_slug(conn, 1, slug)
        assert row is not None
        assert row["is_draft"] == 0
        topics = profiles_repo.topic_filters(conn, int(row["id"]))
        topic_ids = {t["id"] for t in (topics.get("topics") or [])}
        assert topic_ids == {"T100"}
    finally:
        conn.close()


def test_delete_draft_removes_row(app, tmp_path):
    db = str(tmp_path / "radar.db")
    resp = _request(app, "POST", "/api/profiles/draft", json={"name": "Throwaway"})
    slug = resp.json()["slug"]

    resp = _request(app, "DELETE", f"/api/profiles/draft/{slug}")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}

    conn = connect(db)
    try:
        row = profiles_repo.get_by_slug(conn, 1, slug)
        assert row is None
    finally:
        conn.close()


def test_delete_unknown_draft_404(app):
    resp = _request(app, "DELETE", "/api/profiles/draft/does-not-exist")
    assert resp.status_code == 404


def test_dry_run_unknown_draft_404(app):
    resp = _request(
        app, "POST", "/api/profiles/draft/missing/dry-run",
        json={"days": 7},
    )
    assert resp.status_code == 404
