"""Profiles API tests.

Stand up the FastAPI app against a tmp SQLite seeded with one profile +
50 candidates + a finished gather run, then exercise the read endpoints
and the placeholder write endpoints.
"""

from __future__ import annotations

import asyncio
import sqlite3

import httpx
import numpy as np
import pytest

from rag_lib.api import settings as settings_module
from rag_lib.db import apply_migrations, connect, encode_vector
from rag_lib.db.repos import (
    candidates as candidates_repo,
    embeddings as embeddings_repo,
    gather_runs as gather_runs_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    users as users_repo,
)


def _seed_db(db_path) -> dict:
    """Create one profile + 12 seed papers + 50 candidate papers."""
    conn = connect(db_path)
    apply_migrations(conn)
    user = users_repo.upsert(conn, "demo@example.com")
    user_id = int(user["id"])

    rng = np.random.default_rng(42)
    profile_id = profiles_repo.upsert(
        conn,
        user_id=user_id,
        name="provenance / fairscape",
        embedding_model="placeholder-v1",
        n_seed=12,
        topic_filters={"topics": [
            {"id": "T11431", "display_name": "Research Data Provenance and Evidence", "count": 11},
            {"id": "T10123", "display_name": "FAIR Data Principles", "count": 9},
        ]},
        threshold=0.85,
        coherence_median=0.78,
        coherence_iqr=0.08,
        coherence_bimodal=False,
    )

    # 12 seed papers + their embeddings.
    for i in range(12):
        oa = f"W{1000000 + i}"
        papers_repo.upsert(conn, {
            "openalex_id": oa,
            "title": f"Seed paper {i+1}: provenance topic {i}",
            "year": 2024,
            "venue": "JAMIA",
            "source": "user_csv",
            "primary_topic": {"id": "T11431", "display_name": "Research Data Provenance and Evidence"},
            "topics": [{"id": "T10123", "display_name": "FAIR Data Principles"}],
        })
        v = rng.standard_normal(64).astype("float32")
        v /= (np.linalg.norm(v) + 1e-9)
        embeddings_repo.upsert(conn, oa, "placeholder-v1", v)
        profiles_repo.attach_seed(conn, profile_id, oa)

    # 50 candidates with descending scores.
    run_id = gather_runs_repo.start(
        conn, profile_id=profile_id, user_id=user_id,
        since_date="2026-03-26", filter_string="topics.id:T11431",
        tier_used="must-have-AND",
    )
    for i in range(50):
        oa = f"W{2000000 + i}"
        score = 0.95 - (i * 0.01)
        papers_repo.upsert(conn, {
            "openalex_id": oa,
            "doi": f"10.1234/foo.{i}" if i % 2 == 0 else None,
            "title": f"Candidate paper {i+1}: about something",
            "abstract": "We present a method for " + (" ".join(["lorem"] * (40 + i))),
            "year": 2026,
            "venue": "bioRxiv" if i % 3 == 0 else "Nature Methods",
            "publication_date": "2026-04-15",
            "source": "openalex_gatherer",
            "primary_topic": (
                {"id": "T11431", "display_name": "Research Data Provenance and Evidence"}
                if i % 2 == 0 else
                {"id": "T99999", "display_name": "Something else"}
            ),
            "topics": [{"id": "T10123", "display_name": "FAIR Data Principles"}] if i % 4 == 0 else [],
        })
        candidates_repo.insert_dedup(
            conn,
            profile_id=profile_id,
            openalex_id=oa,
            score=score,
            tier_used="must-have-AND",
            gather_run_id=run_id,
        )
    gather_runs_repo.finish(
        conn, run_id,
        n_fetched=50, n_new=50, n_redup=0, api_calls=3,
    )
    conn.close()
    return {"user_id": user_id, "profile_id": profile_id}


@pytest.fixture()
def app(tmp_path, monkeypatch):
    db = tmp_path / "radar.db"
    monkeypatch.setenv("RADAR_DB_PATH", str(db))
    monkeypatch.setenv("RADAR_VAULT_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("RADAR_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("RADAR_LOG_JSON", "false")
    settings_module.get_settings.cache_clear()

    _seed_db(db)

    from rag_lib.api.app import create_app

    app = create_app()
    yield app

    settings_module.get_settings.cache_clear()


def _request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(method, path, **kwargs)
    return asyncio.run(_run())


def test_list_profiles_returns_seeded_profile(app):
    resp = _request(app, "GET", "/api/profiles")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body, list) and len(body) == 1
    p = body[0]
    assert p["key"] == "provenance-fairscape"
    assert p["name"] == "provenance / fairscape"
    assert p["seeds"] == 12
    assert p["threshold"] == 0.85
    assert isinstance(p["hue"], int) and 0 <= p["hue"] < 360
    # Health derives from coherence + 30d engagement; coherence=0.78 OK,
    # but no saves/dismisses yet → warn.
    assert p["health"] in {"warn", "ok"}


def test_get_profile_by_key(app):
    resp = _request(app, "GET", "/api/profiles/provenance-fairscape")
    assert resp.status_code == 200, resp.text
    p = resp.json()
    assert p["key"] == "provenance-fairscape"


def test_get_profile_unknown_key_returns_null(app):
    resp = _request(app, "GET", "/api/profiles/does-not-exist")
    assert resp.status_code == 200
    assert resp.json() is None


def test_profile_detail_shape(app):
    resp = _request(app, "GET", "/api/profiles/provenance-fairscape/detail")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body.keys()) >= {
        "profile", "seeds", "topics", "sweep",
        "coherenceBins", "coherenceStats",
        "feedbackLog", "feedbackMoreCount",
    }
    assert body["profile"]["key"] == "provenance-fairscape"
    assert len(body["seeds"]) == 12
    assert body["seeds"][0]["idx"] == 1
    assert {"id", "name", "count", "on"} <= set(body["topics"][0].keys())
    assert isinstance(body["coherenceBins"], list) and len(body["coherenceBins"]) == 16
    assert body["feedbackLog"] == [] and body["feedbackMoreCount"] == 0


def test_refit_returns_placeholder_cost(app):
    resp = _request(app, "POST", "/api/profiles/provenance-fairscape/refit")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["key"] == "provenance-fairscape"
    assert isinstance(body["cost"], str)


def test_refit_unknown_profile_404(app):
    resp = _request(app, "POST", "/api/profiles/does-not-exist/refit")
    assert resp.status_code == 404


def test_dry_run_reports_candidate_count(app):
    resp = _request(app, "POST", "/api/profiles/provenance-fairscape/dry-run")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["key"] == "provenance-fairscape"
    assert body["n"] == 50
    # Per-candidate scores power the histogram + slider on the
    # profile detail page; one score per persisted candidate.
    assert isinstance(body["scores"], list)
    assert len(body["scores"]) == 50
