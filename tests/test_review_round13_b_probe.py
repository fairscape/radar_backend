"""round 13 review (b): exploratory probes over the older feature surfaces."""

from __future__ import annotations

import asyncio
import io
import json
import sqlite3
from pathlib import Path

import httpx
import pytest

from rag_lib.api import settings as settings_module
from rag_lib.api.services import vault as vault_service
from rag_lib.db import connect
from rag_lib.db.repos import candidates as candidates_repo
from rag_lib.db.repos import papers as papers_repo
from rag_lib.db.repos import profiles as profiles_repo
from rag_lib.db.repos import users as users_repo
from rag_lib.scheduler import jobs as scheduler_jobs
from rag_lib.vault import PdfIngestRecord
from tests.fixture_gatherer import FixtureGatherer
from tests.test_api_profiles import _seed_db
from tests.test_review_round12_journeys import InlineScheduler

A = {"X-User-Email": "demo@example.com"}
B = {"X-User-Email": "other@example.com"}
SLUG = "provenance-fairscape"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    db = tmp_path / "radar.db"
    for k, v in {
        "RADAR_DB_PATH": str(db), "RADAR_VAULT_DIR": str(tmp_path / "vault"),
        "RADAR_CHROMA_DIR": str(tmp_path / "chroma"), "RADAR_LOG_JSON": "false",
        "RADAR_SCHEDULER_ENABLED": "false", "RADAR_DEFAULT_EMBEDDING_MODEL": "placeholder-v1",
        "RADAR_UMLS_ENABLED": "false", "RADAR_CONTENT_TOPICS_ENABLED": "false",
        "RADAR_RERANKER_ENABLED": "false",
    }.items():
        monkeypatch.setenv(k, v)
    settings_module.get_settings.cache_clear()
    _seed_db(db)
    # The seed helper stores 64-dim random vectors; re-embed the seeds with
    # the real placeholder embedder so later seeds (from uploads / the feed)
    # have the same dimension.
    from rag_lib.embedders import get_embedder
    from rag_lib.db.repos import embeddings as embeddings_repo
    emb = get_embedder("placeholder-v1")
    conn = connect(db)
    for (oa, title) in conn.execute("SELECT openalex_id, title FROM papers").fetchall():
        embeddings_repo.upsert(conn, oa, "placeholder-v1", emb(title))
    conn.close()

    def _fake_ingest_pdf(path):
        p = Path(path)
        # No title in the PDF: ingest_pdf falls back to the path stem.
        if b"notitle" in p.read_bytes():
            return PdfIngestRecord(path=str(p), title=p.stem, body_text="body text " * 20, doi=None, n_pages=2)
        return PdfIngestRecord(path=str(p), title=f"Uploaded {p.stem[1:9]}",
                               body_text=f"body of {p.stem[1:9]} " * 5, doi=None, n_pages=2)

    class _NoNetClient:
        def __init__(self, **_): pass
        def lookup_by_doi(self, doi): return None
        def search_by_title(self, *a, **kw): return []
        def paper_from_work(self, *a, **kw): return None

    monkeypatch.setattr(vault_service, "ingest_pdf", _fake_ingest_pdf)
    monkeypatch.setattr(vault_service, "OpenAlexClient", _NoNetClient)
    from rag_lib.rag import indexer as rag_indexer
    monkeypatch.setattr(rag_indexer, "index_user_collection", lambda _settings, _user_id: None)
    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer",
                        lambda **kw: FixtureGatherer(seed=7, n_candidates=10))
    yield db
    settings_module.get_settings.cache_clear()


@pytest.fixture()
def app(env):
    from rag_lib.api.app import create_app
    from rag_lib.api.deps import get_scheduler

    app = create_app()
    app.state.inline = InlineScheduler()
    app.dependency_overrides[get_scheduler] = lambda: app.state.inline
    return app


def _request(app, method, path, **kwargs) -> httpx.Response:
    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(method, path, **kwargs)
    return asyncio.run(_run())


def _ok(resp, code=200):
    assert resp.status_code == code, f"{resp.request.method} {resp.request.url} -> {resp.status_code}: {resp.text}"
    return resp.json() if resp.content else None


def _upload(app, content: bytes, filename: str, slug: str | None = None, headers=None):
    files = {"file": (filename, io.BytesIO(b"%PDF-1.4\n" + content), "application/pdf")}
    data = {"profile_slug": slug} if slug else {}
    return _request(app, "POST", "/api/vault/upload", files=files, data=data, headers=headers or A)


def _q(env, sql, *params):
    conn = connect(env)
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------------------


def test_probe_cross_user_feed_and_triage(app, env):
    daily = _ok(_request(app, "GET", "/api/radar/daily", headers=A))
    cid = daily["cards"][0]["id"]
    # B cannot see or touch A's card.
    assert _ok(_request(app, "GET", "/api/radar/daily", headers=B))["cards"] == []
    assert _request(app, "POST", "/api/radar/cards/save", json={"card_id": cid}, headers=B).status_code == 404
    assert _request(app, "POST", "/api/radar/cards/save", json={"card_id": cid, "profile": SLUG}, headers=B).status_code == 404
    assert _ok(_request(app, "GET", f"/api/radar/daily?profile={SLUG}", headers=B))["cards"] == []
    assert _request(app, "GET", f"/api/profiles/{SLUG}/feedback", headers=B).status_code == 404
    assert _ok(_request(app, "GET", "/api/vault/docs?tag=" + SLUG, headers=B)) == []


def test_probe_feed_year_filters_and_years_patch(app, env):
    # A candidate with no year passes "published since".
    conn = connect(env)
    conn.execute("UPDATE papers SET year = NULL, publication_date = NULL WHERE openalex_id = 'W2000000'")
    conn.commit(); conn.close()
    d = _ok(_request(app, "GET", "/api/radar/daily?year_from=2030", headers=A))
    print("year_from=2030 ->", [c["id"] for c in d["cards"]])
    # Years PATCH validation.
    r = _request(app, "PATCH", f"/api/profiles/{SLUG}", json={"years": {"year_from": 2030, "year_to": None}}, headers=A)
    print("years 2030:", r.status_code, r.text)
    r = _request(app, "PATCH", f"/api/profiles/{SLUG}", json={"years": {"year_from": 2020, "year_to": 2010}}, headers=A)
    print("years reversed:", r.status_code, r.text)
    r = _request(app, "PATCH", f"/api/profiles/{SLUG}", json={"years": {"year_from": None, "year_to": 2020}}, headers=A)
    print("years to-only:", r.status_code, r.text[:200])
    d = _ok(_request(app, "GET", f"/api/radar/daily?profile={SLUG}", headers=A))
    print("feed after to=2020:", len(d["cards"]), [c["id"] for c in d["cards"]][:5])
    dr = _ok(_request(app, "POST", f"/api/profiles/{SLUG}/dry-run", headers=A))
    print("dry-run after to=2020: n=", dr["n"], "papers=", len(dr["papers"]))
    p = _ok(_request(app, "GET", f"/api/profiles/{SLUG}", headers=A))
    print("profile years:", p["yearFrom"], p["yearTo"])
    # clear
    _ok(_request(app, "PATCH", f"/api/profiles/{SLUG}", json={"years": {"year_from": None, "year_to": None}}, headers=A))
    d = _ok(_request(app, "GET", f"/api/radar/daily?profile={SLUG}", headers=A))
    print("feed after clear:", len(d["cards"]))


def test_probe_backfill(app, env):
    r = _request(app, "POST", f"/api/profiles/{SLUG}/backfill", json={"year_from": None, "year_to": None}, headers=A)
    print("backfill none:", r.status_code, r.text)
    r = _request(app, "POST", f"/api/profiles/{SLUG}/backfill", json={"year_from": 2030}, headers=A)
    print("backfill 2030:", r.status_code, r.text)
    r = _request(app, "POST", f"/api/profiles/{SLUG}/backfill", json={"year_from": 2000, "year_to": 2003}, headers=A)
    print("backfill 2000-2003:", r.status_code, r.text)
    runs = _ok(_request(app, "GET", f"/api/profiles/{SLUG}/runs", headers=A))
    print("runs:", [(x["id"], x["tier_used"], x["since_date"], x["n_fetched"], x["n_new"], x["error"]) for x in runs])
    rows = _q(env, "SELECT pc.openalex_id, p.year, pc.score_raw, pc.score_pct, pc.tier_used FROM profile_candidates pc JOIN papers p USING (openalex_id) WHERE pc.gather_run_id = ?", runs[0]["id"])
    print("backfilled:", rows[:12])
    d = _ok(_request(app, "GET", f"/api/radar/daily?profile={SLUG}", headers=A))
    print("feed fetchedAt/scored:", d["fetchedAt"], d["candidatesScored"], len(d["cards"]))
    # Now with years saved to 2001-2002: feed should restrict.
    _ok(_request(app, "PATCH", f"/api/profiles/{SLUG}", json={"years": {"year_from": 2001, "year_to": 2002}}, headers=A))
    d = _ok(_request(app, "GET", f"/api/radar/daily?profile={SLUG}", headers=A))
    print("feed 2001-2002:", [(c["id"], c["date"], c["bucket"], c["score"]) for c in d["cards"]])
    rows = _q(env, "SELECT pc.openalex_id, p.year, pc.score_pct, pc.score_blended FROM profile_candidates pc JOIN papers p USING (openalex_id) WHERE pc.profile_id = 1 AND p.year < 2010")
    print("pool after years:", rows)


def test_probe_feed_add_to_seeds(app, env):
    daily = _ok(_request(app, "GET", f"/api/radar/daily?profile={SLUG}", headers=A))
    cid = daily["cards"][0]["id"]
    r = _request(app, "POST", f"/api/profiles/{SLUG}/seeds", json={"openalex_ids": [cid]}, headers=A)
    print("add seed:", r.status_code, r.text[:300])
    d2 = _ok(_request(app, "GET", f"/api/radar/daily?profile={SLUG}", headers=A))
    print("still in feed?", cid in [c["id"] for c in d2["cards"]])
    docs = _ok(_request(app, "GET", "/api/vault/docs?tag=" + SLUG, headers=A))
    print("vault has it?", cid in [x["id"] for x in docs])
    det = _ok(_request(app, "GET", f"/api/profiles/{SLUG}/detail", headers=A))
    print("seeds:", det["profile"]["seeds"], [s["id"] for s in det["seeds"]][-2:])
    # And remove it again: does it come back to the feed?
    r = _request(app, "DELETE", f"/api/profiles/{SLUG}/seeds?openalex_id={cid}", headers=A)
    print("remove seed:", r.status_code, r.text[:300])
    d3 = _ok(_request(app, "GET", f"/api/radar/daily?profile={SLUG}", headers=A))
    print("back in feed?", cid in [c["id"] for c in d3["cards"]])
    rows = _q(env, "SELECT openalex_id, score, score_raw, score_pct, score_blended FROM profile_candidates WHERE openalex_id = ?", cid)
    print("row:", rows)
    # B cannot seed A's paper
    r = _request(app, "POST", f"/api/profiles/{SLUG}/seeds", json={"openalex_ids": [cid]}, headers=B)
    print("B add seed:", r.status_code, r.text[:200])


def test_probe_vault(app, env):
    r1 = _ok(_upload(app, b"bytes-1", "first.pdf"))
    print("up1:", r1)
    r2 = _ok(_upload(app, b"bytes-1", "first.pdf"))
    print("up1 again:", r2)
    r3 = _ok(_upload(app, b"bytes-2", "first.pdf"))
    print("changed file same name:", r3)
    # No-title PDF: title from the filename?
    r4 = _ok(_upload(app, b"notitle-bytes", "My Great Paper.pdf"))
    print("no title:", r4["title"], r4["id"])
    # another user same bytes
    r5 = _ok(_upload(app, b"bytes-1", "copy.pdf", headers=B))
    print("B same bytes:", r5)
    print("A docs:", [(d["id"], d["title"]) for d in _ok(_request(app, "GET", "/api/vault/docs", headers=A))])
    print("B docs:", [(d["id"], d["title"]) for d in _ok(_request(app, "GET", "/api/vault/docs", headers=B))])
    print("A stats:", _ok(_request(app, "GET", "/api/vault/stats", headers=A)))
    print("B stats:", _ok(_request(app, "GET", "/api/vault/stats", headers=B)))
    # local id through the seed routes.
    lid = r1["id"]
    r = _request(app, "POST", f"/api/profiles/{SLUG}/seeds", json={"openalex_ids": [lid]}, headers=A)
    print("attach local to live:", r.status_code, r.text[:300])
    det = _ok(_request(app, "GET", f"/api/profiles/{SLUG}/detail", headers=A))
    print("seeds now:", det["profile"]["seeds"])
    r = _request(app, "GET", f"/api/profiles/{SLUG}/umls", headers=A)
    print("umls status:", r.status_code, r.text[:300])
    r = _request(app, "GET", f"/api/profiles/{SLUG}/reranker-comparison", headers=A)
    print("rerank cmp:", r.status_code, r.text[:300])
    r = _request(app, "POST", f"/api/profiles/{SLUG}/recompute-topics", headers=A)
    print("recompute topics:", r.status_code, r.text[:300])
    # upload to profile of other user
    r = _upload(app, b"bytes-9", "x.pdf", slug=SLUG, headers=B)
    print("B upload to A's profile:", r.status_code, r.text[:200])
    # tags
    print("A tags:", _ok(_request(app, "GET", "/api/vault/tags", headers=A)))
    # upload with slug to live profile → refit → feed still OK?
    r = _upload(app, b"bytes-3", "third.pdf", slug=SLUG)
    print("upload to live:", r.status_code, r.text[:300])
    d = _ok(_request(app, "GET", f"/api/radar/daily?profile={SLUG}", headers=A))
    print("feed after upload-seed:", len(d["cards"]), d["cards"][0]["similarity"] if d["cards"] else None)


def test_probe_reranker_comparison(app, env):
    conn = connect(env)
    ids = [r[0] for r in conn.execute("SELECT openalex_id FROM profile_candidates WHERE profile_id = 1 ORDER BY score DESC").fetchall()]
    # score_raw for all; reranker logits for half; one row out of years.
    for i, oa in enumerate(ids):
        conn.execute("UPDATE profile_candidates SET score_raw = ? WHERE openalex_id = ?", (0.95 - i * 0.01, oa))
        if i % 2 == 0:
            conn.execute("UPDATE profile_candidates SET score_reranker_raw = ? WHERE openalex_id = ?", (float(i), oa))
    conn.commit()
    from rag_lib.scoring.pool import rescore_profile_pool
    rescore_profile_pool(conn, 1, alpha=0.4, beta=0.6)
    conn.close()
    r = _ok(_request(app, "GET", f"/api/profiles/{SLUG}/reranker-comparison?limit=5", headers=A))
    print("cmp:", json.dumps(r)[:800])
    d = _ok(_request(app, "GET", f"/api/radar/daily?profile={SLUG}", headers=A))
    print("feed top:", [(c["id"], c["score"], c["similarity"], c["bucket"]) for c in d["cards"][:6]])
    rows = _q(env, "SELECT openalex_id, score_raw, score_reranker_raw, score_reranker_norm, score_blended, score_pct FROM profile_candidates WHERE profile_id=1 ORDER BY COALESCE(score_blended, score) DESC LIMIT 6")
    print(rows)


def test_probe_users_health_providers(app, env):
    me = _ok(_request(app, "GET", "/api/users/me", headers=A))
    print("me:", me)
    r = _request(app, "PATCH", "/api/users/me", json={"mailto": "  Someone@Example.org "}, headers=A)
    print("patch:", r.status_code, r.text)
    r = _request(app, "PATCH", "/api/users/me", json={"mailto": ""}, headers=A)
    print("patch empty:", r.status_code, r.text)
    r = _request(app, "PATCH", "/api/users/me", json={}, headers=A)
    print("patch none:", r.status_code, r.text)
    r = _request(app, "GET", "/api/health")
    print("health:", r.status_code, r.text)
    r = _request(app, "GET", "/api/chat/providers", headers=A)
    print("providers:", r.status_code, r.text[:300])
    r = _request(app, "GET", "/api/users/me", headers={"X-User-Email": "Not an email"})
    print("bad email:", r.status_code, r.text)


def test_probe_feedback_after_undo(app, env):
    daily = _ok(_request(app, "GET", f"/api/radar/daily?profile={SLUG}", headers=A))
    cid = daily["cards"][0]["id"]
    _ok(_request(app, "POST", "/api/radar/cards/save", json={"card_id": cid, "profile": SLUG, "on": True}, headers=A))
    _ok(_request(app, "POST", "/api/radar/cards/save", json={"card_id": cid, "profile": SLUG, "on": False}, headers=A))
    fb = _ok(_request(app, "GET", f"/api/profiles/{SLUG}/feedback", headers=A))
    print("feedback after undo:", [(x["action"], x["openalex_id"]) for x in fb])
    det = _ok(_request(app, "GET", f"/api/profiles/{SLUG}/detail", headers=A))
    print("detail feedbackLog:", det["feedbackLog"], det["profile"]["saves30"])
    ft = _ok(_request(app, "GET", f"/api/profiles/{SLUG}/feedback-threshold", headers=A))
    print("feedback-threshold:", ft)
    # below-the-bar dismiss
    conn = connect(env)
    conn.execute("UPDATE profile_candidates SET score_raw = 0.5 WHERE openalex_id = 'W2000049'")
    conn.commit(); conn.close()
    low = _ok(_request(app, "GET", f"/api/radar/daily?profile={SLUG}&bucket=low", headers=A))
    print("low:", [(c["id"], c["similarity"], c["bucket"], c["score"]) for c in low["cards"]])
    r = _request(app, "POST", "/api/radar/cards/dismiss", json={"card_id": "W2000049", "profile": SLUG, "on": True}, headers=A)
    print("dismiss low:", r.status_code, r.text)
    low2 = _ok(_request(app, "GET", f"/api/radar/daily?profile={SLUG}&bucket=low", headers=A))
    print("low after dismiss:", [(c["id"], c["state"]) for c in low2["cards"]], low2["states"])


def test_probe_cli_slug_collision(env):
    conn = connect(env)
    uid = int(users_repo.get(conn, 1)["id"])
    p1 = profiles_repo.upsert(conn, user_id=uid, name="My Topic", embedding_model="placeholder-v1", n_seed=0)
    p2 = profiles_repo.upsert(conn, user_id=uid, name="my topic", embedding_model="placeholder-v1", n_seed=0)
    print("slugs:", [dict(r) for r in conn.execute("SELECT id, name, slug FROM profiles WHERE id IN (?, ?)", (p1, p2))])
    try:
        profiles_repo.upsert(conn, user_id=uid, name="my topic", embedding_model="placeholder-v1", n_seed=1)
        print("second upsert OK:", [dict(r) for r in conn.execute("SELECT id, name, slug FROM profiles WHERE id IN (?, ?)", (p1, p2))])
    except sqlite3.IntegrityError as exc:
        print("IntegrityError:", exc)
    conn.close()
