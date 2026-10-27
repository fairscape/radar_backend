"""Round-3 review (d): complete user journeys through the HTTP API.

Every call below is one the frontend makes (``src/api/endpoints/*.ts``), in
the order the pages make them. Offline: Prosopia / OpenAlex import stubs
from the import suites, the FixtureGatherer for the wizard's trial and --
through an inline stand-in for the scheduler -- for "Scan now" and the
backfill, a fake ``ingest_pdf`` for uploads.
"""

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
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import gather_runs as gather_runs_repo, users as users_repo
from rag_lib.scheduler import jobs as scheduler_jobs
from rag_lib.vault import PdfIngestRecord
from tests.fixture_gatherer import FixtureGatherer
from tests.test_orcid_import import ORCID, WORKS, StubOrcidOpenAlex
from tests.test_prosopia_import import StubProsopia, _stub_openalex, _stub_profile
from tests.test_researcher_topics import DOCUMENT


B = {"X-User-Email": "b@example.com"}       # a second user; no header = demo user (id 1)


# ---------------------------------------------------------------------------
# Fixture: one app, every outside call stubbed, background jobs run inline
# ---------------------------------------------------------------------------


class InlineScheduler:
    """Runs what it is handed, right away, in the request."""

    def __init__(self):
        self.jobs = []

    def add_job(self, func, *a, args=(), kwargs=None, **k):
        self.jobs.append((func, args, kwargs, k))
        return func(*args, **(kwargs or {}))

    def get_job(self, *a, **k):
        return None

    def remove_job(self, *a, **k):
        return None


def _profile_with_document():
    profile = _stub_profile()
    profile.metadata = DOCUMENT
    profile.rid = "0000-0001-5643-4068"
    return profile


def _both_stubs():
    oa = StubOrcidOpenAlex(WORKS)
    prosopia = _stub_openalex()
    for attr in ("by_id", "by_doi", "by_pmcid", "title_hits"):
        getattr(oa, attr).update(getattr(prosopia, attr))
    return oa


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
    conn = connect(db)
    apply_migrations(conn)
    users_repo.upsert(conn, "demo@example.com")
    conn.close()

    # Uploads: no pdfplumber, no OpenAlex.
    def _fake_ingest_pdf(path):
        p = Path(path)
        return PdfIngestRecord(path=str(p), title=f"Uploaded {p.stem[1:9]}",
                               body_text=f"body of {p.stem[1:9]} " * 5, doi=None, n_pages=2)

    class _NoNetClient:
        def __init__(self, **_): pass
        def lookup_by_doi(self, doi): return None
        def search_by_title(self, *a, **kw): return []
        def paper_from_work(self, *a, **kw): return None

    monkeypatch.setattr(vault_service, "ingest_pdf", _fake_ingest_pdf)
    monkeypatch.setattr(vault_service, "OpenAlexClient", _NoNetClient)
    # No Chroma: these journeys do not read chunk counts, and opening one
    # more persistent client late in the full suite (a fresh directory per
    # test, hundreds of them) hung the rust binding's ``start`` for good.
    from rag_lib.rag import indexer as rag_indexer
    monkeypatch.setattr(rag_indexer, "index_user_collection", lambda _settings, _user_id: None)
    # Scans (gather-now / backfill) fetch from the fixture gatherer.
    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer",
                        lambda **kw: FixtureGatherer(seed=7, n_candidates=10))
    yield db
    settings_module.get_settings.cache_clear()


@pytest.fixture()
def app(env):
    from rag_lib.api.app import create_app
    from rag_lib.api.deps import get_scheduler
    from rag_lib.api.routers.profiles import get_wizard_gatherer
    from rag_lib.api.routers.prosopia import get_import_openalex_client, get_prosopia_client

    app = create_app()
    app.state.prosopia = StubProsopia(_profile_with_document())
    app.dependency_overrides[get_prosopia_client] = lambda: app.state.prosopia
    app.state.openalex = _both_stubs()
    app.dependency_overrides[get_import_openalex_client] = lambda: app.state.openalex
    app.dependency_overrides[get_wizard_gatherer] = lambda: FixtureGatherer(seed=42, n_candidates=12)
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


def _ok(resp: httpx.Response, code: int = 200):
    assert resp.status_code == code, f"{resp.request.method} {resp.request.url} -> {resp.status_code}: {resp.text}"
    return resp.json() if resp.content else None


def _upload(app, content: bytes, filename: str, slug: str | None = None, headers=None):
    files = {"file": (filename, io.BytesIO(b"%PDF-1.4\n" + content), "application/pdf")}
    data = {"profile_slug": slug} if slug else {}
    return _request(app, "POST", "/api/vault/upload", files=files, data=data, headers=headers or {})


def _q(env, sql, *params):
    conn = connect(env)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _one(env, sql, *params):
    rows = _q(env, sql, *params)
    return rows[0] if rows else None


def _import_researcher(app, ref="sheffield-nathan", source="prosopia", headers=None, **extra) -> int:
    body = {"source": source, "ref": ref, **extra}
    return _ok(_request(app, "POST", "/api/researchers/import", json=body, headers=headers or {}))["researcher_id"]


def _go_live(app, env, name="Live", *, seeds_from_researcher=True, headers=None, threshold=None):
    """The wizard start to finish: draft, seeds, topics, trial, commit. Returns (slug, rid)."""
    h = headers or {}
    rid = _import_researcher(app, headers=h) if seeds_from_researcher else None
    if rid is not None:
        papers = _ok(_request(app, "GET", f"/api/researchers/{rid}", headers=h))["papers"]
        ids = [p["id"] for p in papers]
        slug = _ok(_request(app, "POST", f"/api/researchers/{rid}/interests",
                            json={"name": name, "openalex_ids": ids}, headers=h))["draft"]["slug"]
    else:
        slug = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": name}, headers=h))["slug"]
        _ok(_upload(app, b"seed-one-" + name.encode(), "one.pdf", slug, headers=h))
        _ok(_upload(app, b"seed-two-" + name.encode(), "two.pdf", slug, headers=h))
    topics = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics", headers=h))
    on = [t["id"] for t in topics if t["on"]]
    run = _ok(_request(app, "POST", f"/api/profiles/draft/{slug}/dry-run",
                       json={"days": 30, "selected_topic_ids": on}, headers=h))
    st = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/dry-run/{run['run_id']}", headers=h))
    scores = st["result"]["scores"] if st["result"] else []
    if threshold is None:
        threshold = max(0.0, min(scores)) if scores else 0.0
    _ok(_request(app, "POST", "/api/profiles", json={
        "slug": slug, "threshold": round(threshold, 3), "selected_topic_ids": on,
        "cron": "0 4 * * *", "tz": "UTC", "years": None}, headers=h))
    return slug, rid


def _scan(app, slug, headers=None, days=7, limit=500):
    return _ok(_request(app, "POST", f"/api/profiles/{slug}/gather-now?days={days}&limit={limit}", headers=headers or {}))


def _feed(app, slug=None, headers=None, **params):
    if slug:
        params["profile"] = slug
    return _ok(_request(app, "GET", "/api/radar/daily", params=params, headers=headers or {}))


# ===========================================================================
# J1 -- a new user builds an interest and triages the feed
# ===========================================================================


def test_j1_wizard_to_feed_and_back(app, env):
    # Step 0: the interests page asks for drafts and interests; nothing yet.
    assert _ok(_request(app, "GET", "/api/profiles")) == []
    assert _ok(_request(app, "GET", "/api/profiles/drafts")) == []

    # Step 1: name -> draft.
    draft = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": "Neonatal Vitals"}))
    slug = draft["slug"]
    assert slug == "neonatal-vitals" and draft["name"] == "Neonatal Vitals"
    drafts = _ok(_request(app, "GET", "/api/profiles/drafts"))
    assert [d["slug"] for d in drafts] == [slug] and drafts[0]["phase"] == "empty" and drafts[0]["n_seeds"] == 0

    # Seeds by PDF upload (the drop zone), one at a time as the wizard does.
    up1 = _ok(_upload(app, b"paper-one", "one.pdf", slug))
    up2 = _ok(_upload(app, b"paper-two", "two.pdf", slug))
    assert up1["id"].startswith("local:") and up2["id"].startswith("local:") and up1["id"] != up2["id"]
    assert slug in up1["tags"] and up1["duplicate"] is not True
    # The same PDF again is a duplicate, still tagged, still one seed.
    again = _ok(_upload(app, b"paper-one", "one-copy.pdf", slug))
    assert again["id"] == up1["id"] and again["duplicate"] is True
    docs = _ok(_request(app, "GET", "/api/vault/docs", params={"tag": slug}))
    assert {d["id"] for d in docs} == {up1["id"], up2["id"]}

    # Seeds by OpenAlex id: a saved researcher's papers ("Add from a saved profile").
    rid = _import_researcher(app)
    detail = _ok(_request(app, "GET", f"/api/researchers/{rid}"))
    oa_ids = [p["id"] for p in detail["papers"] if p["id"].startswith("https://openalex.org/")]
    assert len(oa_ids) >= 3
    added = _ok(_request(app, "POST", f"/api/profiles/{slug}/seeds", json={"openalex_ids": oa_ids}))
    assert added == {"attached": len(oa_ids), "rejected": [], "rescored": None}
    # Something the user does not own is rejected, not attached, not a 500.
    stranger = _ok(_request(app, "POST", f"/api/profiles/{slug}/seeds", json={"openalex_ids": ["https://openalex.org/W999999999"]}))
    assert stranger["attached"] == 0 and stranger["rejected"] == ["https://openalex.org/W999999999"]
    drafts = _ok(_request(app, "GET", "/api/profiles/drafts"))
    assert drafts[0]["n_seeds"] == 2 + len(oa_ids) and drafts[0]["phase"] == "seeded"

    # Step 2: coherence.
    coh = _ok(_request(app, "POST", f"/api/profiles/draft/{slug}/coherence"))
    assert coh["n"] == 2 + len(oa_ids) and len(coh["bins"]) == 16

    # Step 3: UMLS status (off) then the topic list; the user toggles.
    umls = _ok(_request(app, "POST", f"/api/profiles/{slug}/umls"))
    assert umls["enabled"] is False and umls["pending"] == 0
    topics = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))
    by_id = {t["id"]: t for t in topics}
    assert "https://openalex.org/T10001" in by_id or "T10001" in by_id
    t_main = next(t for t in topics if t["id"].endswith("T10001"))
    assert t_main["on"] is True and t_main["count"] >= len(oa_ids)
    # ``count`` is assignments (the primary topic twice); ``papers`` is seeds.
    assert t_main["papers"] == len(oa_ids)
    on_ids = [t["id"] for t in topics if t["on"]]

    # Step 4: the trial scan with step 3's ticks; the ticks are saved.
    kicked = _ok(_request(app, "POST", f"/api/profiles/draft/{slug}/dry-run",
                          json={"days": 30, "selected_topic_ids": on_ids}))
    status = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/dry-run/{kicked['run_id']}"))
    assert status["run"]["finished_at"] and status["run"]["error"] is None
    result = status["result"]
    assert result is not None and len(result["scores"]) == len(result["papers"]) > 0
    assert all(p["profile"] == slug for p in result["preview"])
    assert {t["id"]: t["on"] for t in _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))} \
        == {t["id"]: (t["id"] in on_ids) for t in topics}

    # Untick everything but one and re-run: the saved ticks follow.
    only = [on_ids[0]]
    _ok(_request(app, "POST", f"/api/profiles/draft/{slug}/dry-run", json={"days": 30, "selected_topic_ids": only}))
    assert [t["id"] for t in _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics")) if t["on"]] == only

    # Save: commit with the threshold the slider shows.
    threshold = max(0.0, min(result["scores"]))
    profile = _ok(_request(app, "POST", "/api/profiles", json={
        "slug": slug, "threshold": round(threshold, 3), "selected_topic_ids": only,
        "cron": "0 4 * * *", "tz": "UTC", "years": None}))
    assert profile["key"] == slug and profile["seeds"] == 2 + len(oa_ids)
    assert [p["key"] for p in _ok(_request(app, "GET", "/api/profiles"))] == [slug]
    assert _ok(_request(app, "GET", "/api/profiles/drafts")) == []
    row = _one(env, "SELECT is_draft, threshold, n_seed, centroid FROM profiles WHERE slug = ?", slug)
    assert row["is_draft"] == 0 and row["n_seed"] == 2 + len(oa_ids) and row["centroid"] is not None
    detail = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))
    assert detail["profile"]["key"] == slug and len(detail["seeds"]) == 2 + len(oa_ids)
    assert [t["id"] for t in detail["topics"] if t["on"]] == only
    # The vault tag followed the interest.
    assert {d["id"] for d in _ok(_request(app, "GET", "/api/vault/docs", params={"tag": slug}))} >= {up1["id"], up2["id"]}

    # First scan (the wizard starts it right after the commit).
    run = _scan(app, slug)
    runs = _ok(_request(app, "GET", f"/api/profiles/{slug}/runs"))
    mine = next(r for r in runs if r["id"] == run["run_id"])
    assert mine["finished_at"] and mine["error"] is None and mine["tier_used"] == "manual"
    assert mine["n_fetched"] == 10 and mine["n_new"] is not None

    # The feed.
    feed = _feed(app)
    assert feed["candidatesScored"] == 10 and feed["fetchedAt"]
    cards = feed["cards"]
    assert cards and all(c["profile"] == slug and c["state"] is None for c in cards)
    scores = [c["score"] for c in cards]
    assert scores == sorted(scores, reverse=True)
    assert all({"id", "title", "authors", "venue", "date", "doi", "openalex", "score", "bucket",
                "abstract", "mesh", "terms", "matched", "mins"} <= set(c) for c in cards)
    assert _feed(app, slug)["cards"] == cards
    n_cards = len(cards)
    assert n_cards >= 3, "the threshold was set at the lowest score; most papers should pass"

    # Feedback, with the state wanted (not a toggle), under the interest shown.
    liked, disliked, saved_then_undone = cards[0], cards[1], cards[2]
    r = _ok(_request(app, "POST", "/api/radar/cards/save", json={"card_id": liked["id"], "profile": slug, "on": True}))
    assert r == {"id": liked["id"], "state": "saved", "profile": slug}
    # A second click (double click) with the same wanted state changes nothing.
    r = _ok(_request(app, "POST", "/api/radar/cards/save", json={"card_id": liked["id"], "profile": slug, "on": True}))
    assert r["state"] == "saved"
    r = _ok(_request(app, "POST", "/api/radar/cards/dismiss", json={"card_id": disliked["id"], "profile": slug, "on": True}))
    assert r["state"] == "dismissed"
    _ok(_request(app, "POST", "/api/radar/cards/save", json={"card_id": saved_then_undone["id"], "profile": slug, "on": True}))
    r = _ok(_request(app, "POST", "/api/radar/cards/save", json={"card_id": saved_then_undone["id"], "profile": slug, "on": False}))
    assert r["state"] is None
    # Dismissing a saved card clears the save.
    r = _ok(_request(app, "POST", "/api/radar/cards/dismiss", json={"card_id": liked["id"], "profile": slug, "on": True}))
    assert r["state"] == "dismissed"
    r = _ok(_request(app, "POST", "/api/radar/cards/save", json={"card_id": liked["id"], "profile": slug, "on": True}))
    assert r["state"] == "saved"
    assert _one(env, "SELECT saved_at, dismissed_at FROM profile_candidates WHERE openalex_id = ?", liked["id"])["dismissed_at"] is None

    feed2 = _feed(app)
    st = {c["id"]: c["state"] for c in feed2["cards"]}
    assert st[liked["id"]] == "saved" and st[disliked["id"]] == "dismissed" and st[saved_then_undone["id"]] is None
    assert feed2["states"][liked["id"]] == "saved" and feed2["states"][disliked["id"]] == "dismissed"
    assert saved_then_undone["id"] not in feed2["states"]
    # Every card is still listed once (the page hides triaged ones itself).
    assert sorted(c["id"] for c in feed2["cards"]) == sorted(c["id"] for c in cards)
    # The feedback log saw one save per transition and one dismiss each.
    events = _ok(_request(app, "GET", f"/api/profiles/{slug}/feedback", params={"limit": 50}))
    actions = sorted((e["openalex_id"], e["action"]) for e in events)
    assert actions.count((liked["id"], "saved")) == 2 and actions.count((liked["id"], "dismissed")) == 1
    assert actions.count((disliked["id"], "dismissed")) == 1 and actions.count((saved_then_undone["id"], "saved")) == 1
    fb = _ok(_request(app, "GET", f"/api/profiles/{slug}/feedback-threshold"))
    assert fb["n_saved"] == 1 and fb["n_dismissed"] == 1

    # Scan now again: the same papers come back; triage survives, nothing doubles.
    run2 = _scan(app, slug)
    mine2 = next(r for r in _ok(_request(app, "GET", f"/api/profiles/{slug}/runs")) if r["id"] == run2["run_id"])
    assert mine2["error"] is None and mine2["n_new"] == 0 and mine2["n_redup"] >= n_cards
    feed3 = _feed(app)
    assert sorted(c["id"] for c in feed3["cards"]) == sorted(c["id"] for c in cards)
    st3 = {c["id"]: c["state"] for c in feed3["cards"]}
    assert st3[liked["id"]] == "saved" and st3[disliked["id"]] == "dismissed"
    assert _one(env, "SELECT COUNT(*) AS n FROM profile_candidates WHERE profile_id = (SELECT id FROM profiles WHERE slug = ?)", slug)["n"] == 10
    # Ordering still by score, fresh and triaged alike.
    s3 = [c["score"] for c in feed3["cards"]]
    assert s3 == sorted(s3, reverse=True)

    # "Add to the seeds" from the feed: the card leaves the feed and the fit moves.
    fit_before = _one(env, "SELECT selector_config_json FROM profiles WHERE slug = ?", slug)[0]
    seeded = _ok(_request(app, "POST", f"/api/profiles/{slug}/seeds", json={"openalex_ids": [saved_then_undone["id"]]}))
    assert seeded["attached"] == 1 and seeded["rejected"] == [] and isinstance(seeded["rescored"], int)
    assert _one(env, "SELECT selector_config_json FROM profiles WHERE slug = ?", slug)[0] != fit_before
    feed4 = _feed(app)
    assert saved_then_undone["id"] not in {c["id"] for c in feed4["cards"]}
    assert len(_ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))["seeds"]) == 3 + len(oa_ids)
    # Already a seed: nothing attached, nothing rejected (the page says "already a seed").
    again = _ok(_request(app, "POST", f"/api/profiles/{slug}/seeds", json={"openalex_ids": [saved_then_undone["id"]]}))
    assert again["attached"] == 0 and again["rejected"] == []

    # Below the bar / buckets / year filter answer, never 500.
    for params in ({"bucket": "high"}, {"bucket": "medium"}, {"bucket": "low"}, {"year_from": 2030}, {"profile": "nope"}):
        body = _ok(_request(app, "GET", "/api/radar/daily", params=params))
        assert isinstance(body["cards"], list)


# ===========================================================================
# J2 -- researcher profiles
# ===========================================================================


def test_j2_researcher_profiles_end_to_end(app, env):
    # Import from Prosopia; the Profiles page polls the run.
    start = _ok(_request(app, "POST", "/api/researchers/import", json={"source": "prosopia", "ref": "sheffield-nathan"}))
    rid = start["researcher_id"]
    assert start["already_running"] is False
    run = _ok(_request(app, "GET", f"/api/import/prosopia/{start['run_id']}"))
    assert run["run"]["finished_at"] and run["run"]["error"] is None and run["result"]["researcher_id"] == rid
    assert run["result"]["drafted"] == 5 and run["run"]["profile_id"] is None

    listed = _ok(_request(app, "GET", "/api/researchers"))
    assert [r["id"] for r in listed] == [rid]
    r0 = listed[0]
    assert r0["name"] == "Nathan C. Sheffield" and r0["n_papers"] == 5 and r0["importing"] is False
    assert r0["mark"] is None and r0["position"] is None and r0["n_interests"] == 0

    detail = _ok(_request(app, "GET", f"/api/researchers/{rid}"))
    topics = detail["topics"]
    assert topics["available"] is True
    assert [t["name"] for t in topics["wanted"]] == ["Genomics and Chromatin Dynamics", "Cancer-related molecular mechanisms"]
    assert [t["id"] for t in topics["unwanted"]] == ["https://openalex.org/T10001", "https://openalex.org/T11986"]
    sugg = _ok(_request(app, "GET", f"/api/researchers/{rid}/suggestions"))
    assert sugg["researcher_id"] == rid and sugg["n_papers"] == 5

    # "New interest from this profile" with a selection.
    picked = [p["id"] for p in detail["papers"]][:3]
    made = _ok(_request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": "From Nathan", "openalex_ids": picked}))
    slug = made["draft"]["slug"]
    assert made["n_seeds"] == 3 and slug == "from-nathan"
    assert _ok(_request(app, "GET", f"/api/researchers/{rid}"))["researcher"]["n_interests"] == 1
    drafts = _ok(_request(app, "GET", "/api/profiles/drafts"))
    assert drafts[0]["slug"] == slug and drafts[0]["researcher_name"] == "Nathan C. Sheffield" \
        and drafts[0]["researcher_source"] == "prosopia" and drafts[0]["n_seeds"] == 3 and drafts[0]["phase"] == "seeded"

    # Step 3 with the profile's topics box.
    view = _ok(_request(app, "GET", f"/api/profiles/{slug}/researcher-topics"))
    assert view["researcher_id"] == rid and view["applied"] is False and view["topics"]["available"] is True
    before = {t["id"]: t for t in _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))}
    assert before["T10001"]["on"] is True and before["T10001"]["profile"] is None and before["T10001"]["source"] is None
    kept = {t["id"]: t for t in _ok(_request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}))}
    # The seeds carry T10001 too, so it is kept until the user filters it out.
    assert kept["T10001"]["on"] is True and kept["T10001"]["profile"] is None
    rows = _ok(_request(app, "GET", f"/api/profiles/{slug}/researcher-topics"))["topics"]["not_interest"]
    assert [(r["id"], r["choice"], r["overlap"]) for r in rows] == [
        ("https://openalex.org/T10001", "keep", ["openalex"]), ("https://openalex.org/T11986", "filter", [])]
    applied = {t["id"]: t for t in _ok(_request(app, "PUT", f"/api/profiles/{slug}/researcher-topics/not-interest",
                                                 json={"topic_id": "https://openalex.org/T10001", "choice": "filter"}))}
    assert applied["T10001"]["on"] is False and applied["T10001"]["profile"] == "not_interest"
    for tid in ("https://openalex.org/T10222", "https://openalex.org/T10515"):
        assert applied[tid]["on"] is True and applied[tid]["source"] == "researcher" and applied[tid]["profile"] == "interest"
    assert _ok(_request(app, "GET", f"/api/profiles/{slug}/researcher-topics"))["applied"] is True
    # The list re-read (focus) says the same.
    reread = {t["id"]: (t["on"], t["source"], t["profile"]) for t in _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))}
    assert reread == {k: (v["on"], v["source"], v["profile"]) for k, v in applied.items()}

    # Marks and order, with a second (ORCID) researcher.
    rid2 = _import_researcher(app, ref=ORCID, source="orcid")
    assert [r["id"] for r in _ok(_request(app, "GET", "/api/researchers"))] == [rid2, rid]     # by name
    me = _ok(_request(app, "PATCH", f"/api/researchers/{rid}", json={"mark": "self"}))
    assert me["mark"] == "self" and me["position"] == 0
    assert [(r["id"], r["mark"], r["position"]) for r in _ok(_request(app, "GET", "/api/researchers"))] == [(rid, "self", 0), (rid2, None, 1)]
    fol = _ok(_request(app, "PATCH", f"/api/researchers/{rid2}", json={"mark": "followed"}))
    assert fol["mark"] == "followed" and fol["position"] == 1
    order = _ok(_request(app, "PUT", "/api/researchers/order", json={"ids": [rid2, rid]}))
    assert [(r["id"], r["position"]) for r in order] == [(rid2, 0), (rid, 1)]
    assert [r["id"] for r in _ok(_request(app, "GET", "/api/researchers"))] == [rid2, rid]
    # The detail page shows the mark too.
    assert _ok(_request(app, "GET", f"/api/researchers/{rid}"))["researcher"]["mark"] == "self"

    # "Check for new papers": the source read again, saved ones marked; then
    # a refresh import of only the new ones.
    works = _ok(_request(app, "GET", f"/api/researchers/{rid}/works"))
    assert works["source"] == "prosopia" and works["key"] == "sheffield-nathan"
    assert set(works["saved_ids"]) == {w["id"] for w in works["works"]}
    n_before = _one(env, "SELECT COUNT(*) AS n FROM researcher_papers WHERE researcher_id = ?", rid)["n"]
    refreshed = _ok(_request(app, "POST", "/api/researchers/import",
                             json={"source": "prosopia", "ref": "sheffield-nathan", "paper_ids": [works["works"][0]["id"]]}))
    assert refreshed["researcher_id"] == rid and refreshed["already_running"] is False
    assert _one(env, "SELECT COUNT(*) AS n FROM researcher_papers WHERE researcher_id = ?", rid)["n"] == n_before
    after = _ok(_request(app, "GET", f"/api/researchers/{rid}"))["researcher"]
    # The refresh keeps the user's mark and place and the interest count.
    assert after["mark"] == "self" and after["position"] == 1 and after["n_interests"] == 1 and after["n_papers"] == 5
    assert [r["id"] for r in _ok(_request(app, "GET", "/api/researchers"))] == [rid2, rid]
    # ... and the interest still points at them with the topics applied.
    assert _ok(_request(app, "GET", f"/api/profiles/{slug}/researcher-topics"))["applied"] is True

    # Commit the interest (threshold step as the wizard does), with T10001 off.
    on = [t["id"] for t in applied.values() if t["on"]]
    kicked = _ok(_request(app, "POST", f"/api/profiles/draft/{slug}/dry-run", json={"days": 30, "selected_topic_ids": on}))
    _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/dry-run/{kicked['run_id']}"))
    prof = _ok(_request(app, "POST", "/api/profiles", json={"slug": slug, "threshold": 0.5, "selected_topic_ids": on}))
    assert prof["key"] == slug
    detail = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))
    assert {t["id"] for t in detail["topics"] if t["on"]} == set(on)
    assert _ok(_request(app, "GET", f"/api/researchers/{rid}"))["interests"][0]["key"] == slug

    # Forget the researcher: the marks go and the list stays as the user
    # saved it (round 12): the profile's topics become the interest's own.
    assert _ok(_request(app, "DELETE", f"/api/researchers/{rid}")) == {"ok": True}
    assert _request(app, "GET", f"/api/researchers/{rid}").status_code == 404
    assert [r["id"] for r in _ok(_request(app, "GET", "/api/researchers"))] == [rid2]
    stored = json.loads(_one(env, "SELECT topic_filters_json FROM profiles WHERE slug = ?", slug)[0])
    assert stored["researcher_topics"] is False
    kept_on = {t["id"] for t in stored["topics"] if t["on"]}
    assert kept_on == set(on)
    assert all("profile" not in t and "on_before_profile" not in t and t.get("source") != "researcher" for t in stored["topics"])
    detail = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))
    assert {t["id"] for t in detail["topics"] if t["on"]} == set(on) and all(t["profile"] is None for t in detail["topics"])
    view = _ok(_request(app, "GET", f"/api/profiles/{slug}/researcher-topics"))
    assert view["researcher_id"] is None and view["applied"] is False and view["topics"]["available"] is False
    assert _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).status_code == 400
    # Its seeds stay, in the vault too.
    assert len(detail["seeds"]) == 3
    assert _one(env, "SELECT researcher_id FROM profiles WHERE slug = ?", slug)[0] is None
    # The researcher's other papers left the vault; the seeds did not.
    vault = {d["id"] for d in _ok(_request(app, "GET", "/api/vault/docs"))}
    assert set(picked) <= vault

    # The edit page: preview keeps the adopted topics on; saving them sticks.
    preview = {t["id"]: t for t in _ok(_request(app, "GET", f"/api/profiles/{slug}/topics/preview"))}
    assert set(on) <= set(preview) and all(preview[i]["on"] for i in on)
    saved = _ok(_request(app, "PUT", f"/api/profiles/{slug}/topics", json={"selected_topic_ids": on}))
    assert {t["id"] for t in saved if t["on"]} == set(on)

    # Re-import the same person: a new row, the interest is not re-linked
    # (forgetting was the user's choice), the list and order are sane.
    rid3 = _import_researcher(app)
    assert rid3 != rid
    rows = _ok(_request(app, "GET", "/api/researchers"))
    assert [r["id"] for r in rows] == [rid2, rid3]
    assert rows[1]["mark"] is None and rows[1]["n_interests"] == 0 and rows[1]["n_papers"] == 5
    assert _ok(_request(app, "GET", f"/api/researchers/{rid3}"))["interests"] == []
    assert _ok(_request(app, "GET", f"/api/profiles/{slug}/researcher-topics"))["researcher_id"] is None
    # A new interest from the re-imported person works as the first did.
    slug2 = _ok(_request(app, "POST", f"/api/researchers/{rid3}/interests", json={"name": "From Nathan"}))["draft"]["slug"]
    assert slug2 == "from-nathan-2"
    assert _ok(_request(app, "GET", f"/api/profiles/{slug2}/researcher-topics"))["researcher_id"] == rid3
    reorder = _ok(_request(app, "PUT", "/api/researchers/order", json={"ids": [rid3, rid2]}))
    assert [r["id"] for r in reorder] == [rid3, rid2]


def test_j2_removing_the_profile_topics_on_a_draft_restores_the_seed_topics(app, env):
    rid = _import_researcher(app)
    slug = _ok(_request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": "N"}))["draft"]["slug"]
    _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))
    _ok(_request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}))
    off = {t["id"]: t for t in _ok(_request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": False}))}
    assert set(off) == {"T10001"} and off["T10001"]["on"] is True and off["T10001"]["profile"] is None
    assert _ok(_request(app, "GET", f"/api/profiles/{slug}/researcher-topics"))["applied"] is False
    # Removing again is harmless.
    assert _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": False}).status_code == 200


# ===========================================================================
# J3 -- editing a saved interest
# ===========================================================================


def test_j3_edit_a_saved_interest(app, env):
    slug, rid = _go_live(app, env, name="Edit Me")
    pid = _one(env, "SELECT id FROM profiles WHERE slug = ?", slug)["id"]
    _scan(app, slug)
    assert _feed(app, slug)["cards"]

    # Threshold tab: the stored scores, a new threshold, the feed follows.
    scores = _ok(_request(app, "POST", f"/api/profiles/{slug}/dry-run"))
    assert scores["key"] == slug and scores["n"] == 10 and len(scores["papers"]) == 10
    top = max(scores["scores"])
    tight = _ok(_request(app, "PATCH", f"/api/profiles/{slug}", json={"threshold": round(top + 0.001, 3) if top < 0.999 else 1.0}))
    assert tight["key"] == slug
    assert _feed(app, slug)["cards"] == []
    assert len(_feed(app, slug, bucket="low")["cards"]) == 10
    loose = _ok(_request(app, "PATCH", f"/api/profiles/{slug}", json={"threshold": 0.0}))
    assert loose["threshold"] == 0.0
    n_pass = len(_feed(app, slug)["cards"])
    assert n_pass > 0
    # Bad bodies: 422 (schema) or 400 (nothing), never 500.
    assert _request(app, "PATCH", f"/api/profiles/{slug}", json={"threshold": 1.5}).status_code == 422
    assert _request(app, "PATCH", f"/api/profiles/{slug}", json={"threshold": -0.1}).status_code == 422
    assert _request(app, "PATCH", f"/api/profiles/{slug}", json={}).status_code == 400
    assert _request(app, "PATCH", f"/api/profiles/{slug}", json={"name": "Renamed"}).status_code == 400
    assert _request(app, "PATCH", "/api/profiles/nope", json={"threshold": 0.5}).status_code == 404

    # Years: set, the feed narrows, clear.
    yr = _ok(_request(app, "PATCH", f"/api/profiles/{slug}", json={"years": {"year_from": 2020, "year_to": 2021}}))
    assert yr["key"] == slug and yr["yearFrom"] == 2020 and yr["yearTo"] == 2021
    assert isinstance(_feed(app, slug)["cards"], list)
    assert _request(app, "PATCH", f"/api/profiles/{slug}", json={"years": {"year_from": 2030, "year_to": None}}).status_code == 400
    assert _request(app, "PATCH", f"/api/profiles/{slug}", json={"years": {"year_from": 2022, "year_to": 2019}}).status_code == 400
    assert _request(app, "PATCH", f"/api/profiles/{slug}", json={"years": {"year_from": 1800, "year_to": None}}).status_code == 400
    cleared = _ok(_request(app, "PATCH", f"/api/profiles/{slug}", json={"years": {"year_from": None, "year_to": None}}))
    assert cleared["key"] == slug and len(_feed(app, slug)["cards"]) == n_pass

    # Seeds tab: add a researcher paper not yet a seed... all are; upload a new PDF instead.
    detail = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))
    n_seeds = len(detail["seeds"])
    fit0 = _one(env, "SELECT selector_config_json FROM profiles WHERE id = ?", pid)[0]
    up = _ok(_upload(app, b"late-seed", "late.pdf", slug))
    assert slug in up["tags"]
    detail = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))
    assert len(detail["seeds"]) == n_seeds + 1 and detail["profile"]["seeds"] == n_seeds + 1
    assert _one(env, "SELECT selector_config_json FROM profiles WHERE id = ?", pid)[0] != fit0, "a live interest is re-fitted on upload"
    # Recompute coherence (the Seeds tab's button); the edit page's preview.
    coh = _ok(_request(app, "POST", f"/api/profiles/{slug}/recompute-coherence"))
    assert coh["n"] == n_seeds + 1
    assert _request(app, "POST", f"/api/profiles/draft/{slug}/coherence").status_code == 404     # drafts only

    # Remove a seed (query-string id, as the client sends it).
    gone = up["id"]
    assert _ok(_request(app, "DELETE", f"/api/profiles/{slug}/seeds", params={"openalex_id": gone})) == {"ok": True}
    assert len(_ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))["seeds"]) == n_seeds
    # Twice: 404, not 500.
    assert _request(app, "DELETE", f"/api/profiles/{slug}/seeds", params={"openalex_id": gone}).status_code == 404
    # The paper is still in the vault, just not a seed here.
    assert gone in {d["id"] for d in _ok(_request(app, "GET", "/api/vault/docs"))}
    assert gone not in {d["id"] for d in _ok(_request(app, "GET", "/api/vault/docs", params={"tag": slug}))}
    # A bare W-id names the same seed as the full URL.
    full = next(s["id"] for s in detail["seeds"] if s["id"].startswith("https://openalex.org/W"))
    bare = full.rsplit("/", 1)[-1]
    assert _ok(_request(app, "DELETE", f"/api/profiles/{slug}/seeds", params={"openalex_id": bare})) == {"ok": True}
    assert full not in {s["id"] for s in _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))["seeds"]}
    # Put it back from the researcher's papers (Add from a saved profile).
    back = _ok(_request(app, "POST", f"/api/profiles/{slug}/seeds", json={"openalex_ids": [full]}))
    assert back["attached"] == 1 and isinstance(back["rescored"], int)

    # While a scan is running (an open manual run): removing a seed is refused
    # before anything is written; an upload still lands; refit is refused.
    conn = connect(env)
    open_run = gather_runs_repo.start(conn, profile_id=pid, user_id=1, tier_used="manual")
    conn.close()
    res = _request(app, "DELETE", f"/api/profiles/{slug}/seeds", params={"openalex_id": full})
    assert res.status_code == 409, res.text
    assert full in {s["id"] for s in _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))["seeds"]}
    assert _request(app, "POST", f"/api/profiles/{slug}/refit").status_code == 409
    assert _request(app, "POST", f"/api/profiles/{slug}/gather-now?days=7&limit=500").status_code == 409
    assert _request(app, "POST", f"/api/profiles/{slug}/backfill", json={"year_from": 2020, "year_to": 2021}).status_code == 409
    during = _ok(_upload(app, b"under-a-scan", "scan.pdf", slug))
    assert slug in during["tags"]
    assert during["id"] in {s["id"] for s in _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))["seeds"]}
    # Adding seeds under a scan attaches them and reports no rescore.
    added = _ok(_request(app, "POST", f"/api/profiles/{slug}/seeds", json={"openalex_ids": [during["id"]]}))
    assert added["attached"] == 0 and added["rescored"] is None
    conn = connect(env)
    gather_runs_repo.finish(conn, open_run, n_fetched=0, n_new=0, n_redup=0)
    conn.close()
    assert _ok(_request(app, "POST", f"/api/profiles/{slug}/refit"))["ok"] is True

    # Topics on the edit page: preview (writes nothing), save, refuse an empty set.
    preview = _ok(_request(app, "GET", f"/api/profiles/{slug}/topics/preview"))
    assert preview and all("new" in t for t in preview)
    stored_before = _one(env, "SELECT topic_filters_json FROM profiles WHERE id = ?", pid)[0]
    assert _one(env, "SELECT topic_filters_json FROM profiles WHERE id = ?", pid)[0] == stored_before
    assert _request(app, "PUT", f"/api/profiles/{slug}/topics", json={"selected_topic_ids": []}).status_code == 409
    assert _request(app, "PUT", f"/api/profiles/{slug}/topics", json={"selected_topic_ids": ["T-not-there"]}).status_code == 409
    sel = [t["id"] for t in preview][:1]
    saved = _ok(_request(app, "PUT", f"/api/profiles/{slug}/topics", json={"selected_topic_ids": sel + sel}))
    assert [t["id"] for t in saved if t["on"]] == sel
    recomputed = _ok(_request(app, "POST", f"/api/profiles/{slug}/recompute-topics"))
    assert {t["id"] for t in recomputed} >= set(sel)

    # Year backfill (a background run, inline here).
    bf = _ok(_request(app, "POST", f"/api/profiles/{slug}/backfill", json={"year_from": 2020, "year_to": 2021}))
    run = next(r for r in _ok(_request(app, "GET", f"/api/profiles/{slug}/runs")) if r["id"] == bf["run_id"])
    assert run["tier_used"] == "backfill" and run["finished_at"] and run["error"] is None and run["since_date"] == "2020-01-01"
    assert _request(app, "POST", f"/api/profiles/{slug}/backfill", json={"year_from": 2022, "year_to": 2019}).status_code == 400
    assert _request(app, "POST", f"/api/profiles/{slug}/backfill", json={"year_to": 2019}).status_code == 400
    assert _request(app, "POST", "/api/profiles/nope/backfill", json={"year_from": 2020}).status_code == 404

    # The last seed cannot go.
    for s in _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))["seeds"][:-1]:
        assert _ok(_request(app, "DELETE", f"/api/profiles/{slug}/seeds", params={"openalex_id": s["id"]})) == {"ok": True}
    last = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))["seeds"]
    assert len(last) == 1
    res = _request(app, "DELETE", f"/api/profiles/{slug}/seeds", params={"openalex_id": last[0]["id"]})
    assert res.status_code == 409, res.text
    assert len(_ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))["seeds"]) == 1
    # Reads on the page still answer.
    for path in ("detail", "runs", "feedback", "feedback-threshold", "topic-yield", "reranker-comparison", "umls", "terms", "researcher-topics"):
        _ok(_request(app, "GET", f"/api/profiles/{slug}/{path}"))


# ===========================================================================
# J4 -- two users
# ===========================================================================


def test_j4_two_users_do_not_see_each_other(app, env):
    # Both import the same person and build the same-named interest.
    slug_a, rid_a = _go_live(app, env, name="Shared")
    slug_b, rid_b = _go_live(app, env, name="Shared", headers=B)
    assert slug_a == slug_b == "shared" and rid_a != rid_b
    assert [r["id"] for r in _ok(_request(app, "GET", "/api/researchers"))] == [rid_a]
    assert [r["id"] for r in _ok(_request(app, "GET", "/api/researchers", headers=B))] == [rid_b]
    assert _request(app, "GET", f"/api/researchers/{rid_b}").status_code == 404
    assert _request(app, "PATCH", f"/api/researchers/{rid_b}", json={"mark": "self"}).status_code == 404
    assert _request(app, "DELETE", f"/api/researchers/{rid_b}").status_code == 404
    assert _request(app, "POST", f"/api/researchers/{rid_b}/interests", json={"name": "x"}).status_code == 404
    assert _request(app, "GET", f"/api/researchers/{rid_b}/works").status_code == 404
    assert _request(app, "GET", f"/api/researchers/{rid_b}/suggestions").status_code == 404
    assert _request(app, "PUT", "/api/researchers/order", json={"ids": [rid_b]}).status_code == 400
    assert _request(app, "PUT", "/api/researchers/order", json={"ids": [rid_a, rid_b]}).status_code == 400
    # B marks theirs; A's row is untouched.
    _ok(_request(app, "PATCH", f"/api/researchers/{rid_b}", json={"mark": "self"}, headers=B))
    assert _ok(_request(app, "GET", "/api/researchers"))[0]["mark"] is None

    # Same slug, two interests: each user sees their own.
    assert [p["key"] for p in _ok(_request(app, "GET", "/api/profiles"))] == ["shared"]
    assert [p["key"] for p in _ok(_request(app, "GET", "/api/profiles", headers=B))] == ["shared"]
    ida = _one(env, "SELECT id FROM profiles WHERE slug = 'shared' AND user_id = 1")["id"]
    idb = _one(env, "SELECT id FROM profiles WHERE slug = 'shared' AND user_id != 1")["id"]
    assert ida != idb

    # Both scan: the same fixture papers land in both pools.
    _scan(app, "shared")
    _scan(app, "shared", headers=B)
    fa = _feed(app)
    fb = _feed(app, headers=B)
    shared_ids = {c["id"] for c in fa["cards"]} & {c["id"] for c in fb["cards"]}
    assert shared_ids
    paper = sorted(shared_ids)[0]
    # A dismisses, B saves the same paper: no cross-talk.
    assert _ok(_request(app, "POST", "/api/radar/cards/dismiss", json={"card_id": paper, "profile": "shared", "on": True}))["state"] == "dismissed"
    assert _ok(_request(app, "POST", "/api/radar/cards/save", json={"card_id": paper, "profile": "shared", "on": True}, headers=B))["state"] == "saved"
    assert {c["id"]: c["state"] for c in _feed(app)["cards"]}[paper] == "dismissed"
    assert {c["id"]: c["state"] for c in _feed(app, headers=B)["cards"]}[paper] == "saved"
    states = {int(r["profile_id"]): (r["saved_at"], r["dismissed_at"]) for r in
              _q(env, "SELECT profile_id, saved_at, dismissed_at FROM profile_candidates WHERE openalex_id = ?", paper)}
    assert states[ida][0] is None and states[ida][1] is not None
    assert states[idb][0] is not None and states[idb][1] is None
    # Feedback events are per interest.
    assert {e["action"] for e in _ok(_request(app, "GET", "/api/profiles/shared/feedback"))} == {"dismissed"}
    assert {e["action"] for e in _ok(_request(app, "GET", "/api/profiles/shared/feedback", headers=B))} == {"saved"}
    # A paper only in B's pool is not A's card.
    only_b = [c["id"] for c in fb["cards"] if c["id"] not in {c["id"] for c in fa["cards"]}]
    if only_b:
        assert _request(app, "POST", "/api/radar/cards/save", json={"card_id": only_b[0], "profile": "shared", "on": True}).status_code == 404

    # A adds the paper as a seed: B's interest does not change.
    fit_b = _one(env, "SELECT selector_config_json, n_seed FROM profiles WHERE id = ?", idb)
    _ok(_request(app, "POST", "/api/profiles/shared/seeds", json={"openalex_ids": [paper]}))
    assert paper not in {c["id"] for c in _feed(app)["cards"]}
    assert paper in {c["id"] for c in _feed(app, headers=B)["cards"]}
    fit_b2 = _one(env, "SELECT selector_config_json, n_seed FROM profiles WHERE id = ?", idb)
    assert tuple(fit_b) == tuple(fit_b2)
    # B cannot use A's paper ids in A's... B's own interest is the only target;
    # a paper B does not own is rejected.
    local_a = _ok(_upload(app, b"only-a", "a.pdf"))["id"]
    assert local_a not in {d["id"] for d in _ok(_request(app, "GET", "/api/vault/docs", headers=B))}
    rej = _ok(_request(app, "POST", "/api/profiles/shared/seeds", json={"openalex_ids": [local_a]}, headers=B))
    assert rej == {"attached": 0, "rejected": [local_a], "rescored": None}

    # The same PDF uploaded by both: one paper row, each user lists it, each
    # user's seeds are their own.
    da = _ok(_upload(app, b"the-same-pdf", "same.pdf", "shared"))
    db_ = _ok(_upload(app, b"the-same-pdf", "same.pdf", "shared", headers=B))
    assert da["id"] == db_["id"] and da["duplicate"] is not True and db_["duplicate"] is not True
    assert da["id"] in {d["id"] for d in _ok(_request(app, "GET", "/api/vault/docs", params={"tag": "shared"}))}
    assert da["id"] in {d["id"] for d in _ok(_request(app, "GET", "/api/vault/docs", params={"tag": "shared"}, headers=B))}
    assert _q(env, "SELECT user_id FROM paper_uploads WHERE openalex_id = ?", da["id"]).__len__() == 2
    # A removes it from A's seeds; B keeps it.
    _ok(_request(app, "DELETE", "/api/profiles/shared/seeds", params={"openalex_id": da["id"]}))
    assert da["id"] not in {s["id"] for s in _ok(_request(app, "GET", "/api/profiles/shared/detail"))["seeds"]}
    assert da["id"] in {s["id"] for s in _ok(_request(app, "GET", "/api/profiles/shared/detail", headers=B))["seeds"]}
    assert da["id"] in {d["id"] for d in _ok(_request(app, "GET", "/api/vault/docs"))}, "A's upload is still A's"
    # Runs are per user: B cannot read A's run, and the other way round.
    run_a = _ok(_request(app, "GET", "/api/profiles/shared/runs"))[0]["id"]
    run_b = _ok(_request(app, "GET", "/api/profiles/shared/runs", headers=B))[0]["id"]
    assert run_a != run_b
    assert _request(app, "GET", f"/api/import/prosopia/{run_a}", headers=B).status_code == 404
    # Drafts and detail pages too.
    d_a = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": "Only A"}))["slug"]
    assert _request(app, "GET", f"/api/profiles/draft/{d_a}/topics", headers=B).status_code == 404
    assert _request(app, "DELETE", f"/api/profiles/draft/{d_a}", headers=B).status_code == 404
    assert _ok(_request(app, "GET", f"/api/profiles/{d_a}/detail", headers=B)) is None
    assert _request(app, "POST", f"/api/profiles/{d_a}/seeds", json={"openalex_ids": [paper]}, headers=B).status_code == 404
    assert _ok(_request(app, "GET", "/api/profiles/drafts", headers=B)) == []
    # B forgetting their researcher leaves A's interest's researcher link alone.
    _ok(_request(app, "DELETE", f"/api/researchers/{rid_b}", headers=B))
    assert _ok(_request(app, "GET", "/api/profiles/shared/researcher-topics"))["researcher_id"] == rid_a


# ===========================================================================
# J5 -- abnormal sequences
# ===========================================================================


def test_j5_commit_twice_and_wizard_calls_on_a_live_interest(app, env):
    slug, _rid = _go_live(app, env, name="Twice")
    # Commit again: the draft is gone; a 404, not a second live interest or a 500.
    res = _request(app, "POST", "/api/profiles", json={"slug": slug, "threshold": 0.5, "selected_topic_ids": ["T10001"]})
    assert res.status_code == 404, res.text
    assert _one(env, "SELECT COUNT(*) AS n FROM profiles")["n"] == 1
    # Wizard-only routes on a live interest.
    assert _request(app, "POST", f"/api/profiles/draft/{slug}/dry-run", json={"days": 7}).status_code == 404
    assert _request(app, "GET", f"/api/profiles/draft/{slug}/dry-run/1").status_code == 404
    assert _request(app, "GET", f"/api/profiles/draft/{slug}/topics").status_code == 404
    assert _request(app, "POST", f"/api/profiles/draft/{slug}/coherence").status_code == 404
    assert _request(app, "DELETE", f"/api/profiles/draft/{slug}").status_code == 404
    assert _request(app, "DELETE", f"/api/profiles/draft/{slug}/seeds/W1").status_code == 404
    # Live-only routes on a draft.
    d = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": "Still a draft"}))["slug"]
    assert _request(app, "POST", f"/api/profiles/{d}/gather-now").status_code == 404
    assert _request(app, "POST", f"/api/profiles/{d}/backfill", json={"year_from": 2020}).status_code == 404
    assert _request(app, "POST", f"/api/profiles/{d}/refit").status_code == 400
    assert _request(app, "POST", "/api/profiles/draft", json={"name": "Twice"}).json()["slug"] == "twice-2"
    # Unknown run id and another draft's run id.
    other = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": "Other"}))["slug"]
    _ok(_upload(app, b"o1", "o1.pdf", other))
    kicked = _ok(_request(app, "POST", f"/api/profiles/draft/{other}/dry-run", json={"days": 7}))
    assert _request(app, "GET", f"/api/profiles/draft/{d}/dry-run/{kicked['run_id']}").status_code == 404
    assert _request(app, "GET", f"/api/profiles/draft/{other}/dry-run/999999").status_code == 404
    # The trial on a draft with no seeds is an empty result, not a 500.
    empty = _ok(_request(app, "POST", f"/api/profiles/draft/{d}/dry-run", json={"days": 7}))
    st = _ok(_request(app, "GET", f"/api/profiles/draft/{d}/dry-run/{empty['run_id']}"))
    assert st["run"]["finished_at"] and st["result"] is not None and st["result"]["scores"] == []
    # Committing a draft with no seeds: refused or a live interest that scans cleanly, never 500.
    res = _request(app, "POST", "/api/profiles", json={"slug": d, "threshold": 0.5, "selected_topic_ids": []})
    assert res.status_code in (200, 400, 409), res.text


def test_j5_feedback_on_papers_not_in_the_profile(app, env):
    slug, _rid = _go_live(app, env, name="Feedback")
    _scan(app, slug)
    card = _feed(app)["cards"][0]["id"]
    # A paper that is nowhere.
    assert _request(app, "POST", "/api/radar/cards/save", json={"card_id": "W0", "profile": slug, "on": True}).status_code == 404
    # A real paper, but under another (unknown) interest.
    assert _request(app, "POST", "/api/radar/cards/save", json={"card_id": card, "profile": "nope", "on": True}).status_code == 404
    # A seed is not a card.
    seed = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))["seeds"][0]["id"]
    assert _request(app, "POST", "/api/radar/cards/dismiss", json={"card_id": seed, "profile": slug, "on": True}).status_code == 404
    # Older clients: no profile, a toggle.
    r = _ok(_request(app, "POST", "/api/radar/cards/save", json={"card_id": card}))
    assert r["state"] == "saved" and r["profile"] == slug
    assert _ok(_request(app, "POST", "/api/radar/cards/save", json={"card_id": card}))["state"] is None
    # Un-dismissing what is not dismissed changes nothing and logs nothing.
    n = len(_ok(_request(app, "GET", f"/api/profiles/{slug}/feedback")))
    assert _ok(_request(app, "POST", "/api/radar/cards/dismiss", json={"card_id": card, "profile": slug, "on": False}))["state"] is None
    assert len(_ok(_request(app, "GET", f"/api/profiles/{slug}/feedback"))) == n
    # Bad bodies.
    assert _request(app, "POST", "/api/radar/cards/save", json={}).status_code == 422
    assert _request(app, "POST", "/api/radar/cards/save", json={"card_id": card, "on": "maybe"}).status_code == 422


def test_j5_names_slugs_and_odd_payloads(app, env):
    # Empty / whitespace names.
    assert _request(app, "POST", "/api/profiles/draft", json={"name": ""}).status_code == 422
    assert _request(app, "POST", "/api/profiles/draft", json={"name": "   "}).status_code == 400
    assert _request(app, "POST", "/api/profiles/draft", json={}).status_code == 422
    rid = _import_researcher(app)
    assert _request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": "   "}).status_code == 400
    assert _request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": ""}).status_code == 422
    assert _request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": "x", "openalex_ids": []}).status_code == 400
    assert _request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": "x", "openalex_ids": ["W0"]}).status_code == 400
    assert _request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": "x", "openalex_ids": ["", "  "]}).status_code == 400
    assert _request(app, "POST", "/api/researchers/999/interests", json={"name": "x"}).status_code == 404
    # Very long names: at the limit fine, over it refused.
    long_ok = "x" * 120
    d = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": long_ok}))
    assert d["name"] == long_ok and len(d["slug"]) <= 80
    assert _request(app, "POST", "/api/profiles/draft", json={"name": "x" * 121}).status_code == 422
    d2 = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": long_ok}))
    assert d2["slug"] != d["slug"] and d2["name"] != d["name"] and len(d2["name"]) <= 120
    # Names that collide only in slug both live.
    a = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": "My Topic"}))
    b = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": "my_topic"}))
    assert a["slug"] == "my-topic" and b["slug"] == "my-topic-2" and b["name"] == "my_topic"
    # Reserved words and punctuation-only names get reachable slugs.
    for name, want in (("drafts", "drafts-profile"), ("New", "new-profile"), ("!!! ???", "interest")):
        got = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": name}))
        assert got["slug"] == want, (name, got)
        assert _ok(_request(app, "GET", f"/api/profiles/{want}/detail"))["profile"]["key"] == want
        assert _ok(_request(app, "GET", f"/api/profiles/{want}"))["key"] == want
    # Unicode: a Chinese name keeps its letters in the slug and every route
    # reaches it with the slug percent-encoded as the client sends it.
    uni = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": "神经网络 研究 / Ünïcödé"}))
    slug = uni["slug"]
    assert "神经网络" in slug and "ünïcödé" in slug
    enc = httpx.URL("http://x/").copy_with(path=f"/api/profiles/draft/{slug}/topics").raw_path.decode()
    assert "%" in enc
    _ok(_upload(app, b"uni", "uni.pdf", slug))
    _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))
    _ok(_request(app, "POST", f"/api/profiles/draft/{slug}/coherence"))
    _ok(_request(app, "GET", f"/api/profiles/{slug}/researcher-topics"))
    kicked = _ok(_request(app, "POST", f"/api/profiles/draft/{slug}/dry-run", json={"days": 7}))
    _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/dry-run/{kicked['run_id']}"))
    live = _ok(_request(app, "POST", "/api/profiles", json={"slug": slug, "threshold": 0.0, "selected_topic_ids": []}))
    assert live["key"] == slug
    _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))
    _scan(app, slug)
    assert all(c["profile"] == slug for c in _feed(app, slug)["cards"])
    doc = _ok(_upload(app, b"uni-2", "uni2.pdf", slug))
    _ok(_request(app, "DELETE", f"/api/profiles/{slug}/seeds", params={"openalex_id": doc["id"]}))
    assert _ok(_request(app, "GET", "/api/vault/tags")).get(slug) == 1

    # Topic selections with duplicate / unknown ids.
    assert _request(app, "POST", "/api/profiles", json={"slug": "my-topic", "threshold": 0.5, "selected_topic_ids": ["T1", "T1"]}).status_code in (200, 400)
    # Mark payloads.
    assert _request(app, "PATCH", f"/api/researchers/{rid}", json={}).status_code == 422
    assert _request(app, "PATCH", f"/api/researchers/{rid}", json={"mark": "me"}).status_code == 422
    assert _request(app, "PUT", "/api/researchers/order", json={"ids": [rid, rid]}).status_code == 400
    assert _request(app, "PUT", "/api/researchers/order", json={"ids": ["a"]}).status_code == 422
    assert _request(app, "PUT", "/api/researchers/order", json={"ids": [rid, 10**12]}).status_code == 400
    assert _ok(_request(app, "PUT", "/api/researchers/order", json={"ids": [rid]}))[0]["position"] == 0
    # Import payloads.
    assert _request(app, "POST", "/api/researchers/import", json={"source": "github", "ref": "x"}).status_code == 400
    assert _request(app, "POST", "/api/researchers/import", json={"source": "orcid", "ref": "not-an-orcid"}).status_code == 400
    assert _request(app, "POST", "/api/researchers/import", json={"source": "prosopia", "ref": "   "}).status_code == 400
    assert _request(app, "POST", "/api/researchers/import", json={"source": "prosopia", "ref": "sheffield-nathan", "paper_ids": ["nope"]}).status_code == 404
    assert _request(app, "POST", "/api/researchers/import", json={"source": "prosopia", "ref": "sheffield-nathan", "paper_ids": []}).status_code == 400
    # Bad emails.
    assert _request(app, "GET", "/api/profiles", headers={"X-User-Email": "not-an-email"}).status_code == 400
    # Seeds payloads.
    assert _request(app, "POST", "/api/profiles/my-topic/seeds", json={"openalex_ids": []}).status_code == 400
    assert _request(app, "POST", "/api/profiles/my-topic/seeds", json={"openalex_ids": ["", " "]}).status_code == 400
    assert _request(app, "POST", "/api/profiles/my-topic/seeds", json={}).status_code == 422
    assert _request(app, "DELETE", "/api/profiles/my-topic/seeds").status_code == 422
    assert _request(app, "DELETE", "/api/profiles/my-topic/seeds", params={"openalex_id": ""}).status_code == 422
    assert _request(app, "DELETE", "/api/profiles/my-topic/seeds", params={"openalex_id": "W1"}).status_code == 404
    # Gather-now params.
    assert _request(app, "POST", "/api/profiles/my-topic/gather-now?days=0").status_code in (404, 422)
    assert _request(app, "POST", "/api/profiles/my-topic/gather-now?limit=99999").status_code in (404, 422)


def test_j5_delete_draft_journeys(app, env):
    rid = _import_researcher(app)
    slug = _ok(_request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": "Throwaway"}))["draft"]["slug"]
    _ok(_upload(app, b"tw", "tw.pdf", slug))
    _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))
    _ok(_request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}))
    kicked = _ok(_request(app, "POST", f"/api/profiles/draft/{slug}/dry-run", json={"days": 7}))
    assert _ok(_request(app, "DELETE", f"/api/profiles/draft/{slug}")) == {"ok": True}
    assert _request(app, "DELETE", f"/api/profiles/draft/{slug}").status_code == 404
    assert _ok(_request(app, "GET", "/api/profiles/drafts")) == []
    # The researcher and their papers stay; the trial's run row went with the draft (or is unreachable).
    r = _ok(_request(app, "GET", f"/api/researchers/{rid}"))
    assert r["researcher"]["n_interests"] == 0 and r["researcher"]["n_papers"] == 5 and r["researcher"]["importing"] is False
    assert _request(app, "GET", f"/api/profiles/draft/{slug}/dry-run/{kicked['run_id']}").status_code == 404
    # The uploaded PDF is still in the vault, untagged.
    docs = _ok(_request(app, "GET", "/api/vault/docs"))
    assert any(d["title"].startswith("Uploaded") and d["tags"] == [] for d in docs)
    # The name is free again.
    assert _ok(_request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": "Throwaway"}))["draft"]["slug"] == "throwaway"


# ===========================================================================
# J1b -- the wizard's own import sources (Prosopia / ORCID), and adding more
# ===========================================================================


def test_j1b_wizard_imports_from_prosopia_then_adds_an_orcid(app, env):
    # "A Prosopia profile" as the seed source: the draft exists at once.
    start = _ok(_request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan", "name": "Nathan's work"}))
    slug, run_id = start["draft_slug"], start["run_id"]
    assert slug == "nathan-s-work" and start["already_running"] is False
    st = _ok(_request(app, "GET", f"/api/import/prosopia/{run_id}"))
    assert st["run"]["finished_at"] and st["run"]["error"] is None and st["run"]["profile_id"] is not None
    assert st["result"]["draft_slug"] == slug and st["result"]["drafted"] == 5 and st["result"]["unresolved"] == ["ghost2020nowhere"]
    drafts = _ok(_request(app, "GET", "/api/profiles/drafts"))
    assert drafts[0]["slug"] == slug and drafts[0]["phase"] == "seeded" and drafts[0]["n_seeds"] == 5
    assert drafts[0]["researcher_source"] == "prosopia" and drafts[0]["researcher_name"] == "Nathan C. Sheffield"
    # The researcher was recorded, with the draft as their interest.
    rs = _ok(_request(app, "GET", "/api/researchers"))
    assert len(rs) == 1 and rs[0]["n_papers"] == 5 and rs[0]["n_interests"] == 1 and rs[0]["importing"] is False
    rid = rs[0]["id"]
    seeds = {d["id"] for d in _ok(_request(app, "GET", "/api/vault/docs", params={"tag": slug}))}
    assert len(seeds) == 5
    # Prosopia's topics box works on this draft.
    view = _ok(_request(app, "GET", f"/api/profiles/{slug}/researcher-topics"))
    assert view["researcher_id"] == rid and view["topics"]["available"] is True
    _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))
    applied = _ok(_request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}))
    assert any(t["source"] == "researcher" for t in applied)

    # "Add more": the works of an ORCID, picked, into this draft.
    works = _ok(_request(app, "GET", f"/api/import/orcid/{ORCID}/works"))
    assert works["orcid"] == ORCID and len(works["works"]) == 3
    picked = [w["openalex_id"] for w in works["works"]][:2]
    more = _ok(_request(app, "POST", "/api/import/orcid", json={"orcid": ORCID, "openalex_ids": picked, "into": slug}))
    assert more["draft_slug"] == slug and more["already_running"] is False
    st2 = _ok(_request(app, "GET", f"/api/import/prosopia/{more['run_id']}"))
    assert st2["run"]["error"] is None and st2["result"]["drafted"] == 2
    assert len(_ok(_request(app, "GET", "/api/vault/docs", params={"tag": slug}))) == 7
    # The draft stays Nathan's: the topics box still names him and the
    # applied profile is intact; the ORCID person was recorded separately.
    view = _ok(_request(app, "GET", f"/api/profiles/{slug}/researcher-topics"))
    assert view["researcher_id"] == rid and view["applied"] is True
    rs = {r["source"]: r for r in _ok(_request(app, "GET", "/api/researchers"))}
    assert set(rs) == {"prosopia", "orcid"} and rs["orcid"]["n_papers"] == 2 and rs["orcid"]["n_interests"] == 0
    assert _ok(_request(app, "GET", "/api/profiles/drafts"))[0]["n_seeds"] == 7
    # ``into`` a draft that is not this user's / not a draft: 404.
    assert _request(app, "POST", "/api/import/orcid", json={"orcid": ORCID, "openalex_ids": picked, "into": "nope"}).status_code == 404
    assert _request(app, "POST", "/api/import/orcid", json={"orcid": ORCID, "openalex_ids": picked, "into": slug}, headers=B).status_code == 404
    assert _request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan", "into": slug}, headers=B).status_code == 404
    # The same Prosopia profile again into the same draft: nothing doubles.
    again = _ok(_request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan", "into": slug}))
    assert again["draft_slug"] == slug
    assert len(_ok(_request(app, "GET", "/api/vault/docs", params={"tag": slug}))) == 7
    assert _one(env, "SELECT COUNT(*) AS n FROM researchers")["n"] == 2

    # Through the rest of the wizard and live.
    topics = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))
    on = [t["id"] for t in topics if t["on"]]
    kicked = _ok(_request(app, "POST", f"/api/profiles/draft/{slug}/dry-run", json={"days": 30, "selected_topic_ids": on}))
    res = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/dry-run/{kicked['run_id']}"))
    assert res["result"] and res["result"]["papers"]
    # A bad cron is refused before the draft is flipped; the retry works.
    bad = _request(app, "POST", "/api/profiles", json={"slug": slug, "threshold": 0.5, "selected_topic_ids": on, "cron": "not a cron", "tz": "UTC"})
    assert bad.status_code == 400, bad.text
    assert _one(env, "SELECT is_draft FROM profiles WHERE slug = ?", slug)["is_draft"] == 1
    bad = _request(app, "POST", "/api/profiles", json={"slug": slug, "threshold": 0.5, "selected_topic_ids": on, "tz": "Mars/Olympus"})
    assert bad.status_code == 400, bad.text
    live = _ok(_request(app, "POST", "/api/profiles", json={"slug": slug, "threshold": 0.0, "selected_topic_ids": on, "cron": "0 4 * * *", "tz": "UTC",
                                                           "years": {"year_from": 2020, "year_to": 2021}}))
    assert live["isDraft"] is False and live["seeds"] == 7 and live["yearFrom"] == 2020 and live["yearTo"] == 2021
    assert live["researcherId"] == rid
    # The wizard then starts the backfill for the years chosen.
    bf = _ok(_request(app, "POST", f"/api/profiles/{slug}/backfill", json={"year_from": 2020, "year_to": 2021}))
    run = next(r for r in _ok(_request(app, "GET", f"/api/profiles/{slug}/runs")) if r["id"] == bf["run_id"])
    assert run["error"] is None and run["tier_used"] == "backfill"
    # The Scans list shows the imports too, as the researcher's history (not as scans).
    tiers = {r["tier_used"] for r in _ok(_request(app, "GET", f"/api/profiles/{slug}/runs"))}
    assert "backfill" in tiers
    assert _ok(_request(app, "GET", f"/api/researchers/{rid}"))["interests"][0]["key"] == slug


def test_j5_everything_waits_while_an_import_fills_the_draft(app, env):
    rid = _import_researcher(app)
    slug = _ok(_request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": "Filling"}))["draft"]["slug"]
    pid = _one(env, "SELECT id FROM profiles WHERE slug = ?", slug)["id"]
    _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))
    seed = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))["seeds"][0]["id"]
    conn = connect(env)
    run_id = gather_runs_repo.start(conn, profile_id=pid, user_id=1, tier_used="prosopia_import", researcher_id=rid)
    from rag_lib.db.repos import researchers as researchers_repo
    researchers_repo.set_last_run(conn, rid, run_id)      # as launch_import does
    conn.close()
    d = _ok(_request(app, "GET", "/api/profiles/drafts"))[0]
    assert d["phase"] == "importing" and d["import_run_id"] == run_id and d["import_error"] is None
    # Every step that reads the seeds waits; nothing is written.
    assert _request(app, "POST", f"/api/profiles/draft/{slug}/coherence").status_code == 409
    assert _request(app, "GET", f"/api/profiles/draft/{slug}/topics").status_code == 409
    assert _request(app, "POST", f"/api/profiles/draft/{slug}/dry-run", json={"days": 7}).status_code == 409
    assert _request(app, "POST", "/api/profiles", json={"slug": slug, "threshold": 0.5, "selected_topic_ids": ["T10001"]}).status_code == 409
    assert _request(app, "DELETE", f"/api/profiles/draft/{slug}").status_code == 409
    assert _upload(app, b"mid-import", "mid.pdf", slug).status_code == 409
    assert _request(app, "POST", f"/api/profiles/{slug}/seeds", json={"openalex_ids": [seed]}).status_code == 409
    assert _request(app, "DELETE", f"/api/profiles/{slug}/seeds", params={"openalex_id": seed}).status_code == 409
    assert _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).status_code == 409
    assert _request(app, "POST", f"/api/profiles/{slug}/recompute-topics").status_code == 409
    # The researcher is "importing": not forgotten, no new interest, no second import.
    assert _ok(_request(app, "GET", f"/api/researchers/{rid}"))["researcher"]["importing"] is True
    assert _request(app, "DELETE", f"/api/researchers/{rid}").status_code == 409
    assert _request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": "x"}).status_code == 409
    joined = _ok(_request(app, "POST", "/api/researchers/import", json={"source": "prosopia", "ref": "sheffield-nathan"}))
    assert joined["already_running"] is True and joined["run_id"] == run_id
    # A new-draft import of the same person joins the run filling their draft (designed).
    joined1 = _ok(_request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan"}))
    assert joined1["already_running"] is True and joined1["draft_slug"] == slug and joined1["run_id"] == run_id
    joined2 = _ok(_request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan", "into": slug}))
    assert joined2["already_running"] is True and joined2["draft_slug"] == slug
    assert _one(env, "SELECT is_draft FROM profiles WHERE id = ?", pid)["is_draft"] == 1
    assert _one(env, "SELECT COUNT(*) AS n FROM profiles")["n"] == 1
    # The import fails: the draft says so, and the user can carry on.
    conn = connect(env)
    gather_runs_repo.finish(conn, run_id, n_fetched=0, n_new=0, n_redup=0, error="ProsopiaError: connection reset")
    conn.close()
    d = _ok(_request(app, "GET", "/api/profiles/drafts"))[0]
    # Seeds are attached, so the phase is "seeded" with the error alongside (documented).
    assert d["phase"] == "seeded" and d["import_error"] and d["import_run_id"] is None
    assert _ok(_request(app, "GET", f"/api/researchers/{rid}"))["researcher"]["importing"] is False
    assert _ok(_request(app, "GET", f"/api/researchers/{rid}"))["researcher"]["last_error"]
    _ok(_request(app, "POST", f"/api/profiles/draft/{slug}/coherence"))
    _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))
    assert _ok(_request(app, "GET", "/api/profiles/drafts"))[0]["phase"] == "seeded"
    assert _ok(_request(app, "DELETE", f"/api/researchers/{rid}")) == {"ok": True}


def test_j5_stale_tabs_patch_and_scan_drafts(app, env):
    """A tab left on an interest that was... never live: PATCH / scan on a draft."""
    d = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": "Draft"}))["slug"]
    _ok(_upload(app, b"d1", "d1.pdf", d))
    # PATCH on a draft: whatever the answer, not a 500, and the draft stays a draft.
    res = _request(app, "PATCH", f"/api/profiles/{d}", json={"threshold": 0.3})
    assert res.status_code in (200, 400, 404, 409), res.text
    assert _one(env, "SELECT is_draft FROM profiles WHERE slug = ?", d)["is_draft"] == 1
    res = _request(app, "PATCH", f"/api/profiles/{d}", json={"years": {"year_from": 2020, "year_to": 2021}})
    assert res.status_code in (200, 400, 404, 409), res.text
    for path, method, body in (("topics/preview", "GET", None), ("topics", "PUT", {"selected_topic_ids": []}),
                               ("recompute-coherence", "POST", None), ("recompute-topics", "POST", None),
                               ("dry-run", "POST", None), ("feedback-threshold", "GET", None), ("runs", "GET", None),
                               ("feedback", "GET", None), ("topic-yield", "GET", None), ("reranker-comparison", "GET", None),
                               ("terms", "GET", None), ("umls", "GET", None), ("umls", "POST", None)):
        res = _request(app, method, f"/api/profiles/{d}/{path}", json=body) if body is not None else _request(app, method, f"/api/profiles/{d}/{path}")
        assert res.status_code < 500, (path, res.status_code, res.text)
    assert _ok(_request(app, "GET", f"/api/profiles/{d}"))["isDraft"] is True
    assert _ok(_request(app, "GET", "/api/radar/daily", params={"profile": d}))["cards"] == []


def test_j4_one_user_two_interests_share_a_paper(app, env):
    """The feed with two interests: fair shares, each card its own state."""
    a, _ = _go_live(app, env, name="A")
    b = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": "B"}))["slug"]
    _ok(_upload(app, b"b-seed", "b.pdf", b))
    kicked = _ok(_request(app, "POST", f"/api/profiles/draft/{b}/dry-run", json={"days": 7}))
    _ok(_request(app, "GET", f"/api/profiles/draft/{b}/dry-run/{kicked['run_id']}"))
    _ok(_request(app, "POST", "/api/profiles", json={"slug": b, "threshold": 0.0, "selected_topic_ids": []}))
    _scan(app, a)
    _scan(app, b)
    feed = _feed(app)
    by = {}
    for c in feed["cards"]:
        by.setdefault(c["id"], {})[c["profile"]] = c
    shared = [pid for pid, m in by.items() if len(m) == 2]
    assert shared, "the fixture gatherer gives both interests the same papers"
    paper = shared[0]
    assert _ok(_request(app, "POST", "/api/radar/cards/save", json={"card_id": paper, "profile": a, "on": True}))["state"] == "saved"
    feed = _feed(app)
    cards = {c["profile"]: c for c in feed["cards"] if c["id"] == paper}
    assert cards[a]["state"] == "saved" and cards[b]["state"] is None
    assert _one(env, "SELECT saved_at FROM profile_candidates WHERE openalex_id = ? AND profile_id = (SELECT id FROM profiles WHERE slug = ?)", paper, b)["saved_at"] is None
    # Per-interest feeds agree.
    assert {c["id"]: c["state"] for c in _feed(app, a)["cards"]}[paper] == "saved"
    assert {c["id"]: c["state"] for c in _feed(app, b)["cards"]}[paper] is None
    # Dismiss in B, un-save in A: independent.
    assert _ok(_request(app, "POST", "/api/radar/cards/dismiss", json={"card_id": paper, "profile": b, "on": True}))["state"] == "dismissed"
    assert _ok(_request(app, "POST", "/api/radar/cards/save", json={"card_id": paper, "profile": a, "on": False}))["state"] is None
    cards = {c["profile"]: c for c in _feed(app)["cards"] if c["id"] == paper}
    assert cards[a]["state"] is None and cards[b]["state"] == "dismissed"
    # Seeding it in A leaves B's card.
    _ok(_request(app, "POST", f"/api/profiles/{a}/seeds", json={"openalex_ids": [paper]}))
    cards = {c["profile"]: c for c in _feed(app)["cards"] if c["id"] == paper}
    assert set(cards) == {b}


def test_j2_refresh_brings_only_the_new_paper_and_leaves_interests_alone(app, env):
    from tests.test_prosopia_import import _record, _work

    rid = _import_researcher(app)
    slug = _ok(_request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": "All of Nathan"}))["draft"]["slug"]
    assert len(_ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))["seeds"]) == 5
    # The source gains a paper.
    app.state.prosopia.profile.papers.append(_record(name="A new preprint", summary="New.", paper_id="new2026",
                                                     doi="10.1/new", datePublished="2026"))
    app.state.openalex.by_doi["10.1/new"] = _work("10.1/new", "W999000111", "A new preprint", 2026)
    works = _ok(_request(app, "GET", f"/api/researchers/{rid}/works"))
    fresh = [w for w in works["works"] if w["id"] not in works["saved_ids"]]
    assert [w["id"] for w in fresh] == ["new2026"]
    started = _ok(_request(app, "POST", "/api/researchers/import",
                           json={"source": "prosopia", "ref": "sheffield-nathan", "paper_ids": ["new2026"]}))
    assert started["researcher_id"] == rid and started["already_running"] is False
    st = _ok(_request(app, "GET", f"/api/import/prosopia/{started['run_id']}"))
    assert st["run"]["error"] is None and st["result"]["drafted"] == 1 and st["run"]["researcher_id"] == rid
    r = _ok(_request(app, "GET", f"/api/researchers/{rid}"))
    assert r["researcher"]["n_papers"] == 6 and r["researcher"]["last_run_id"] == started["run_id"]
    assert "https://openalex.org/W999000111" in {p["id"] for p in r["papers"]}
    # The interest built earlier keeps its 5 seeds; the new paper can be added by hand.
    assert len(_ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))["seeds"]) == 5
    assert "https://openalex.org/W999000111" in {d["id"] for d in _ok(_request(app, "GET", "/api/vault/docs"))}
    added = _ok(_request(app, "POST", f"/api/profiles/{slug}/seeds", json={"openalex_ids": ["https://openalex.org/W999000111"]}))
    assert added["attached"] == 1
    assert _ok(_request(app, "GET", f"/api/researchers/{rid}/works"))["saved_ids"].__len__() == 6
    # Another user does not see the paper in their vault, cannot seed with it.
    assert "https://openalex.org/W999000111" not in {d["id"] for d in _ok(_request(app, "GET", "/api/vault/docs", headers=B))}


def test_settings_page_mailto(app, env):
    me = _ok(_request(app, "GET", "/api/users/me"))
    assert me["email"] == "demo@example.com" and me["mailto"] == "demo@example.com"
    assert _ok(_request(app, "PATCH", "/api/users/me", json={"mailto": "contact@example.org"}))["mailto"] == "contact@example.org"
    assert _request(app, "PATCH", "/api/users/me", json={"mailto": "not an email"}).status_code == 400
    assert _request(app, "PATCH", "/api/users/me", json={"mailto": "   "}).status_code in (400, 422)
    assert _ok(_request(app, "GET", "/api/users/me"))["mailto"] == "contact@example.org"
    other = _ok(_request(app, "GET", "/api/users/me", headers=B))
    assert other["email"] == "b@example.com" and other["id"] != me["id"]
    assert _ok(_request(app, "GET", "/api/users/me", headers={"X-User-Email": " B@Example.COM "}))["id"] == other["id"]


def test_j3_backfill_then_the_feeds_year_filter(app, env):
    slug, _rid = _go_live(app, env, name="Years", threshold=0.0)
    bf = _ok(_request(app, "POST", f"/api/profiles/{slug}/backfill", json={"year_from": 2020, "year_to": 2021}))
    run = next(r for r in _ok(_request(app, "GET", f"/api/profiles/{slug}/runs")) if r["id"] == bf["run_id"])
    assert run["error"] is None and run["n_fetched"] == 10
    years = {c["id"]: c["date"][:4] for c in _feed(app, slug)["cards"]}
    assert set(years.values()) <= {"2020", "2021"} and len(years) >= 2
    later = _feed(app, slug, year_from=2021)["cards"]
    assert later and all(c["date"].startswith("2021") for c in later)
    assert _feed(app, slug, year_from=2022)["cards"] == []
    # The interest's own range narrows the feed the same way, and clears.
    _ok(_request(app, "PATCH", f"/api/profiles/{slug}", json={"years": {"year_from": 2021, "year_to": 2021}}))
    assert {c["date"][:4] for c in _feed(app, slug)["cards"]} == {"2021"}
    _ok(_request(app, "PATCH", f"/api/profiles/{slug}", json={"years": {"year_from": None, "year_to": None}}))
    assert len(_feed(app, slug)["cards"]) == len(years)


def test_j5_tiny_drafts_zero_and_one_seed(app, env):
    # No seeds at all: every wizard step answers, nothing 500s.
    d0 = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": "Empty"}))["slug"]
    r = _request(app, "POST", f"/api/profiles/draft/{d0}/coherence")
    assert r.status_code < 500, r.text
    assert _ok(_request(app, "GET", f"/api/profiles/draft/{d0}/topics")) == []
    kicked = _ok(_request(app, "POST", f"/api/profiles/draft/{d0}/dry-run", json={"days": 7}))
    assert _ok(_request(app, "GET", f"/api/profiles/draft/{d0}/dry-run/{kicked['run_id']}"))["run"]["error"] is None
    r = _request(app, "POST", "/api/profiles", json={"slug": d0, "threshold": 0.5, "selected_topic_ids": []})
    assert r.status_code < 500, r.text
    if r.status_code == 200:
        # Live with no seeds: a scan must not 500 and must close its run.
        run = _scan(app, d0)
        mine = next(x for x in _ok(_request(app, "GET", f"/api/profiles/{d0}/runs")) if x["id"] == run["run_id"])
        assert mine["finished_at"]
        assert isinstance(_feed(app, d0)["cards"], list)

    # One seed: coherence has nothing to compare; the rest works.
    d1 = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": "One"}))["slug"]
    one = _ok(_upload(app, b"only-seed", "only.pdf", d1))
    coh = _request(app, "POST", f"/api/profiles/draft/{d1}/coherence")
    assert coh.status_code < 500, coh.text
    if coh.status_code == 200:
        assert coh.json()["n"] == 1
    kicked = _ok(_request(app, "POST", f"/api/profiles/draft/{d1}/dry-run", json={"days": 7}))
    st = _ok(_request(app, "GET", f"/api/profiles/draft/{d1}/dry-run/{kicked['run_id']}"))
    assert st["run"]["error"] is None and st["result"]["scores"]
    live = _ok(_request(app, "POST", "/api/profiles", json={"slug": d1, "threshold": 0.0, "selected_topic_ids": []}))
    assert live["seeds"] == 1 and live["seedSimMin"] is None
    _scan(app, d1)
    assert _feed(app, d1)["cards"]
    detail = _ok(_request(app, "GET", f"/api/profiles/{d1}/detail"))
    assert len(detail["seeds"]) == 1 and detail["seeds"][0]["id"] == one["id"]
    assert _ok(_request(app, "POST", f"/api/profiles/{d1}/recompute-coherence"))["n"] == 1
    for bucket in ("high", "medium", "low"):
        assert isinstance(_feed(app, d1, bucket=bucket)["cards"], list)
    # Its only seed cannot go; removing a draft's only seed can.
    assert _request(app, "DELETE", f"/api/profiles/{d1}/seeds", params={"openalex_id": one["id"]}).status_code == 409
    d2 = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": "Two"}))["slug"]
    two = _ok(_upload(app, b"only-seed-2", "only2.pdf", d2))
    assert _ok(_request(app, "DELETE", f"/api/profiles/{d2}/seeds", params={"openalex_id": two["id"]})) == {"ok": True}
    assert _ok(_request(app, "GET", "/api/profiles/drafts"))[0]["n_seeds"] == 0
