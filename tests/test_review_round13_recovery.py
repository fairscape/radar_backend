"""Review round 13 (c): restart recovery and in-memory job flags.

1. After a hard kill mid-scan / mid-trial / mid-import, the next boot (with
   the scheduler on) closes every open run as interrupted and a new scan can
   be started and runs.
2. ``SEED_UMLS_RUNNING`` is set before the job's try/finally: a failure to
   open the database leaves the interest flagged "running" for the life of
   the process, and ``POST /{key}/umls`` then never starts extraction again.
3. A pre-created scan run whose job cannot open the database stays open and
   counts as a running scan (409 on "Scan now") for SCAN_STALE_MINUTES.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from rag_lib.api import settings as settings_module
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import (
    gather_runs as gather_runs_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    researchers as researchers_repo,
    users as users_repo,
)
from rag_lib.paper import Paper
from rag_lib.persistence import store_profile_from_object
from rag_lib.profile import Profile
from rag_lib.scheduler import jobs as scheduler_jobs


def _seed_paper(oa_id: str = "W1") -> Paper:
    p = Paper(doi=f"10.1/{oa_id.lower()}", openalex_id=oa_id, title=f"Seed {oa_id}",
              abstract="abstract", year=2024, source="openalex_gatherer")
    p.embeddings["placeholder-v1"] = np.random.RandomState(1).rand(8).astype(float).tolist()
    return p


def _live_profile(conn, user_id: int, name: str) -> int:
    return store_profile_from_object(
        conn, user_id=user_id,
        profile=Profile(name=name, papers=[_seed_paper()], embedding_model="placeholder-v1",
                        topic_filters={"topics": [{"id": "T1", "display_name": "Topic", "count": 1}]},
                        selector_config={"type": "centroid", "embedding_model": "placeholder-v1",
                                         "centroid": [0.1] * 8, "diagnostics_snapshot": {"status": "fit"}}),
    )


def _request(app, method, path, **kw):
    async def _run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
                return await c.request(method, path, **kw)
    return asyncio.run(_run())


@pytest.fixture()
def env(tmp_path, monkeypatch):
    db = tmp_path / "radar.db"
    for k, v in {"RADAR_DB_PATH": str(db), "RADAR_VAULT_DIR": str(tmp_path / "vault"),
                 "RADAR_CHROMA_DIR": str(tmp_path / "chroma"), "RADAR_LOG_JSON": "false",
                 "RADAR_UMLS_ENABLED": "false", "RADAR_SQLITE_JOURNAL_MODE": "wal",
                 "RADAR_DEFAULT_EMBEDDING_MODEL": "placeholder-v1"}.items():
        monkeypatch.setenv(k, v)
    settings_module.get_settings.cache_clear()
    yield db
    settings_module.get_settings.cache_clear()


# ---------------------------------------------------------------------------
# 1. Boot after a kill -9 with one run of every kind still open
# ---------------------------------------------------------------------------


def test_boot_closes_every_open_run_and_a_new_scan_can_start(env, monkeypatch):
    monkeypatch.setenv("RADAR_SCHEDULER_ENABLED", "true")
    settings_module.get_settings.cache_clear()
    conn = connect(env)
    apply_migrations(conn)
    uid = int(users_repo.upsert(conn, "demo@example.com")["id"])
    live = _live_profile(conn, uid, "Live One")
    draft = profiles_repo.upsert(conn, user_id=uid, name="Draft One", embedding_model="placeholder-v1",
                                 n_seed=0, topic_filters={"topics": []}, threshold=0.8)
    conn.execute("UPDATE profiles SET is_draft = 1 WHERE id = ?", (draft,))
    conn.commit()
    rid = researchers_repo.upsert(conn, user_id=uid, source="orcid", key="0000-0001-0000-0001",
                                  name="R", orcid="0000-0001-0000-0001")
    open_runs = {
        "scan": gather_runs_repo.start(conn, profile_id=live, user_id=uid, tier_used="manual"),
        "backfill": gather_runs_repo.start(conn, profile_id=live, user_id=uid, tier_used="backfill"),
        "dry_run": gather_runs_repo.start(conn, profile_id=draft, user_id=uid, tier_used="dry_run"),
        "import": gather_runs_repo.start(conn, profile_id=draft, user_id=uid, tier_used="prosopia_import",
                                         researcher_id=rid),
        "researcher_import": gather_runs_repo.start(conn, profile_id=None, user_id=uid,
                                                    tier_used="researcher_import", researcher_id=rid),
    }
    researchers_repo.set_last_run(conn, rid, open_runs["researcher_import"])
    for run_id in open_runs.values():
        gather_runs_repo.set_step(conn, run_id, "embedding", n_total=10, message="half way")
    conn.close()

    # Before the boot: everything blocks.
    conn = connect(env)
    assert gather_runs_repo.running_scan(conn, live) is not None
    assert gather_runs_repo.has_unfinished_import(conn, draft)
    assert gather_runs_repo.is_inflight(conn, open_runs["researcher_import"])
    conn.close()

    # The scan job is stubbed: no OpenAlex. It closes the row it is handed.
    ran: list[int] = []

    def fake_gather(user_id, profile_id, *, settings=None, run_id=None, **kw):
        c = connect(env)
        try:
            gather_runs_repo.finish(c, run_id, n_fetched=0, n_new=0, n_redup=0, tier_used="manual")
        finally:
            c.close()
        ran.append(run_id)
        return run_id

    monkeypatch.setattr(scheduler_jobs, "gather_for_profile", fake_gather)
    from rag_lib.api.app import create_app
    app = create_app()

    async def _run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
                runs = (await c.get("/api/profiles/live-one/runs")).json()
                researcher = (await c.get(f"/api/researchers/{rid}")).json()
                draft_state = (await c.get("/api/profiles/draft/draft-one/dry-run/%d" % open_runs["dry_run"])).json()
                started = await c.post("/api/profiles/live-one/gather-now")
                deadline = time.monotonic() + 10
                while not ran and time.monotonic() < deadline:
                    await asyncio.sleep(0.1)
                after = (await c.get("/api/profiles/live-one/runs")).json()
                return runs, researcher, draft_state, started, after
    runs, researcher, draft_state, started, after = asyncio.run(_run())

    conn = connect(env)
    try:
        for kind, run_id in open_runs.items():
            row = gather_runs_repo.get(conn, run_id)
            assert row["finished_at"] is not None, kind
            assert row["error"] == gather_runs_repo.INTERRUPTED, kind
        assert gather_runs_repo.running_scan(conn, live) is None
        assert not gather_runs_repo.has_unfinished_import(conn, draft)
    finally:
        conn.close()
    assert {r["id"] for r in runs} >= {open_runs["scan"], open_runs["backfill"]}
    assert all(r["error"] == gather_runs_repo.INTERRUPTED for r in runs if r["id"] in open_runs.values())
    assert researcher["researcher"]["importing"] is False
    assert draft_state["run"]["error"] == gather_runs_repo.INTERRUPTED
    assert started.status_code == 200, started.text
    assert ran == [started.json()["run_id"]]
    assert [r for r in after if r["id"] == ran[0]][0]["finished_at"] is not None


# ---------------------------------------------------------------------------
# 2. SEED_UMLS_RUNNING is cleared however the job dies, and expires
# ---------------------------------------------------------------------------


def test_seed_umls_flag_is_cleared_when_the_job_cannot_open_the_db(env, monkeypatch):
    monkeypatch.setenv("RADAR_SCHEDULER_ENABLED", "false")
    monkeypatch.setenv("RADAR_UMLS_ENABLED", "true")
    settings_module.get_settings.cache_clear()
    conn = connect(env)
    apply_migrations(conn)
    uid = int(users_repo.upsert(conn, "demo@example.com")["id"])
    pid = profiles_repo.upsert(conn, user_id=uid, name="Chromatin", embedding_model="placeholder-v1",
                               n_seed=1, topic_filters={"topics": []}, threshold=0.8)
    papers_repo.upsert(conn, {"openalex_id": "W1", "title": "Paper", "abstract": "a", "source": "orcid"})
    profiles_repo.attach_seed(conn, pid, "W1")
    conn.close()
    scheduler_jobs.SEED_UMLS_RUNNING.clear()

    # The first attempt: the database cannot be opened (fd exhaustion, the
    # /var/tmp directory gone -- both have happened on this host).
    calls = {"n": 0}
    real_connect = scheduler_jobs.connect

    def flaky_connect(path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("unable to open database file")
        return real_connect(path)

    monkeypatch.setattr(scheduler_jobs, "connect", flaky_connect)
    # The job swallows the failure (the seeds stay pending, the sweep retries)
    # and the flag does not outlive it.
    assert scheduler_jobs.extract_seed_umls_job(pid, SimpleNamespace(RADAR_DB_PATH=env, RADAR_UMLS_ENABLED=True)) == 0
    assert pid not in scheduler_jobs.SEED_UMLS_RUNNING

    # At the API: not "running", and POST starts the extraction again.
    from rag_lib.api.services import vault

    def fake_extract(conn, settings, oa, text):
        conn.execute("UPDATE papers SET umls_concepts_json = '[]' WHERE openalex_id = ?", (oa,))
        conn.commit()
        return True

    monkeypatch.setattr(vault, "_try_extract_umls_unlocked", fake_extract)
    monkeypatch.setattr(vault, "_UMLS_FAILED", {})
    from rag_lib.umls import extractor
    monkeypatch.setattr(extractor, "extract_umls_concepts", lambda *a, **k: [])   # skip the warmup
    from rag_lib.api.app import create_app
    app = create_app()
    try:
        before = _request(app, "GET", "/api/profiles/chromatin/umls").json()
        assert (before["pending"], before["running"]) == (1, False)
        after = _request(app, "POST", "/api/profiles/chromatin/umls").json()
        assert after["running"] is False
        assert after["pending"] == 0            # the extraction ran this time
        assert calls["n"] == 2
    finally:
        scheduler_jobs.SEED_UMLS_RUNNING.discard(pid)


def test_a_seed_umls_flag_nobody_clears_expires(monkeypatch):
    """The route sets the flag before the scheduler runs the job; a job the
    scheduler drops (max instances, misfire) never clears it."""
    flags = scheduler_jobs._RunningFlags()
    flags.add(7)
    assert 7 in flags
    flags._since[7] -= (flags.MAX_MINUTES * 60 + 1)
    assert 7 not in flags
    flags.add(8)
    flags.discard(8)
    assert 8 not in flags


# ---------------------------------------------------------------------------
# 3. A pre-created scan row whose job cannot open the database is closed
# ---------------------------------------------------------------------------


def test_scan_row_is_closed_when_the_job_cannot_open_the_db(env, monkeypatch):
    conn = connect(env)
    apply_migrations(conn)
    uid = int(users_repo.upsert(conn, "demo@example.com")["id"])
    pid = _live_profile(conn, uid, "Live")
    run_id = gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="manual")
    conn.close()

    real_connect = scheduler_jobs.connect
    calls = {"n": 0}

    def flaky_connect(path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("unable to open database file")
        return real_connect(path)

    monkeypatch.setattr(scheduler_jobs, "connect", flaky_connect)
    settings = SimpleNamespace(RADAR_DB_PATH=env, RADAR_DEFAULT_MAILTO="d@example.com",
                               RADAR_GATHER_MAX_CANDIDATES=None)
    with pytest.raises(sqlite3.OperationalError):
        scheduler_jobs.gather_for_profile(uid, pid, settings=settings, run_id=run_id, tier="manual")
    # The lock is released and the row closed with the error: "Scan now" is
    # not a 409 for the next two hours.
    assert not scheduler_jobs._scan_locks[pid].locked()
    conn = connect(env)
    try:
        row = gather_runs_repo.get(conn, run_id)
        assert row["finished_at"] is not None and "unable to open" in row["error"]
        assert gather_runs_repo.running_scan(conn, pid) is None
    finally:
        conn.close()
