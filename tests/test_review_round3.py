"""Fixes from the 2026-10-15 functional test round (report items 3, 8-29)."""

from __future__ import annotations

import asyncio
import json
import shutil

import httpx
import pytest

from rag_lib.api import settings as settings_module
from rag_lib.db import apply_migrations, connect
from rag_lib.db.migrate import MIGRATIONS_DIR
from rag_lib.db.repos import (
    candidates as candidates_repo,
    embeddings as embeddings_repo,
    gather_runs as gather_runs_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    users as users_repo,
)
from rag_lib.paper import Paper


def _db(tmp_path):
    conn = connect(tmp_path / "r.db")
    apply_migrations(conn)
    uid = int(users_repo.upsert(conn, "a@example.com")["id"])
    return conn, uid


def _interest(conn, uid, name, threshold=0.5):
    return profiles_repo.upsert(conn, user_id=uid, name=name, embedding_model="placeholder-v1",
                                n_seed=0, topic_filters={"topics": []}, threshold=threshold)


# --- Prosopia: the paged API ------------------------------------------------


def _prosopia_transport(pages: list[list[dict]], *, detail: dict, summaries: dict[str, str]):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        path = request.url.path
        if path.endswith("/papers"):
            i = int(request.url.params.get("cursor") or 0)
            more = i + 1 < len(pages)
            return httpx.Response(200, json={
                "items": pages[i], "total": sum(map(len, pages)),
                "next_cursor": str(i + 1) if more else None, "has_more": more,
            })
        if path.endswith("/summaries"):
            pid = request.url.params["ids"]
            return httpx.Response(200, json={"summaries": {pid: summaries[pid]}, "unavailable": {}})
        if path.endswith("/profiles/jane"):
            return httpx.Response(200, json=detail)
        return httpx.Response(404, json={"detail": "nope"})

    return httpx.MockTransport(handler), calls


def test_prosopia_reads_every_page_and_the_new_summary_shape():
    from rag_lib.prosopia_client import ApiProfile, PagedApiStorage, read_summary

    pages = [[{"paper_id": f"p{i}", "title": f"T{i}", "doi": f"10.1/{i}",
               "summary": {"available": i == 3}} for i in range(j * 2, j * 2 + 2)]
             for j in range(3)]
    transport, calls = _prosopia_transport(
        pages, detail={"slug": "jane", "rid": "0000-0001-2345-6789",
                       "fields": {"name": "Jane Doe", "affiliation": "UVA"},
                       "soul": "S", "expertise": "E"},
        summaries={"p3": "the summary of p3"},
    )
    client = httpx.Client(base_url="http://prosopia.test", transport=transport)
    prof = ApiProfile(PagedApiStorage(slug="jane", base_url="http://prosopia.test", client=client))
    assert [p.paper_id for p in prof.papers] == [f"p{i}" for i in range(6)]
    assert sum("/papers" in c for c in calls) == 3          # walked the cursor
    assert prof.name == "Jane Doe" and prof.rid == "0000-0001-2345-6789"
    assert prof.expertise == "E" and prof.soul == "S" and prof.affiliation == "UVA"
    assert list(prof.summaries) == ["p3"]
    assert read_summary(prof, "p3") == "the summary of p3"


def test_prosopia_still_reads_the_old_unpaged_list():
    from rag_lib.prosopia_client import ApiProfile, PagedApiStorage

    def handler(request):
        if request.url.path.endswith("/papers"):
            return httpx.Response(200, json=[{"paper_id": "a", "title": "A", "summary_available": True}])
        return httpx.Response(200, json={"metadata": {"name": "Old Server"}})

    client = httpx.Client(base_url="http://old.test", transport=httpx.MockTransport(handler))
    prof = ApiProfile(PagedApiStorage(slug="x", base_url="http://old.test", client=client))
    assert prof.name == "Old Server" and [p.paper_id for p in prof.papers] == ["a"]
    assert list(prof.summaries) == ["a"]


def test_prosopia_base_url_must_be_an_allowed_instance():
    from rag_lib.prosopia_client import DEFAULT_BASE_URL, allowed_base_url

    assert allowed_base_url(None, []) == DEFAULT_BASE_URL
    assert allowed_base_url("https://prosopia.example.org/", ["https://prosopia.example.org"]) \
        == "https://prosopia.example.org"
    for bad in ("http://127.0.0.1:8000", "http://169.254.169.254/latest",
                DEFAULT_BASE_URL + ".evil.example", "file:///etc/passwd"):
        with pytest.raises(ValueError):
            allowed_base_url(bad, [])


# --- feed --------------------------------------------------------------------


def test_fair_shares_fill_the_feed_from_whoever_has_cards():
    from rag_lib.api.services.radar import fair_shares

    assert fair_shares([300, 2, 0, 50], 200) == [148, 2, 0, 50]
    assert sum(fair_shares([30] * 15, 200)) == 200
    assert fair_shares([5, 5], 200) == [5, 5]
    assert fair_shares([], 200) == []


def test_a_seed_is_not_in_its_own_feed_by_id_or_doi(tmp_path):
    conn, uid = _db(tmp_path)
    pid = _interest(conn, uid, "Alpha")
    for oa, doi in (("W1", "10.1/one"), ("W2", "10.1/two"), ("W3", None), ("local:x", "10.1/TWO")):
        papers_repo.upsert(conn, {"openalex_id": oa, "title": oa, "doi": doi, "source": "s"})
    for oa, s in (("W1", 0.99), ("W2", 0.95), ("W3", 0.9)):
        candidates_repo.insert_dedup(conn, profile_id=pid, openalex_id=oa, score=s, score_raw=s)
    profiles_repo.attach_seed(conn, pid, "W1")         # a feed paper made a seed
    profiles_repo.attach_seed(conn, pid, "local:x")    # an upload of W2 (same DOI)
    feed = [r["openalex_id"] for r in candidates_repo.top_for_profile(conn, pid)]
    assert feed == ["W3"]
    assert candidates_repo.scores_for_profile(conn, pid) == [0.9]


def test_without_seeds_drops_scanned_copies_of_the_seeds(tmp_path):
    conn, uid = _db(tmp_path)
    pid = _interest(conn, uid, "Alpha")
    papers_repo.upsert(conn, {"openalex_id": "W1", "title": "s", "doi": "10.1/S", "source": "s"})
    profiles_repo.attach_seed(conn, pid, "W1")
    got = profiles_repo.without_seeds(
        [Paper(doi=None, openalex_id="W1", title="a"), Paper(doi="10.1/s", openalex_id="W9", title="b"),
         Paper(doi="10.1/other", openalex_id="W8", title="c")],
        profiles_repo.seed_keys(conn, pid))
    assert [p.openalex_id for p in got] == ["W8"]


def test_bucket_bounds_are_applied_before_the_limit(tmp_path):
    conn, uid = _db(tmp_path)
    pid = _interest(conn, uid, "Alpha", threshold=0.5)
    for i in range(10):
        oa = f"W{i}"
        papers_repo.upsert(conn, {"openalex_id": oa, "title": oa, "source": "s"})
        candidates_repo.insert_dedup(conn, profile_id=pid, openalex_id=oa, score=0.9 - i * 0.03,
                                     score_raw=0.9 - i * 0.03)
    # The two weakest are under a 0.67 seed band; a limit of 2 still finds them.
    rows = candidates_repo.top_for_profile(conn, pid, limit=2, sim_below=0.67)
    assert sorted(r["openalex_id"] for r in rows) == ["W8", "W9"]
    rows = candidates_repo.top_for_profile(conn, pid, limit=50, sim_at_least=0.85)
    assert [r["openalex_id"] for r in rows] == ["W0", "W1"]


def test_save_sets_a_state_and_reports_one_change(tmp_path):
    conn, uid = _db(tmp_path)
    pid = _interest(conn, uid, "Alpha")
    papers_repo.upsert(conn, {"openalex_id": "W1", "title": "x", "source": "s"})
    candidates_repo.insert_dedup(conn, profile_id=pid, openalex_id="W1", score=0.9)
    assert candidates_repo.set_state(conn, pid, "W1", "saved_at", True) == ("saved", True)
    assert candidates_repo.set_state(conn, pid, "W1", "saved_at", True) == ("saved", False)  # double click
    assert candidates_repo.set_state(conn, pid, "W1", "dismissed_at", True) == ("dismissed", True)
    assert conn.execute("SELECT saved_at FROM profile_candidates").fetchone()[0] is None
    assert candidates_repo.set_state(conn, pid, "W1", "dismissed_at", False) == (None, True)
    assert candidates_repo.set_state(conn, pid, "W404", "saved_at", True) is None


def test_feed_header_is_the_latest_scan_with_its_date(tmp_path):
    from rag_lib.api.services.radar import _last_run_metrics

    conn, uid = _db(tmp_path)
    a, b = _interest(conn, uid, "Alpha"), _interest(conn, uid, "Beta")
    for pid, n, tier in ((a, 500, "scheduled"), (a, 7, "scheduled"), (b, 3, "manual"), (b, 999, "dry_run")):
        rid = gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used=tier)
        gather_runs_repo.finish(conn, rid, n_fetched=n, n_new=0, n_redup=0)
    fetched_at, scored = _last_run_metrics(conn, [a, b])
    assert scored == 7 + 3
    assert len(fetched_at) == 20 and fetched_at[10] == "T" and fetched_at.endswith("Z")


# --- scans -------------------------------------------------------------------


def test_a_second_scan_of_an_interest_is_skipped_while_one_runs(tmp_path, monkeypatch):
    from rag_lib.scheduler import jobs

    conn, uid = _db(tmp_path)
    pid = _interest(conn, uid, "Alpha")
    monkeypatch.setenv("RADAR_DB_PATH", str(tmp_path / "r.db"))
    settings_module.get_settings.cache_clear()
    lock = jobs._scan_locks.setdefault(pid, jobs.threading.Lock())
    lock.acquire()
    try:
        run_id = jobs.gather_for_profile(uid, pid, settings=settings_module.get_settings())
    finally:
        lock.release()
        settings_module.get_settings.cache_clear()
    row = conn.execute("SELECT finished_at, error FROM gather_runs WHERE id = ?", (run_id,)).fetchone()
    assert row[0] is not None and row[1] == jobs.SCAN_BUSY


def test_running_scan_ignores_trials_imports_and_dead_runs(tmp_path):
    conn, uid = _db(tmp_path)
    pid = _interest(conn, uid, "Alpha")
    gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="dry_run")
    gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="orcid_import")
    dead = gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="manual")
    conn.execute("UPDATE gather_runs SET started_at = datetime('now', '-1 day') WHERE id = ?", (dead,))
    conn.commit()
    assert gather_runs_repo.running_scan(conn, pid) is None
    live = gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="manual")
    assert int(gather_runs_repo.running_scan(conn, pid)["id"]) == live


def test_resurfaced_papers_are_scored_with_their_stored_vector(tmp_path):
    conn, uid = _db(tmp_path)
    papers_repo.upsert(conn, {"openalex_id": "W1", "title": "x", "source": "s"})
    embeddings_repo.upsert(conn, "W1", "m", [1.0, 0.0])
    known, fresh = Paper(doi=None, openalex_id="W1", title="x"), Paper(doi=None, openalex_id="W2", title="y")
    assert embeddings_repo.attach_stored(conn, [known, fresh], "m") == 1
    assert known.embeddings["m"] == [1.0, 0.0] and "m" not in fresh.embeddings


# --- names and slugs -----------------------------------------------------------


def test_names_and_slugs_are_unique_per_user_only(tmp_path):
    from rag_lib.api.services.wizard import create_draft

    conn, uid = _db(tmp_path)
    other = int(users_repo.upsert(conn, "b@example.com")["id"])
    mine = create_draft(conn, user_id=uid, name="Protein folding", embedding_model="placeholder-v1")
    theirs = create_draft(conn, user_id=other, name="Protein folding", embedding_model="placeholder-v1")
    assert mine == theirs == {"slug": "protein-folding", "name": "Protein folding"}
    again = create_draft(conn, user_id=uid, name="Protein folding", embedding_model="placeholder-v1")
    assert again == {"slug": "protein-folding-2", "name": "Protein folding (2)"}
    with pytest.raises(ValueError):
        create_draft(conn, user_id=uid, name="   ", embedding_model="placeholder-v1")


def test_slugs_keep_letters_of_any_script():
    assert profiles_repo.slugify("蛋白质 折叠") == "蛋白质-折叠"
    assert profiles_repo.slugify("タンパク質") == "タンパク質"
    assert profiles_repo.slugify("🧬🧪") == "interest"


def test_rebuilding_profiles_keeps_every_child_row(tmp_path):
    """0021 drops and recreates ``profiles``; with foreign keys on, that
    would have cascaded and emptied the seeds, candidates and runs."""
    old = tmp_path / "old_migrations"
    old.mkdir()
    for f in MIGRATIONS_DIR.glob("*.*"):
        if f.stem < "0021" and f.suffix == ".sql":
            shutil.copy(f, old / f.name)
    conn = connect(tmp_path / "m.db")
    apply_migrations(conn, old)
    uid = int(users_repo.upsert(conn, "a@example.com")["id"])
    pid = _interest(conn, uid, "Alpha")
    papers_repo.upsert(conn, {"openalex_id": "W1", "title": "x", "source": "s"})
    profiles_repo.attach_seed(conn, pid, "W1")
    candidates_repo.insert_dedup(conn, profile_id=pid, openalex_id="W1", score=0.9)
    gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="manual")
    conn.execute("DELETE FROM profiles WHERE id = ?", (_interest(conn, uid, "Gone"),))
    conn.commit()
    assert apply_migrations(conn) == ["0021_profile_names_per_user"]
    for table in ("profile_seeds", "profile_candidates", "gather_runs"):
        assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 1, table
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    # A deleted interest's id is not handed out again.
    assert _interest(conn, uid, "Beta") > pid + 1


def test_imports_no_longer_count_as_uploads(tmp_path):
    old = tmp_path / "old_migrations"
    old.mkdir()
    for f in MIGRATIONS_DIR.glob("*.*"):
        if f.stem < "0020" and f.suffix == ".sql":
            shutil.copy(f, old / f.name)
    conn = connect(tmp_path / "m.db")
    apply_migrations(conn, old)
    uid = int(users_repo.upsert(conn, "a@example.com")["id"])
    papers_repo.upsert(conn, {"openalex_id": "W1", "title": "upload", "source": "user_pdf",
                              "uploaded_by_user_id": uid, "file_hash": "h", "local_path": "/v/h.pdf"})
    papers_repo.upsert(conn, {"openalex_id": "W2", "title": "import", "source": "prosopia",
                              "uploaded_by_user_id": uid})
    apply_migrations(conn)
    assert [tuple(r) for r in conn.execute("SELECT user_id, openalex_id, file_hash FROM paper_uploads")] \
        == [(uid, "W1", "h")]
    assert conn.execute("SELECT uploaded_by_user_id FROM papers WHERE openalex_id='W2'").fetchone()[0] is None


# --- errors ------------------------------------------------------------------


def test_public_error_drops_query_strings_and_emails():
    from rag_lib.redact import public_error

    msg = ("HTTPError: 401 for url: https://api.openalex.org/works?filter=x&mailto=me@uni.edu"
           "&cursor=abc (contact me@uni.edu)")
    out = public_error(msg)
    assert "mailto" not in out and "me@uni.edu" not in out
    assert "https://api.openalex.org/works" in out and "401" in out


def test_a_failed_runs_stored_error_is_redacted(tmp_path):
    conn, uid = _db(tmp_path)
    pid = _interest(conn, uid, "Alpha")
    rid = gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="manual")
    gather_runs_repo.finish(conn, rid, n_fetched=0, n_new=0, n_redup=0,
                            error="HTTPError: https://api.openalex.org/works?mailto=x@y.org")
    assert "x@y.org" not in conn.execute("SELECT error FROM gather_runs").fetchone()[0]


def test_drafts_list_reports_a_dead_or_failed_import(tmp_path):
    conn, uid = _db(tmp_path)
    from rag_lib.api.services.wizard import create_draft

    dead = create_draft(conn, user_id=uid, name="Dead", embedding_model="placeholder-v1")["slug"]
    failed = create_draft(conn, user_id=uid, name="Failed", embedding_model="placeholder-v1")["slug"]
    d_id = profiles_repo.get_by_slug(conn, uid, dead)["id"]
    f_id = profiles_repo.get_by_slug(conn, uid, failed)["id"]
    r = gather_runs_repo.start(conn, profile_id=d_id, user_id=uid, tier_used="orcid_import")
    conn.execute("UPDATE gather_runs SET started_at = datetime('now', '-2 hours') WHERE id = ?", (r,))
    conn.commit()
    r2 = gather_runs_repo.start(conn, profile_id=f_id, user_id=uid, tier_used="prosopia_import")
    gather_runs_repo.finish(conn, r2, n_fetched=0, n_new=0, n_redup=0, error="ProsopiaError: boom")
    rows = {r["slug"]: r for r in profiles_repo.list_drafts_for_user(conn, uid)}
    assert rows[dead]["phase"] == "failed" and rows[dead]["import_run_id"] is None
    assert "stopped" in rows[dead]["import_error"]
    assert rows[failed]["phase"] == "failed" and rows[failed]["import_error"] == "ProsopiaError: boom"


# --- API edges ------------------------------------------------------------------


@pytest.fixture()
def app(tmp_path, monkeypatch):
    db = tmp_path / "radar.db"
    for k, v in {"RADAR_DB_PATH": str(db), "RADAR_VAULT_DIR": str(tmp_path / "vault"),
                 "RADAR_CHROMA_DIR": str(tmp_path / "chroma"), "RADAR_LOG_JSON": "false",
                 "RADAR_SCHEDULER_ENABLED": "false", "RADAR_UMLS_ENABLED": "false",
                 "RADAR_DEFAULT_EMBEDDING_MODEL": "placeholder-v1"}.items():
        monkeypatch.setenv(k, v)
    settings_module.get_settings.cache_clear()
    conn = connect(db)
    apply_migrations(conn)
    uid = int(users_repo.upsert(conn, "demo@example.com")["id"])
    pid = _interest(conn, uid, "Chromatin")
    papers_repo.upsert(conn, {"openalex_id": "https://openalex.org/W1", "title": "x", "source": "s"})
    candidates_repo.insert_dedup(conn, profile_id=pid, openalex_id="https://openalex.org/W1",
                                 score=0.9, score_raw=0.9)
    conn.close()
    from rag_lib.umls import extractor
    monkeypatch.setattr(extractor, "extract_umls_concepts", lambda *a, **k: [])
    from rag_lib.api.app import create_app
    from rag_lib.api.deps import get_scheduler

    class _Sched:
        def add_job(self, *a, **k): pass
        def remove_job(self, *a, **k): pass
        def get_job(self, *a, **k): return None

    app = create_app()
    app.dependency_overrides[get_scheduler] = lambda: _Sched()
    yield app
    settings_module.get_settings.cache_clear()


def _requests(app, calls):
    async def _run():
        out = []
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
                for method, path, kw in calls:
                    out.append(await c.request(method, path, **kw))
        return out
    return asyncio.run(_run())


def test_bad_inputs_are_4xx_not_500(app):
    big = 2 ** 63
    nan_body = '{"threshold": NaN}'
    res = _requests(app, [
        ("PATCH", "/api/profiles/chromatin", {"content": nan_body, "headers": {"content-type": "application/json"}}),
        ("PATCH", "/api/profiles/chromatin", {"json": {"threshold": -5}}),
        ("GET", f"/api/researchers/{big}", {}),
        ("GET", "/api/profiles/chromatin/runs", {"params": {"limit": -1}}),
        ("POST", "/api/profiles/draft", {"json": {"name": "x" * 100_000}}),
        ("PATCH", "/api/profiles/chromatin/schedule", {"json": {"cron": "  "}}),
        ("PATCH", "/api/users/me", {"json": {"mailto": "a" * 300 + "@x.org"}}),
        ("POST", "/api/profiles/draft/dry-run-x", {}),
    ])
    codes = [r.status_code for r in res[:7]]
    assert codes == [422, 422, 404, 422, 422, 400, 422], [(r.status_code, r.text[:120]) for r in res]
    assert "finite" in res[0].text


def test_save_is_idempotent_and_feedback_is_logged_once(app):
    card = {"card_id": "https://openalex.org/W1", "profile": "chromatin", "on": True}
    res = _requests(app, [
        ("POST", "/api/radar/cards/save", {"json": card}),
        ("POST", "/api/radar/cards/save", {"json": card}),
        ("GET", "/api/profiles/chromatin/feedback", {}),
        ("POST", "/api/radar/cards/save", {"json": {**card, "on": False}}),
    ])
    assert [r.json()["state"] for r in (res[0], res[1])] == ["saved", "saved"]
    assert [e["action"] for e in res[2].json()] == ["saved"]
    assert res[3].json()["state"] is None


def test_feedback_threshold_endpoint(app):
    (res,) = _requests(app, [("GET", "/api/profiles/chromatin/feedback-threshold", {})])
    body = res.json()
    assert res.status_code == 200
    assert body["n_saved"] == 0 and body["suggested"] is None and body["min_events"] == 30


def test_scan_now_is_refused_while_a_scan_runs(app, tmp_path):
    conn = connect(tmp_path / "radar.db")
    pid = conn.execute("SELECT id FROM profiles").fetchone()[0]
    gather_runs_repo.start(conn, profile_id=pid, user_id=1, tier_used="manual")
    conn.close()
    (res,) = _requests(app, [("POST", "/api/profiles/chromatin/gather-now", {})])
    assert res.status_code == 409, res.text
    assert json.loads(res.text)["detail"].startswith("A scan of this interest is already running")


def test_the_feed_filters_a_bucket_before_cutting_each_interests_share(tmp_path):
    from rag_lib.api.services import radar

    conn, uid = _db(tmp_path)
    busy = _interest(conn, uid, "Busy", threshold=0.5)
    quiet = _interest(conn, uid, "Quiet", threshold=0.5)
    for pid, n in ((busy, 40), (quiet, 2)):
        for i in range(n):
            oa = f"W{pid}-{i}"
            papers_repo.upsert(conn, {"openalex_id": oa, "title": oa, "source": "s"})
            candidates_repo.insert_dedup(conn, profile_id=pid, openalex_id=oa, score=0.9, score_raw=0.9)
    old = radar._DAILY_LIMIT
    radar._DAILY_LIMIT = 20
    try:
        cards = radar.daily(conn, uid).cards
    finally:
        radar._DAILY_LIMIT = old
    # 2 from the quiet interest; its unused share goes to the busy one.
    assert len(cards) == 20
    assert sum(c.profile == "quiet" for c in cards) == 2
