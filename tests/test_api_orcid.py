"""``POST /api/profiles/draft/from-orcid`` end-to-end on the inline path.

Injects a ``FakeOpenAlexClient`` through ``get_orcid_client`` so the import
runs synchronously inside the request (no APScheduler, no network) and
the draft is fully seeded when the kickoff response returns.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from rag_lib.api import settings as settings_module
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import papers as papers_repo, profiles as profiles_repo, users as users_repo
from tests.fake_openalex_client import FakeOpenAlexClient
from tests.test_orcid_import import AUTHOR, CORPUS, ORCID

HDR = {"X-User-Email": "demo@example.com"}
HDR_B = {"X-User-Email": "other@example.com"}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    db = tmp_path / "radar.db"
    monkeypatch.setenv("RADAR_DB_PATH", str(db))
    monkeypatch.setenv("RADAR_VAULT_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("RADAR_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("RADAR_RP_PROFILES_DIR", str(tmp_path / "rp"))
    monkeypatch.setenv("RADAR_DEFAULT_EMBEDDING_MODEL", "placeholder-v1")
    monkeypatch.setenv("RADAR_LOG_JSON", "false")
    monkeypatch.setenv("RADAR_SCHEDULER_ENABLED", "false")
    monkeypatch.setenv("RADAR_UMLS_ENABLED", "false")
    settings_module.get_settings.cache_clear()
    conn = connect(db)
    apply_migrations(conn)
    users_repo.upsert(conn, "demo@example.com")
    conn.close()
    yield {"db": db, "tmp": tmp_path}
    settings_module.get_settings.cache_clear()


def _app(fake: FakeOpenAlexClient | None):
    from rag_lib.api.app import create_app
    from rag_lib.api.routers.profiles import get_orcid_client

    app = create_app()
    if fake is not None:
        app.dependency_overrides[get_orcid_client] = lambda: fake
    return app


def _request(app, method: str, path: str, **kwargs) -> httpx.Response:
    return _requests(app, [(method, path, kwargs)])[0]


def _requests(app, calls: list[tuple], between=None) -> list[httpx.Response]:
    """Run several requests inside ONE lifespan.

    Startup closes every unfinished gather_runs row ("server restarted"),
    so a test that stages an in-flight run must not restart the app
    between staging it and the request that should see it. ``between``
    is called after each request with its index (to mutate the DB).
    """
    async def _run() -> list[httpx.Response]:
        out: list[httpx.Response] = []
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                for i, (method, path, kwargs) in enumerate(calls):
                    out.append(await client.request(method, path, **kwargs))
                    if between is not None:
                        between(i, out)
        return out
    return asyncio.run(_run())


def _fake():
    return FakeOpenAlexClient(authors={ORCID: AUTHOR}, orcid_works=CORPUS)


def _select(app, slug: str, ids: list[str] | None = None, headers=HDR):
    """Phase B: confirm ``ids`` (default: the rule's pre-checked works) and return (start, status)."""
    if ids is None:
        works = _request(app, "GET", f"/api/profiles/draft/{slug}/works", headers=headers).json()
        ids = [w["openalex_id"] for w in works if w["default_selected"]]
    r = _request(app, "POST", f"/api/profiles/draft/{slug}/seeds/select", json={"openalex_ids": ids}, headers=headers)
    assert r.status_code == 200, r.text
    start = r.json()
    st = _request(app, "GET", f"/api/profiles/draft/{slug}/import/{start['run_id']}", headers=headers).json()
    return start, st


def test_happy_path_seeds_topics_and_rp_files(env):
    app = _app(_fake())
    r = _request(app, "POST", "/api/profiles/draft/from-orcid", json={"orcid": f"https://orcid.org/{ORCID}"}, headers=HDR)
    assert r.status_code == 200, r.text
    start = r.json()
    assert start["name"] == "Nathan C. Sheffield" and start["slug"] and start["run_id"] >= 1
    slug, run_id = start["slug"], start["run_id"]

    st = _request(app, "GET", f"/api/profiles/draft/{slug}/import/{run_id}", headers=HDR).json()
    assert st["run"]["finished_at"] and st["run"]["error"] is None and st["run"]["tier_used"] == "orcid_import"
    res = st["result"]
    # Phase A: works fetched and offered, nothing seeded yet.
    assert res["phase"] == "fetch"
    assert (res["n_fetched"], res["n_kept"], res["n_works"], res["n_default_seeds"]) == (8, 6, 6, 3)
    assert (res["n_seeds"], res["n_embedded"]) == (0, 0)
    assert res["author"]["institution"] == "University of Virginia"
    assert (env["tmp"] / "rp").exists() and res["rp_profile_dir"].endswith(slug)
    assert _request(app, "GET", f"/api/profiles/draft/{slug}/seeds", headers=HDR).json() == []

    works = _request(app, "GET", f"/api/profiles/draft/{slug}/works", headers=HDR).json()
    by = {w["openalex_id"]: w for w in works}
    assert [w["openalex_id"] for w in works][:3] == ["W1008", "W1002", "W1004"]  # newest first
    assert by["W1008"]["position"] == "middle" and by["W1008"]["is_corresponding"] is True and by["W1008"]["default_selected"]
    assert by["W1003"]["position"] == "middle" and by["W1003"]["default_selected"] is False and by["W1003"]["seed_eligible"]
    assert by["W1004"]["work_type"] == "dataset" and by["W1004"]["seed_eligible"] is False
    assert by["W1005"]["dup_of"] == "W1001" and by["W1005"]["default_selected"] is False and by["W1005"]["has_abstract"] is False
    assert by["W1001"]["has_abstract"] is True and by["W1001"]["first_author"] == "Nathan C. Sheffield"
    assert all(w["selected"] is None and w["is_seed"] is False for w in works)

    # Phase B: confirm the pre-checked works.
    sel, st2 = _select(app, slug)
    assert st2["run"]["tier_used"] == "orcid_seed" and st2["run"]["error"] is None, st2
    res2 = st2["result"]
    assert res2["phase"] == "seed" and (res2["n_seeds"], res2["n_embedded"]) == (3, 3)
    works = _request(app, "GET", f"/api/profiles/draft/{slug}/works", headers=HDR).json()
    assert {w["openalex_id"] for w in works if w["is_seed"]} == {"W1008", "W1002", "W1001"}
    assert {w["openalex_id"] for w in works if w["selected"]} == {"W1008", "W1002", "W1001"}

    seeds = _request(app, "GET", f"/api/profiles/draft/{slug}/seeds", headers=HDR).json()
    assert [s["id"] for s in seeds] == ["W1008", "W1002", "W1001"]
    assert seeds[0]["year"] == 2023 and seeds[0]["pages"] == 0

    topics = _request(app, "GET", f"/api/profiles/draft/{slug}/topics", headers=HDR).json()
    names = {t["name"]: t for t in topics}
    assert "Folate and B Vitamins Research" not in names
    assert all(t["source"] == "orcid" for t in topics)
    # Concepts are aggregated from the selected seeds only: the cancer-classification
    # topic (middle-author consortium paper, not selected) is not listed at all.
    assert {t["name"]: t["on"] for t in topics} == {
        "Genomics and Chromatin Dynamics": True,
        "Single-cell and spatial transcriptomics": True,
    }
    assert names["Genomics and Chromatin Dynamics"]["seed_papers"] == 2
    # a second read is stable (no recompute from seeds)
    assert _request(app, "GET", f"/api/profiles/draft/{slug}/topics", headers=HDR).json() == topics

    conn = connect(env["db"])
    row = profiles_repo.get_by_slug(conn, 1, slug)
    assert row["orcid"] == ORCID and row["researcher_name"] == "Nathan C. Sheffield" and row["is_draft"] == 1
    assert row["embedding_model"] == "placeholder-v1"
    assert conn.execute("SELECT COUNT(*) FROM profile_schedules WHERE profile_id = ?", (row["id"],)).fetchone()[0] == 0
    assert papers_repo.get_by_openalex_id(conn, "W1004")["source"] == "orcid"
    conn.close()

    coh = _request(app, "POST", f"/api/profiles/draft/{slug}/coherence", headers=HDR).json()
    assert coh["n"] == 3
    commit = _request(app, "POST", "/api/profiles", headers=HDR, json={
        "slug": slug, "threshold": 0.8, "selected_topic_ids": [t["id"] for t in topics if t["on"]],
    })
    assert commit.status_code == 200, commit.text
    assert commit.json()["seeds"] == 3


def test_bad_orcid_is_422_and_unknown_orcid_is_404_without_draft(env):
    app = _app(_fake())
    assert _request(app, "POST", "/api/profiles/draft/from-orcid", json={"orcid": "0000-0001-5643-4069"}, headers=HDR).status_code == 422
    app2 = _app(FakeOpenAlexClient(authors={}, orcid_works=[]))
    r = _request(app2, "POST", "/api/profiles/draft/from-orcid", json={"orcid": ORCID}, headers=HDR)
    assert r.status_code == 404
    conn = connect(env["db"])
    assert conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0] == 0
    conn.close()


def test_zero_works_records_error_and_commit_is_409(env):
    app = _app(FakeOpenAlexClient(authors={ORCID: AUTHOR}, orcid_works=[]))
    start = _request(app, "POST", "/api/profiles/draft/from-orcid", json={"orcid": ORCID}, headers=HDR).json()
    st = _request(app, "GET", f"/api/profiles/draft/{start['slug']}/import/{start['run_id']}", headers=HDR).json()
    assert st["run"]["finished_at"] and "no usable works" in st["run"]["error"] and st["result"] is None
    commit = _request(app, "POST", "/api/profiles", headers=HDR,
                      json={"slug": start["slug"], "threshold": 0.8, "selected_topic_ids": []})
    assert commit.status_code == 409


def test_scheduler_disabled_without_fake_is_503_and_creates_nothing(env):
    app = _app(None)
    r = _request(app, "POST", "/api/profiles/draft/from-orcid", json={"orcid": ORCID}, headers=HDR)
    assert r.status_code == 503
    conn = connect(env["db"])
    assert conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0] == 0
    conn.close()


def test_other_user_cannot_poll_or_list(env):
    app = _app(_fake())
    start = _request(app, "POST", "/api/profiles/draft/from-orcid", json={"orcid": ORCID}, headers=HDR).json()
    assert _request(app, "GET", f"/api/profiles/draft/{start['slug']}/import/{start['run_id']}", headers=HDR_B).status_code == 404
    assert _request(app, "GET", f"/api/profiles/draft/{start['slug']}/seeds", headers=HDR_B).status_code == 404


def test_inflight_import_same_orcid_is_409(env):
    app = _app(_fake())
    state: dict = {}

    def between(i, out):
        if i == 0:
            # Re-open the finished run to simulate a still-running import.
            state["start"] = out[0].json()
            conn = connect(env["db"])
            conn.execute("UPDATE gather_runs SET finished_at = NULL WHERE id = ?", (state["start"]["run_id"],))
            conn.commit()
            conn.close()

    resps = _requests(app, [
        ("POST", "/api/profiles/draft/from-orcid", {"json": {"orcid": ORCID}, "headers": HDR}),
        ("POST", "/api/profiles/draft/from-orcid", {"json": {"orcid": ORCID}, "headers": HDR}),
    ], between=between)
    start = state["start"]
    assert resps[0].status_code == 200
    assert resps[1].status_code == 409 and resps[1].json()["detail"]["slug"] == start["slug"]
    # Commit must also refuse while the import is in flight. A fresh lifespan
    # closes the staged run, so re-open it inside this one before committing.
    def reopen(i, out):
        if i == 0:
            conn = connect(env["db"])
            conn.execute("UPDATE gather_runs SET finished_at = NULL WHERE id = ?", (start["run_id"],))
            conn.commit()
            conn.close()

    resps2 = _requests(app, [
        ("GET", "/api/health", {}),
        ("POST", "/api/profiles", {"json": {"slug": start["slug"], "threshold": 0.8,
                                            "selected_topic_ids": []}, "headers": HDR}),
    ], between=reopen)
    assert resps2[1].status_code == 409


def test_restart_closes_orphaned_runs(env):
    conn = connect(env["db"])
    conn.execute("INSERT INTO profiles (user_id, name, slug, embedding_model, topic_filters_json, n_seed, is_draft) "
                 "VALUES (1, 'p', 'p', 'placeholder-v1', '{}', 0, 1)")
    pid = conn.execute("SELECT id FROM profiles WHERE slug='p'").fetchone()[0]
    conn.execute("INSERT INTO gather_runs (profile_id, user_id, tier_used) VALUES (?, 1, 'orcid_import')", (pid,))
    conn.commit()
    conn.close()
    app = _app(_fake())
    _request(app, "GET", "/api/health")  # lifespan runs the cleanup
    conn = connect(env["db"])
    row = conn.execute("SELECT finished_at, error FROM gather_runs").fetchone()
    assert row["finished_at"] and "restarted" in row["error"]
    conn.close()


class _FakeScheduler:
    def __init__(self):
        self.jobs: dict[str, dict] = {}

    def add_job(self, func, trigger=None, **kw):
        self.jobs[kw["id"]] = {"func": func, "trigger": trigger, **kw}

    def remove_job(self, job_id):
        self.jobs.pop(job_id, None)


def test_commit_registers_live_gather_job(env):
    """A profile committed while the server is up must be gathered on its
    cron without waiting for a restart (runner.start only reads the table
    at boot)."""
    app = _app(_fake())
    fake_sched = _FakeScheduler()
    start = _request(app, "POST", "/api/profiles/draft/from-orcid", json={"orcid": ORCID}, headers=HDR).json()
    _select(app, start["slug"])
    topics = _request(app, "GET", f"/api/profiles/draft/{start['slug']}/topics", headers=HDR).json()

    def install_scheduler(i, out):
        # the lifespan sets app.state.scheduler = None (disabled); stand in
        # a fake for the commit request that follows.
        app.state.scheduler = fake_sched

    resps = _requests(app, [
        ("GET", "/api/health", {}),
        ("POST", "/api/profiles", {"json": {"slug": start["slug"], "threshold": 0.8,
                                            "selected_topic_ids": [t["id"] for t in topics]}, "headers": HDR}),
    ], between=install_scheduler)
    assert resps[1].status_code == 200, resps[1].text
    conn = connect(env["db"])
    pid = profiles_repo.get_by_slug(conn, 1, start["slug"])["id"]
    conn.close()
    job = fake_sched.jobs.get(f"gather:{pid}")
    assert job is not None and job["func"].__name__ == "gather_for_profile"
    assert job["args"] == [1, pid] and job["replace_existing"] is True


def test_list_drafts_scoped_to_user_and_reports_seeds_and_import_state(env):
    app = _app(_fake())
    resps = _requests(app, [
        ("POST", "/api/profiles/draft", {"json": {"name": "pdf draft"}, "headers": HDR}),
        ("POST", "/api/profiles/draft/from-orcid", {"json": {"orcid": ORCID}, "headers": HDR}),
        ("GET", "/api/profiles/drafts", {"headers": HDR}),
        ("GET", "/api/profiles/drafts", {"headers": HDR_B}),
    ])
    assert [r.status_code for r in resps] == [200, 200, 200, 200]
    drafts = resps[2].json()
    assert [d["slug"] for d in drafts] == [resps[1].json()["slug"], "pdf-draft"]  # newest first
    orcid_draft, pdf_draft = drafts
    assert orcid_draft["orcid"] == ORCID and orcid_draft["n_seeds"] == 0 and orcid_draft["importing"] is False
    assert orcid_draft["import_run_id"] == resps[1].json()["run_id"]
    assert orcid_draft["n_works"] == 6 and orcid_draft["phase"] == "selecting"
    assert pdf_draft["orcid"] is None and pdf_draft["n_seeds"] == 0 and pdf_draft["import_run_id"] is None
    assert pdf_draft["phase"] is None and pdf_draft["n_works"] == 0
    assert resps[3].json() == []  # another user sees nothing

    _select(app, orcid_draft["slug"])
    seeded = _request(app, "GET", "/api/profiles/drafts", headers=HDR).json()[0]
    assert seeded["n_seeds"] == 3 and seeded["phase"] == "seeded"

    # A running import shows as importing=True; a committed profile drops out.
    conn = connect(env["db"])
    conn.execute("UPDATE gather_runs SET finished_at = NULL WHERE tier_used = 'orcid_import'")
    conn.execute("UPDATE profiles SET is_draft = 0 WHERE slug = 'pdf-draft'")
    conn.commit(); conn.close()
    drafts = _request(app, "GET", "/api/profiles/drafts", headers=HDR).json()
    # (the fresh lifespan closes the orphaned run again, so importing is False here;
    #  the flag itself is covered by the repo query below)
    assert [d["slug"] for d in drafts] == [orcid_draft["slug"]]

    conn = connect(env["db"])
    conn.execute("UPDATE gather_runs SET finished_at = NULL WHERE tier_used = 'orcid_import'")
    conn.commit()
    row = profiles_repo.list_drafts_for_user(conn, 1)[0]
    assert bool(row["importing"]) is True
    conn.close()


# --------------------------------------------------------------------------- seed picker


def test_select_validates_ids_and_rejects_datasets(env):
    app = _app(_fake())
    start = _request(app, "POST", "/api/profiles/draft/from-orcid", json={"orcid": ORCID}, headers=HDR).json()
    slug = start["slug"]
    for body, needle in (
        ({"openalex_ids": []}, "at least one"),
        ({"openalex_ids": ["W9999"]}, "not fetched works"),
        ({"openalex_ids": ["W1004"]}, "cannot be seeds"),
    ):
        r = _request(app, "POST", f"/api/profiles/draft/{slug}/seeds/select", json=body, headers=HDR)
        assert r.status_code == 400 and needle in r.json()["detail"], r.text
    # other users cannot see or select
    assert _request(app, "GET", f"/api/profiles/draft/{slug}/works", headers=HDR_B).status_code == 404
    assert _request(app, "POST", f"/api/profiles/draft/{slug}/seeds/select", json={"openalex_ids": ["W1001"]}, headers=HDR_B).status_code == 404


def test_user_choice_overrides_the_rule_and_can_be_changed(env):
    """A middle-author paper in, a lead-author paper out; then a second choice replaces the first."""
    app = _app(_fake())
    start = _request(app, "POST", "/api/profiles/draft/from-orcid", json={"orcid": ORCID}, headers=HDR).json()
    slug = start["slug"]

    _sel, st = _select(app, slug, ["W1003", "W1008", "W1005"])  # middle-author, corresponding, the preprint copy
    res = st["result"]
    assert st["run"]["error"] is None and res["n_seeds"] == 3
    assert any("middle-author" in w for w in res["warnings"])
    seeds = _request(app, "GET", f"/api/profiles/draft/{slug}/seeds", headers=HDR).json()
    assert {x["id"] for x in seeds} == {"W1003", "W1008", "W1005"}
    topics = {t["name"]: t for t in _request(app, "GET", f"/api/profiles/draft/{slug}/topics", headers=HDR).json()}
    assert "Gene expression and cancer classification" in topics and topics["Gene expression and cancer classification"]["on"]

    _sel2, st2 = _select(app, slug, ["W1001"])
    assert st2["run"]["error"] is None and st2["result"]["n_seeds"] == 1 and st2["result"]["n_embedded"] == 1
    seeds = _request(app, "GET", f"/api/profiles/draft/{slug}/seeds", headers=HDR).json()
    assert [x["id"] for x in seeds] == ["W1001"]
    works = {w["openalex_id"]: w for w in _request(app, "GET", f"/api/profiles/draft/{slug}/works", headers=HDR).json()}
    assert works["W1001"]["selected"] is True and works["W1003"]["selected"] is False and works["W1003"]["is_seed"] is False
    conn = connect(env["db"])
    assert conn.execute("SELECT n_seed FROM profiles WHERE slug = ?", (slug,)).fetchone()[0] == 1
    conn.close()


def test_commit_before_selecting_is_409_and_select_during_fetch_is_409(env):
    app = _app(_fake())
    start = _request(app, "POST", "/api/profiles/draft/from-orcid", json={"orcid": ORCID}, headers=HDR).json()
    slug = start["slug"]
    commit = _request(app, "POST", "/api/profiles", headers=HDR,
                      json={"slug": slug, "threshold": 0.8, "selected_topic_ids": []})
    assert commit.status_code == 409

    def reopen(i, out):
        if i == 0:
            conn = connect(env["db"])
            conn.execute("UPDATE gather_runs SET finished_at = NULL WHERE id = ?", (start["run_id"],))
            conn.commit(); conn.close()

    resps = _requests(app, [
        ("GET", "/api/health", {}),
        ("POST", f"/api/profiles/draft/{slug}/seeds/select", {"json": {"openalex_ids": ["W1001"]}, "headers": HDR}),
    ], between=reopen)
    assert resps[1].status_code == 409


def test_seed_selection_marks_rp_sidecar(env):
    app = _app(_fake())
    start = _request(app, "POST", "/api/profiles/draft/from-orcid", json={"orcid": ORCID}, headers=HDR).json()
    slug = start["slug"]
    import json as _json
    sidecar = env["tmp"] / "rp" / "1" / slug / "meta" / "openalex_topics.json"
    assert sidecar.exists() and not any(r["is_seed"] for r in _json.loads(sidecar.read_text()))
    _select(app, slug, ["W1001", "W1003"])
    flags = {r["openalex_id"]: r["is_seed"] for r in _json.loads(sidecar.read_text())}
    assert flags["W1001"] and flags["W1003"] and not flags["W1002"]
    report = _json.loads((env["tmp"] / "rp" / "1" / ".build" / slug / "meta" / "build_report.json").read_text())
    assert report["n_seeds"] == 2
