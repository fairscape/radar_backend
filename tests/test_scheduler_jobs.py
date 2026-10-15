"""Phase 7 — scheduler.

The job body (``gather_for_profile``) must:
  - resolve the acting user's mailto from ``users.mailto`` or
    ``Settings.RADAR_DEFAULT_MAILTO``;
  - reuse a pre-opened ``gather_runs`` row when ``run_id`` is given
    (the ``gather-now`` endpoint depends on this so it can echo the id
    back synchronously);
  - close the audit row on both success and failure;
  - dedup-insert candidates via the canonical persistence path.

The runner (``build_scheduler`` + ``start``) must:
  - register one job per row returned by ``schedules_repo.list_enabled``;
  - tolerate broken cron strings without aborting boot;
  - drop a cleanly-shutdown lock on ``stop``.
"""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import (
    gather_runs as gather_runs_repo,
    profiles as profiles_repo,
    schedules as schedules_repo,
)
from rag_lib.paper import Paper
from rag_lib.persistence import store_profile_from_object
from rag_lib.profile import Profile
from rag_lib.scheduler import jobs as scheduler_jobs


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _seed_paper(oa_id: str = "W1") -> Paper:
    p = Paper(
        doi=f"10.1/{oa_id.lower()}",
        openalex_id=oa_id,
        title=f"Seed {oa_id}",
        abstract="abstract",
        year=2024,
        source="openalex_gatherer",
    )
    rng = np.random.RandomState(int(oa_id[1:]) or 1)
    p.embeddings["specter2"] = rng.rand(8).astype(float).tolist()
    return p


def _candidate_paper(oa_id: str) -> Paper:
    p = Paper(
        doi=f"10.9/{oa_id.lower()}",
        openalex_id=oa_id,
        title=f"Candidate {oa_id}",
        abstract="cand abstract",
        year=2026,
        source="openalex_gatherer",
    )
    # Pre-embed in the same 8-dim fixture space as seed papers so the
    # selector doesn't fall through to the real (phase1b-extras) SPECTER2
    # embedder, which would produce 768-dim vectors and break cosine.
    rng = np.random.RandomState(int(oa_id[1:]) or 1)
    p.embeddings["specter2"] = rng.rand(8).astype(float).tolist()
    return p


@pytest.fixture()
def db_path(tmp_path):
    p = tmp_path / "radar.db"
    conn = connect(p)
    try:
        apply_migrations(conn)
    finally:
        conn.close()
    return p


@pytest.fixture()
def settings(db_path):
    return SimpleNamespace(
        RADAR_DB_PATH=db_path,
        RADAR_DEFAULT_MAILTO="default@example.com",
    )


@pytest.fixture()
def profile_id(db_path):
    """Persist a tiny profile (one seed) and return its id."""
    conn = connect(db_path)
    try:
        pid = store_profile_from_object(
            conn,
            user_id=1,
            profile=Profile(
                name="neonatal_vitals",
                papers=[_seed_paper("W1")],
                embedding_model="specter2",
                topic_filters={
                    "topics": [{"id": "T1", "display_name": "Topic", "count": 1}]
                },
                selector_config={
                    "type": "centroid",
                    "embedding_model": "specter2",
                    "centroid": [0.1] * 8,
                    "diagnostics_snapshot": {"status": "fit"},
                },
            ),
        )
    finally:
        conn.close()
    return pid


# --------------------------------------------------------------------------
# Schedules repo + migration smoke
# --------------------------------------------------------------------------


def test_migration_backfills_schedule_for_existing_profile(db_path, profile_id):
    conn = connect(db_path)
    try:
        # Migration backfilled when 0004 ran *before* the profile existed,
        # so we re-run upsert here to verify the repo path also covers
        # post-hoc profiles created after the initial migration.
        row = schedules_repo.upsert(conn, profile_id=profile_id)
        assert row["cron"] == "0 4 * * *"
        assert row["tz"] == "UTC"
        assert row["enabled"] == 1
    finally:
        conn.close()


def test_schedules_upsert_updates_in_place(db_path, profile_id):
    conn = connect(db_path)
    try:
        schedules_repo.upsert(conn, profile_id=profile_id)
        row = schedules_repo.upsert(
            conn, profile_id=profile_id, cron="*/5 * * * *", tz="America/New_York"
        )
        assert row["cron"] == "*/5 * * * *"
        assert row["tz"] == "America/New_York"
    finally:
        conn.close()


def test_list_enabled_excludes_disabled(db_path, profile_id):
    conn = connect(db_path)
    try:
        schedules_repo.upsert(conn, profile_id=profile_id, enabled=False)
        rows = schedules_repo.list_enabled(conn)
        assert all(r["profile_id"] != profile_id for r in rows)
    finally:
        conn.close()


# --------------------------------------------------------------------------
# gather_for_profile job body
# --------------------------------------------------------------------------


class _StubGatherer:
    """Stand-in for OpenAlexGatherer — never touches the network."""

    def __init__(self, papers: list[Paper], api_calls: int = 3):
        self._papers = papers
        self._api_calls = api_calls

    def fetch(self, profile, since, *, limit=None):
        return list(self._papers)

    def cost(self):
        return {"wall_seconds": 0.01, "api_calls": self._api_calls}


def _patch_dependencies(monkeypatch, candidates: list[Paper]):
    """Replace the network gatherer + skip selector refit cost."""
    monkeypatch.setattr(
        scheduler_jobs, "OpenAlexGatherer",
        lambda **kw: _StubGatherer(candidates),
    )


def test_gather_for_profile_writes_gather_runs_row(
    monkeypatch, db_path, settings, profile_id
):
    candidates = [_candidate_paper("W100"), _candidate_paper("W101")]
    _patch_dependencies(monkeypatch, candidates)

    run_id = scheduler_jobs.gather_for_profile(
        user_id=1, profile_id=profile_id, settings=settings, days=7,
    )

    conn = connect(db_path)
    try:
        row = gather_runs_repo.get(conn, run_id)
        assert row is not None
        assert row["profile_id"] == profile_id
        assert row["finished_at"] is not None
        assert row["error"] is None
        assert row["n_fetched"] == 2
        assert row["n_new"] == 2
        assert row["n_redup"] == 0
        assert row["api_calls"] == 3
        assert row["tier_used"] == "scheduled"
    finally:
        conn.close()


def test_gather_for_profile_reuses_supplied_run_id(
    monkeypatch, db_path, settings, profile_id
):
    """gather-now opens the audit row first and passes its id in."""
    candidates = [_candidate_paper("W200")]
    _patch_dependencies(monkeypatch, candidates)

    conn = connect(db_path)
    try:
        run_id = gather_runs_repo.start(
            conn, profile_id=profile_id, user_id=1, tier_used="manual",
        )
    finally:
        conn.close()

    returned = scheduler_jobs.gather_for_profile(
        user_id=1, profile_id=profile_id, settings=settings,
        run_id=run_id, tier="manual",
    )
    assert returned == run_id

    conn = connect(db_path)
    try:
        rows = gather_runs_repo.recent_for_profile(conn, profile_id)
        # Exactly one row — the eager one was reused, not duplicated.
        assert len(rows) == 1
        assert rows[0]["id"] == run_id
        assert rows[0]["finished_at"] is not None
        assert rows[0]["n_fetched"] == 1
    finally:
        conn.close()


def test_gather_for_profile_records_error_on_exception(
    monkeypatch, db_path, settings, profile_id
):
    class _BoomGatherer:
        def __init__(self, **_):
            pass

        def fetch(self, *a, **kw):
            raise RuntimeError("HTTP 429 from upstream")

        def cost(self):
            return {"api_calls": 0}

    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer", _BoomGatherer)

    run_id = scheduler_jobs.gather_for_profile(
        user_id=1, profile_id=profile_id, settings=settings,
    )
    assert run_id is not None

    conn = connect(db_path)
    try:
        row = gather_runs_repo.get(conn, run_id)
        assert row["error"] is not None
        assert "429" in row["error"]
        assert row["finished_at"] is not None
    finally:
        conn.close()


def test_gather_for_profile_returns_none_when_profile_missing(
    monkeypatch, db_path, settings
):
    _patch_dependencies(monkeypatch, [])
    result = scheduler_jobs.gather_for_profile(
        user_id=1, profile_id=99999, settings=settings,
    )
    assert result is None


def test_gather_for_profile_uses_users_mailto_when_present(
    monkeypatch, db_path, settings, profile_id
):
    """Mailto resolution: users.mailto > users.email > settings default."""
    captured: dict[str, Any] = {}

    class _CapturingGatherer:
        def __init__(self, **kw):
            captured["mailto"] = kw.get("mailto")

        def fetch(self, *a, **kw):
            return []

        def cost(self):
            return {"api_calls": 0}

    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer", _CapturingGatherer)

    conn = connect(db_path)
    try:
        conn.execute("UPDATE users SET mailto = ? WHERE id = 1", ("custom@x.com",))
        conn.commit()
    finally:
        conn.close()

    scheduler_jobs.gather_for_profile(
        user_id=1, profile_id=profile_id, settings=settings,
    )
    assert captured["mailto"] == "custom@x.com"


# --------------------------------------------------------------------------
# Runner — boots without crashing, registers jobs from DB
# --------------------------------------------------------------------------


apscheduler = pytest.importorskip("apscheduler")


def test_runner_registers_one_job_per_enabled_schedule(
    db_path, settings, profile_id
):
    from rag_lib.scheduler import build_scheduler, start, stop

    # Schedule was backfilled by migration 0004 for the seeded profile.
    scheduler = build_scheduler(settings)
    try:
        start(scheduler, settings)
        job = scheduler.get_job(f"gather:{profile_id}")
        assert job is not None
        assert tuple(job.args) == (1, profile_id)
    finally:
        stop(scheduler)


def test_runner_skips_invalid_cron(db_path, settings, profile_id):
    from rag_lib.scheduler import build_scheduler, start, stop

    conn = connect(db_path)
    try:
        schedules_repo.upsert(conn, profile_id=profile_id, cron="not a cron")
    finally:
        conn.close()

    scheduler = build_scheduler(settings)
    try:
        start(scheduler, settings)  # must not raise
        assert scheduler.get_job(f"gather:{profile_id}") is None
    finally:
        stop(scheduler)


def test_gather_short_circuits_when_all_topics_disabled(
    monkeypatch, db_path, settings, profile_id
):
    """Switching every topic off means "gather nothing".

    The run must close cleanly (not as an error) without querying
    OpenAlex, and leave a message the profile page can surface.
    """
    conn = connect(db_path)
    try:
        conn.execute(
            "UPDATE profiles SET topic_filters_json = ? WHERE id = ?",
            (json.dumps({"topics": [
                {"id": "T1", "display_name": "Topic", "count": 1, "on": False}
            ]}), profile_id),
        )
        conn.commit()
    finally:
        conn.close()

    called = []

    class _ExplodingGatherer:
        def fetch(self, *a, **kw):
            called.append(1)
            raise AssertionError("gatherer must not run")

        def cost(self):
            return {"wall_seconds": 0.0, "api_calls": 0}

    monkeypatch.setattr(
        scheduler_jobs, "OpenAlexGatherer", lambda **kw: _ExplodingGatherer()
    )

    run_id = scheduler_jobs.gather_for_profile(
        user_id=1, profile_id=profile_id, settings=settings, days=7,
    )

    assert called == []
    conn = connect(db_path)
    try:
        row = gather_runs_repo.get(conn, run_id)
        assert row["error"] is None, "disabled topics is not an error"
        assert row["finished_at"] is not None
        assert row["n_fetched"] == 0
        assert row["tier_used"] == "no-enabled-topics"
        assert "switched off" in (row["last_message"] or "")
        n = conn.execute(
            "SELECT COUNT(*) c FROM profile_candidates WHERE profile_id=?",
            (profile_id,),
        ).fetchone()["c"]
        assert n == 0, "nothing should have been persisted"
    finally:
        conn.close()


def test_gather_runs_normally_when_some_topics_enabled(
    monkeypatch, db_path, settings, profile_id
):
    """One topic still on -> the pipeline proceeds as usual."""
    conn = connect(db_path)
    try:
        conn.execute(
            "UPDATE profiles SET topic_filters_json = ? WHERE id = ?",
            (json.dumps({"topics": [
                {"id": "T1", "display_name": "A", "count": 1, "on": False},
                {"id": "T2", "display_name": "B", "count": 1, "on": True},
            ]}), profile_id),
        )
        conn.commit()
    finally:
        conn.close()

    _patch_dependencies(monkeypatch, [_candidate_paper("W100")])
    run_id = scheduler_jobs.gather_for_profile(
        user_id=1, profile_id=profile_id, settings=settings, days=7,
    )
    conn = connect(db_path)
    try:
        row = gather_runs_repo.get(conn, run_id)
        assert row["error"] is None
        assert row["n_fetched"] == 1
    finally:
        conn.close()


def test_gather_persists_source_topic_attribution(
    monkeypatch, db_path, settings, profile_id
):
    """The gatherer's per-topic attribution has to reach the DB.

    This is the wiring the UI depends on to report what each topic
    toggle actually pulled in, and it spans three modules
    (gatherer -> jobs -> db_store), so it needs an end-to-end assertion.
    """
    candidates = [_candidate_paper("W100"), _candidate_paper("W101")]

    class _AttributingGatherer(_StubGatherer):
        last_tier_used = "per-topic-quota"
        last_source_topics = {"W100": "T-alpha", "W101": "T-beta"}

    monkeypatch.setattr(
        scheduler_jobs, "OpenAlexGatherer",
        lambda **kw: _AttributingGatherer(candidates),
    )

    scheduler_jobs.gather_for_profile(
        user_id=1, profile_id=profile_id, settings=settings, days=7,
    )

    conn = connect(db_path)
    try:
        rows = dict(conn.execute(
            "SELECT openalex_id, sourced_by_topic_id FROM profile_candidates "
            "WHERE profile_id=?", (profile_id,),
        ).fetchall())
        assert rows == {"W100": "T-alpha", "W101": "T-beta"}
    finally:
        conn.close()


def test_source_topic_attribution_is_not_rewritten_on_resurface(
    monkeypatch, db_path, settings, profile_id
):
    """First topic to surface a paper keeps the credit.

    A later gather where a different topic's quota happens to claim it
    first must not rewrite the history the UI reports on.
    """
    candidates = [_candidate_paper("W100")]

    def _run(topic: str):
        class _G(_StubGatherer):
            last_tier_used = "per-topic-quota"
            last_source_topics = {"W100": topic}
        monkeypatch.setattr(
            scheduler_jobs, "OpenAlexGatherer", lambda **kw: _G(candidates)
        )
        scheduler_jobs.gather_for_profile(
            user_id=1, profile_id=profile_id, settings=settings, days=7,
        )

    _run("T-first")
    _run("T-second")

    conn = connect(db_path)
    try:
        got = conn.execute(
            "SELECT sourced_by_topic_id FROM profile_candidates "
            "WHERE profile_id=? AND openalex_id='W100'", (profile_id,),
        ).fetchone()[0]
        assert got == "T-first"
    finally:
        conn.close()


def test_two_gathers_leave_one_pool_on_one_scale(monkeypatch, db_path, settings, profile_id):
    """Each gather's best paper used to keep its batch's top percentile
    (1.0) and blend; the feed then ranked both batches together. After the
    second gather the pool must have exactly one top row."""
    _patch_dependencies(monkeypatch, [_candidate_paper("W300"), _candidate_paper("W301")])
    scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings, days=7)
    _patch_dependencies(monkeypatch, [_candidate_paper("W400"), _candidate_paper("W401")])
    scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings, days=7)

    conn = connect(db_path)
    try:
        pcts = [r[0] for r in conn.execute(
            "SELECT score_pct FROM profile_candidates WHERE profile_id = ?", (profile_id,))]
    finally:
        conn.close()
    assert len(pcts) == 4
    assert pcts.count(1.0) == 1, pcts
    assert sorted(pcts) == [0.0, 1 / 3, 2 / 3, 1.0]


def test_runner_registers_the_umls_backfill_sweep(db_path, settings, profile_id):
    from rag_lib.scheduler import build_scheduler, start, stop

    scheduler = build_scheduler(settings)
    try:
        start(scheduler, settings)
        job = scheduler.get_job("umls-backfill")
        assert job is not None
        # A naive datetime.now() is local time, which the UTC scheduler read
        # as UTC: on an EDT host the first run landed ~4 hours in the past.
        from datetime import datetime, timedelta, timezone

        ahead = job.next_run_time - datetime.now(timezone.utc)
        assert timedelta(0) < ahead <= timedelta(minutes=5), ahead
    finally:
        stop(scheduler)


def test_a_gather_stores_authors_and_date_and_never_blanks_an_import(
    monkeypatch, db_path, settings, profile_id,
):
    """Every feed card read 'Unknown authors' and a date of <year>-01-01:
    Paper had neither field, so both were lost at OpenAlex -> Paper."""
    from rag_lib.db.repos import papers as papers_repo

    conn = connect(db_path)
    papers_repo.upsert(conn, {"openalex_id": "W501", "title": "imported", "source": "prosopia",
                              "authors": ["Imported Author"]})
    conn.close()

    fresh = _candidate_paper("W500")
    fresh.authors = ["Ada Lovelace", "Alan Turing"]
    fresh.publication_date = "2026-04-15"
    known = _candidate_paper("W501")          # gathered again, no byline this time
    _patch_dependencies(monkeypatch, [fresh, known])
    scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings, days=7)

    conn = connect(db_path)
    try:
        rows = {r[0]: (r[1], r[2]) for r in conn.execute(
            "SELECT openalex_id, authors_json, publication_date FROM papers WHERE openalex_id IN ('W500','W501')")}
    finally:
        conn.close()
    assert rows["W500"] == ('["Ada Lovelace", "Alan Turing"]', "2026-04-15")
    assert rows["W501"][0] == '["Imported Author"]'


def test_an_unknown_timezone_is_invalid_not_an_exception():
    """ZoneInfoNotFoundError is a KeyError; caught as anything else it escaped,
    and a stored bad tz made start() raise -- the backend would not boot."""
    from rag_lib.scheduler.runner import register_gather_job, valid_cron

    assert valid_cron("0 4 * * *", "Bad/Zone") is not None

    class NoJobs:
        def add_job(self, *a, **k):
            raise AssertionError("must not register")

    assert register_gather_job(NoJobs(), user_id=1, profile_id=1, cron="0 4 * * *", tz="Bad/Zone") is False


def test_a_gather_keeps_the_papers_below_the_threshold_out_of_the_feed(
    monkeypatch, db_path, settings, profile_id,
):
    """Gathers used to drop every paper below the threshold, so lowering it
    later showed nothing new. They are kept now, but not in the feed, not
    in the pool's percentiles, and not counted as the run's new papers."""
    from rag_lib.db.repos import candidates as candidates_repo
    from rag_lib.scoring.pool import passes

    conn = connect(db_path)
    # In the selector config too, as every committed interest has it: a
    # selector rebuilt from config falls back to it when select() is given
    # threshold=None, which silently kept dropping these papers.
    conn.execute(
        "UPDATE profiles SET threshold = 1.01, "
        "selector_config_json = json_set(selector_config_json, '$.threshold', 1.01) WHERE id = ?",
        (profile_id,))
    conn.commit()
    conn.close()
    ids = ["W600", "W601", "W602", "W603"]
    _patch_dependencies(monkeypatch, [_candidate_paper(i) for i in ids])
    run_id = scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings, days=7)

    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT openalex_id, score_raw, score_pct FROM profile_candidates WHERE profile_id = ?",
            (profile_id,)).fetchall()
        assert sorted(r[0] for r in rows) == ids          # all stored ...
        assert all(r[2] is None for r in rows)            # ... none ranked
        assert candidates_repo.top_for_profile(conn, profile_id) == []
        run = conn.execute("SELECT n_new, n_fetched FROM gather_runs WHERE id = ?", (run_id,)).fetchone()
        assert (run[0], run[1]) == (0, 4)

        # Lower the threshold to the middle score: exactly the papers at or
        # above it reach the feed, and the percentiles cover only them.
        raws = sorted(r[1] for r in rows)
        mid = raws[2]
        conn.execute("UPDATE profiles SET threshold = ? WHERE id = ?", (mid, profile_id))
        conn.commit()
        scheduler_jobs.rescore_pool(conn, profile_id, settings)
        feed = {r["openalex_id"] for r in candidates_repo.top_for_profile(conn, profile_id)}
        assert feed == {r[0] for r in rows if passes(r[1], mid)} and len(feed) == 2
        pcts = dict(conn.execute(
            "SELECT openalex_id, score_pct FROM profile_candidates WHERE profile_id = ?", (profile_id,)).fetchall())
        assert sorted(pcts[i] for i in feed) == [0.0, 1.0]
        assert all(pcts[i] is None for i in ids if i not in feed)
    finally:
        conn.close()


def test_a_gather_stores_the_vectors_it_computes(monkeypatch, db_path, settings, profile_id):
    """Candidate vectors were computed and thrown away, so re-scoring stored
    candidates after a refit would have meant embedding them all again."""
    from rag_lib.db.repos import embeddings as embeddings_repo
    from rag_lib.selectors import centroid as centroid_mod

    fresh = _candidate_paper("W700")
    fresh.embeddings = {}                             # arrives without a vector, as from OpenAlex
    monkeypatch.setattr(centroid_mod, "get_embedder", lambda key: (lambda text: [0.3] * 8))
    _patch_dependencies(monkeypatch, [fresh])
    scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings, days=7)

    conn = connect(db_path)
    try:
        vec = embeddings_repo.get(conn, "W700", "specter2")
    finally:
        conn.close()
    assert vec is not None and np.allclose(vec, [0.3] * 8)


def test_a_backfill_stores_papers_from_the_years_asked_for(monkeypatch, db_path, settings, profile_id):
    """Scans only ever looked back a day, so an interest never held an older
    paper. ``years`` fetches those years instead, and the rest of the gather
    -- scoring, keeping what falls below the threshold -- is unchanged."""
    asked = {}

    class YearStub(_StubGatherer):
        def fetch(self, *a, **kw):
            raise AssertionError("a backfill must not fetch by days")

        def fetch_years(self, profile, year_from, year_to, *, per_year):
            asked.update(span=(year_from, year_to), per_year=per_year)
            self.last_tier_used = "backfill-years"
            out = []
            for i, y in enumerate((2018, 2019, 2020)):
                p = _candidate_paper(f"W80{i}")
                p.year = y
                out.append(p)
            return out

    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer", lambda **kw: YearStub([]))
    run_id = scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings,
                                               years=(2018, 2020), tier="backfill")
    conn = connect(db_path)
    try:
        stored = {r[0] for r in conn.execute(
            "SELECT openalex_id FROM profile_candidates WHERE profile_id = ?", (profile_id,))}
        run = conn.execute("SELECT tier_used, since_date, error FROM gather_runs WHERE id = ?", (run_id,)).fetchone()
    finally:
        conn.close()
    assert asked == {"span": (2018, 2020), "per_year": 100}
    assert stored == {"W800", "W801", "W802"}
    assert run[2] is None and run[0] == "backfill-years" and run[1] == "2018-01-01"


# --------------------------------------------------------------------------
# Reranker logits: stale (queries changed) or missing (started passing later)
# --------------------------------------------------------------------------


class _FakeReranker:
    """Logit = 10 x the query-set version, so a stale logit is easy to spot."""

    name = "medcpt"
    alpha, beta = 0.4, 0.6

    def __init__(self, version: int = 1):
        self.version = version
        self.seen: list[str] = []

    def query_key(self, profile):
        return f"K{self.version}"

    def rerank(self, ranked, profile, *, conn=None):
        out = []
        for s, p, bd in ranked:
            self.seen.append(p.openalex_id)
            bd = {**bd, "score_reranker_raw": 10.0 * self.version + float(bd.get("score_raw") or s),
                  "reranker_query_key": f"K{self.version}", "score_blended": s}
            out.append((s, p, bd))
        return out

    def diagnostics(self):
        return {"status": "ok"}

    def cost(self):
        return {"wall_seconds": 0.0}


def _rr_rows(db_path, profile_id):
    conn = connect(db_path)
    try:
        return {r[0]: (r[1], r[2], r[3]) for r in conn.execute(
            "SELECT openalex_id, score_reranker_raw, reranker_query_key, score_blended "
            "FROM profile_candidates WHERE profile_id = ?", (profile_id,))}
    finally:
        conn.close()


def test_a_query_change_reranks_the_pool_and_a_newly_passing_paper_gets_a_logit(
    monkeypatch, db_path, settings, profile_id,
):
    from rag_lib.db.repos import candidates as candidates_repo, papers as papers_repo

    rr = _FakeReranker(1)
    monkeypatch.setattr(scheduler_jobs, "_build_reranker", lambda row, s: rr)
    _patch_dependencies(monkeypatch, [_candidate_paper("W300"), _candidate_paper("W301")])
    scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings, days=7)
    rows = _rr_rows(db_path, profile_id)
    assert {k: v[1] for k, v in rows.items()} == {"W300": "K1", "W301": "K1"}

    # A paper stored without a logit (it was below the threshold at its scan
    # and passes now), and one below the threshold that must stay untouched.
    conn = connect(db_path)
    try:
        for oa in ("W500", "W501"):
            papers_repo.upsert(conn, {"openalex_id": oa, "title": f"Old {oa}", "abstract": "a", "year": 2025})
        candidates_repo.insert_dedup(conn, profile_id=profile_id, openalex_id="W500", score=0.9, score_raw=0.9)
        candidates_repo.insert_dedup(conn, profile_id=profile_id, openalex_id="W501", score=0.1, score_raw=0.1)
        conn.execute("UPDATE profiles SET threshold = 0.5 WHERE id = ?", (profile_id,))
        conn.execute("UPDATE profile_candidates SET score_raw = 0.95 WHERE openalex_id IN ('W300','W301')")
        conn.commit()

        # The queries change (a seed or topic edit): every pool row is stale.
        rr2 = _FakeReranker(2)
        n = scheduler_jobs.refresh_reranker(conn, profile_id, settings, reranker=rr2)
    finally:
        conn.close()
    assert n == 3 and sorted(rr2.seen) == ["W300", "W301", "W500"]
    rows = _rr_rows(db_path, profile_id)
    for oa in ("W300", "W301", "W500"):
        assert rows[oa][1] == "K2" and rows[oa][0] >= 20.0 and rows[oa][2] is not None, (oa, rows[oa])
    assert rows["W501"][0] is None and rows["W501"][2] is None      # below the bar: not reranked

    # Nothing stale left: a second refresh reranks nothing.
    conn = connect(db_path)
    try:
        assert scheduler_jobs.refresh_reranker(conn, profile_id, settings, reranker=_FakeReranker(2)) == 0
    finally:
        conn.close()


def test_a_rescan_without_reranking_keeps_the_key_with_its_logit(db_path, profile_id):
    from rag_lib.db.repos import candidates as candidates_repo, papers as papers_repo

    conn = connect(db_path)
    try:
        papers_repo.upsert(conn, {"openalex_id": "W600", "title": "x", "year": 2025})
        candidates_repo.insert_dedup(conn, profile_id=profile_id, openalex_id="W600", score=0.9,
                                     score_raw=0.9, score_reranker_raw=3.0, reranker_query_key="K1")
        # Seen again below the threshold: no new logit, so the old key stays.
        candidates_repo.insert_dedup(conn, profile_id=profile_id, openalex_id="W600", score=0.2, score_raw=0.2)
        assert tuple(conn.execute("SELECT score_reranker_raw, reranker_query_key FROM profile_candidates "
                            "WHERE openalex_id = 'W600'").fetchone()) == (3.0, "K1")
        # Reranked again: logit and key move together.
        candidates_repo.insert_dedup(conn, profile_id=profile_id, openalex_id="W600", score=0.9,
                                     score_raw=0.9, score_reranker_raw=5.0, reranker_query_key="K2")
        assert tuple(conn.execute("SELECT score_reranker_raw, reranker_query_key FROM profile_candidates "
                            "WHERE openalex_id = 'W600'").fetchone()) == (5.0, "K2")
    finally:
        conn.close()


def test_rescore_pool_asks_for_a_background_refresh(monkeypatch, db_path, settings, profile_id):
    added = []

    class _Sched:
        def add_job(self, fn, **kw):
            added.append((fn, kw["id"], kw["args"]))

    monkeypatch.setattr(scheduler_jobs, "_SCHEDULER", _Sched())
    monkeypatch.setattr(scheduler_jobs, "_build_reranker", lambda row, s: _FakeReranker())
    conn = connect(db_path)
    try:
        scheduler_jobs.rescore_pool(conn, profile_id, settings)
    finally:
        conn.close()
    assert added == [(scheduler_jobs.refresh_reranker_job, f"rerank-refresh:{profile_id}", [profile_id])]

    # The noop reranker has nothing to refresh.
    added.clear()
    from rag_lib.rerankers import NoopReranker
    monkeypatch.setattr(scheduler_jobs, "_build_reranker", lambda row, s: NoopReranker())
    conn = connect(db_path)
    try:
        scheduler_jobs.rescore_pool(conn, profile_id, settings)
    finally:
        conn.close()
    assert added == []


def test_runner_registers_the_rerank_sweep_and_hands_jobs_the_scheduler(db_path, settings, profile_id):
    from rag_lib.scheduler import build_scheduler, start, stop

    scheduler = build_scheduler(settings)
    try:
        start(scheduler, settings)
        assert scheduler.get_job("rerank-refresh-sweep") is not None
        assert scheduler_jobs._SCHEDULER is scheduler
    finally:
        stop(scheduler)
    assert scheduler_jobs._SCHEDULER is None


def test_a_refresh_does_not_overwrite_a_logit_a_scan_wrote_meanwhile(monkeypatch, db_path, settings, profile_id):
    from rag_lib.db.repos import candidates as candidates_repo, papers as papers_repo

    conn = connect(db_path)
    try:
        for oa in ("W700", "W701"):
            papers_repo.upsert(conn, {"openalex_id": oa, "title": oa, "year": 2025})
            candidates_repo.insert_dedup(conn, profile_id=profile_id, openalex_id=oa, score=0.9,
                                         score_raw=0.9, score_reranker_raw=1.0, reranker_query_key="K1")

        class Slow(_FakeReranker):
            def rerank(self, ranked, profile, *, conn=None):
                # While this runs, a scan reranks W700 against newer queries.
                other = connect(db_path)
                try:
                    candidates_repo.insert_dedup(other, profile_id=profile_id, openalex_id="W700", score=0.9,
                                                 score_raw=0.9, score_reranker_raw=99.0, reranker_query_key="K3")
                finally:
                    other.close()
                return super().rerank(ranked, profile)

        scheduler_jobs.refresh_reranker(conn, profile_id, settings, reranker=Slow(2))
        got = {r[0]: (r[1], r[2]) for r in conn.execute(
            "SELECT openalex_id, score_reranker_raw, reranker_query_key FROM profile_candidates "
            "WHERE openalex_id IN ('W700','W701')")}
    finally:
        conn.close()
    assert got["W700"] == (99.0, "K3")          # the scan's newer logit stands
    assert got["W701"][1] == "K2"               # the untouched row was refreshed


def test_a_refresh_asked_for_while_one_runs_makes_the_running_one_go_round_again(
    monkeypatch, db_path, settings, profile_id,
):
    passes = []

    def once(conn, pid, s, *, reranker=None, profile=None, limit=0):
        passes.append(reranker)
        if len(passes) == 1:
            # Another thread asks meanwhile (a seed edit, the sweep): it
            # finds the lock taken and must not be lost.
            assert scheduler_jobs.refresh_reranker(conn, pid, s) == 0
        return 1

    monkeypatch.setattr(scheduler_jobs, "_refresh_once", once)
    conn = connect(db_path)
    try:
        n = scheduler_jobs.refresh_reranker(conn, profile_id, settings, reranker="the scan's")
    finally:
        conn.close()
    # Ran again, and with the reranker rebuilt: the queries may have changed.
    assert n == 2 and passes == ["the scan's", None]


def test_a_scan_that_overlapped_a_seed_change_rescores_against_the_new_seeds(
    monkeypatch, db_path, settings, profile_id,
):
    """The batch was scored against the centroid read at the start; the
    refit's rescore ran before these rows went in."""
    from rag_lib.scoring import rescore

    calls = []

    class RefitMidway(_StubGatherer):
        def fetch(self, profile, *, since, limit):
            c = connect(db_path)          # a seed edit's refit, mid-scan
            try:
                c.execute("UPDATE profiles SET selector_config_json = json_set(selector_config_json, '$.refit', 1) WHERE id = ?", (profile_id,))
                c.commit()
            finally:
                c.close()
            return super().fetch(profile, since=since, limit=limit)

    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer", lambda **kw: RefitMidway([_candidate_paper("W800")]))
    monkeypatch.setattr(rescore, "rescore_candidates", lambda conn, pid: calls.append(pid) or {})
    scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings, days=7)
    assert calls == [profile_id]

    calls.clear()      # an ordinary scan does not
    _patch_dependencies(monkeypatch, [_candidate_paper("W801")])
    scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings, days=7)
    assert calls == []
