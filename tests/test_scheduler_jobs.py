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
