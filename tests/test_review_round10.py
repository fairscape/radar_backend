"""Round 10: a seventh independent review's findings (four reviewers over
the whole backend: feedback logging, kick-off of background runs, first
contact of a new email, refits under a running scan, unreachable ids)."""

from __future__ import annotations

import sqlite3

import pytest

from rag_lib import prosopia_seeds
from rag_lib.api.services import vault as vault_service, wizard
from rag_lib.db import connect
from rag_lib.db.repos import (
    embeddings as embeddings_repo,
    gather_runs as gather_runs_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    users as users_repo,
)
from tests.test_api_radar import PAPER, _request, _saved, app  # noqa: F401  (fixture)
from tests.test_resolve_explicit_id import RECORD, Stub, _http_error, no_waiting  # noqa: F401  (fixture)
from tests.test_review_round4 import _db


# --- feedback logging is best effort ---------------------------------------------


def test_a_feedback_log_that_cannot_be_written_does_not_undo_the_save(app, tmp_path, monkeypatch):
    """The state change is committed before the event is logged; an
    unwritable vault dir then answered 500 for a save that had happened,
    and the retry logged nothing (no transition any more)."""
    from rag_lib.api.routers import radar as radar_router

    def boom(*a, **k):
        raise OSError("read-only file system")

    monkeypatch.setattr(radar_router, "log_event", boom)
    res = _request(app, "POST", "/api/radar/cards/save",
                   json={"card_id": PAPER, "profile": "provenance-fairscape", "on": True})
    assert res.status_code == 200, res.text
    assert res.json()["state"] == "saved"
    assert _saved(tmp_path)["provenance-fairscape"] is True


def test_an_older_client_is_told_which_interest_the_change_landed_on(app):
    res = _request(app, "POST", "/api/radar/cards/save", json={"card_id": PAPER, "on": True})
    assert res.status_code == 200, res.text
    assert res.json()["profile"] == "provenance-fairscape"


# --- a run the scheduler refuses is closed -------------------------------------


class _RefusingScheduler:
    def add_job(self, *a, **k):
        raise RuntimeError("scheduler is shutting down")


def test_a_scan_the_scheduler_refuses_closes_its_run(app, tmp_path):
    """Open, the row counted as a running scan for two hours: every "Scan
    now" and backfill answered 409 and the Scans tab showed it in flight."""
    from rag_lib.api.deps import get_scheduler

    app.dependency_overrides[get_scheduler] = lambda: _RefusingScheduler()
    try:
        res = _request(app, "POST", "/api/profiles/provenance-fairscape/gather-now")
    finally:
        app.dependency_overrides.pop(get_scheduler, None)
    assert res.status_code == 503, res.text
    conn = connect(tmp_path / "radar.db")
    try:
        row = conn.execute("SELECT finished_at, error FROM gather_runs WHERE tier_used = 'manual' ORDER BY id DESC LIMIT 1").fetchone()
        assert row is not None and row["finished_at"] is not None and "could not schedule" in row["error"]
        pid = conn.execute("SELECT id FROM profiles WHERE slug = 'provenance-fairscape'").fetchone()[0]
        assert gather_runs_repo.running_scan(conn, int(pid)) is None
    finally:
        conn.close()


# --- first contact of a new email -----------------------------------------------


def test_two_first_requests_for_one_new_email_both_get_the_row(tmp_path, monkeypatch):
    """Both found no row and both inserted; the second failed on the unique
    email, a 500 on a brand-new person's first page load."""
    conn, _uid = _db(tmp_path)
    other = connect(tmp_path / "r.db")
    try:
        # The other request's insert lands between this one's check and insert.
        real = users_repo.get_by_email
        seen = {"n": 0}

        def racing(c, email):
            seen["n"] += 1
            if seen["n"] == 1:
                users_repo.upsert(other, email)
                return None
            return real(c, email)

        monkeypatch.setattr(users_repo, "get_by_email", racing)
        row = users_repo.upsert(conn, "new@example.com")
        assert row is not None and row["email"] == "new@example.com"
        monkeypatch.undo()
        assert conn.execute("SELECT COUNT(*) FROM users WHERE email = 'new@example.com'").fetchone()[0] == 1
    finally:
        other.close()
        conn.close()


def test_create_twice_at_once_takes_the_next_free_name(tmp_path, monkeypatch):
    conn, uid = _db(tmp_path)
    first = wizard.create_draft(conn, user_id=uid, name="Dup", embedding_model="placeholder-v1")
    real = wizard._unique_slug
    calls = {"n": 0}

    def stale(c, user_id, base_slug, name):
        calls["n"] += 1
        if calls["n"] == 1:
            return first["slug"], 0          # what the other request saw a moment ago
        return real(c, user_id, base_slug, name)

    monkeypatch.setattr(wizard, "_unique_slug", stale)
    second = wizard.create_draft(conn, user_id=uid, name="Dup", embedding_model="placeholder-v1")
    assert second["slug"] != first["slug"] and second["name"] != first["name"]
    assert profiles_repo.get_by_slug(conn, uid, second["slug"]) is not None
    conn.close()


# --- the seed count ---------------------------------------------------------------


def test_coherence_over_embedded_seeds_keeps_the_seed_row_count(tmp_path):
    """One seed without a vector: ``n_seed`` was the number embedded, which
    every other writer disagrees with (12 here, 13 there, on the same interest)."""
    conn, uid = _db(tmp_path)
    d = wizard.create_draft(conn, user_id=uid, name="D", embedding_model="placeholder-v1")
    pid = int(profiles_repo.get_by_slug(conn, uid, d["slug"])["id"])
    for j, oa in enumerate(("https://openalex.org/W1", "https://openalex.org/W2", "https://openalex.org/W3")):
        papers_repo.upsert(conn, {"openalex_id": oa, "title": f"t{j}", "abstract": "words " * 30, "source": "openalex"})
        profiles_repo.attach_seed(conn, pid, oa)
        if j < 2:
            embeddings_repo.upsert(conn, oa, "placeholder-v1", [0.1 * (j + 1)] * 8)
    wizard.compute_draft_coherence(conn, user_id=uid, slug=d["slug"])
    assert profiles_repo.get_by_slug(conn, uid, d["slug"])["n_seed"] == 3
    conn.close()


# --- refits under a running scan ----------------------------------------------------


def _live_interest(conn, uid, name="Live"):
    pid = profiles_repo.upsert(conn, user_id=uid, name=name, embedding_model="placeholder-v1",
                               n_seed=1, topic_filters={"topics": []}, threshold=0.5)
    conn.execute("UPDATE profiles SET is_draft = 0 WHERE id = ?", (pid,))
    conn.commit()
    return pid, profiles_repo.get_by_slug(conn, uid, name.lower())["slug"]


def test_a_refit_is_refused_while_a_scan_runs_and_an_upload_still_lands(tmp_path):
    conn, uid = _db(tmp_path)
    pid, slug = _live_interest(conn, uid)
    papers_repo.upsert(conn, {"openalex_id": "local:abc", "title": "Notes", "body_text": "words " * 20, "source": "user_pdf"})
    gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="manual")   # still running
    with pytest.raises(wizard.ScanRunning):
        wizard.refit_profile(conn, user_id=uid, slug=slug)
    vault_service._attach_to_profile(conn, uid, slug, "local:abc")                 # no 500 for a stored upload
    assert "local:abc" in profiles_repo.list_seed_openalex_ids(conn, pid)
    conn.close()


# --- an unreachable OpenAlex id ------------------------------------------------------


def test_an_unreachable_openalex_id_keeps_its_identity(no_waiting):
    """Rate limited on every lookup, the record used to be stored under a
    synthetic id; a later refresh then offered the same work as new, and
    nothing could dedup a scan's copy against it."""
    client = Stub([_http_error(429), _http_error(429)])
    res = prosopia_seeds.resolve_seed(RECORD, client, slug="0000-0001-5643-4068")
    assert res.rung == "none" and not res.resolved
    assert res.openalex_id == "https://openalex.org/W1"
    assert res.paper["openalex_id"] == "https://openalex.org/W1"
    assert client.title_searches == 0


def test_a_record_openalex_never_held_keeps_a_synthetic_id(no_waiting):
    client = Stub([None])                                     # a 404: the id is wrong
    res = prosopia_seeds.resolve_seed(RECORD, client, slug="0000-0001-5643-4068")
    assert res.openalex_id.startswith("prosopia:") or res.rung == "title"


# --- the ORCID registry ----------------------------------------------------------------


def test_a_registry_answer_that_is_not_an_object_reads_as_unavailable(monkeypatch, live_orcid_registry):
    import rag_lib.orcid_registry as reg

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return ["not", "an", "object"]

    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    assert reg.fetch_claimed("0000-0001-5643-4068") is None


# --- one scratch file per upload ---------------------------------------------------------


def test_the_upload_scratch_name_is_per_request():
    import inspect

    src = inspect.getsource(vault_service.upload)
    assert "threading.get_ident()" in src and ".part" in src
