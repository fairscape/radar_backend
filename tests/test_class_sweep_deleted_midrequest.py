"""Class sweep: the object a request works on is deleted by another tab
part-way through the request.

Rounds 10-18 found this one call at a time (each a 500, some leaving an
orphan draft or a run stuck "running"). Here every write route is run
again and again; on the n-th run the target (interest, draft or
researcher) is deleted on another connection just before the request's
n-th transaction begins, for every n the request reaches. Each run must
answer below 500, leave no run unfinished and, when it fails, create no
interest or draft.
"""

from __future__ import annotations

import sqlite3

import pytest

from rag_lib.db import connect
from tests.edit_helpers import edit_saved
from tests.test_review_round12_journeys import (  # noqa: F401 -- fixtures
    _go_live, _import_researcher, _ok, _q, _request, _scan, _upload, app, env,
)

MAX_TX = 12


def _copy(frm, to):
    src, dst = sqlite3.connect(frm), sqlite3.connect(to)
    src.backup(dst)
    src.close(), dst.close()


def _db_path(env):
    import os
    return os.environ["RADAR_DB_PATH"]


@pytest.fixture()
def world(app, env, tmp_path):
    live, rid = _go_live(app, env, "Live One")
    _scan(app, live)
    draft = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": "Just Draft"}))["slug"]
    _ok(_upload(app, b"draft-seed", "d.pdf", draft))
    rdraft = _ok(_request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": "From R"}))
    rdraft = (rdraft.get("draft") or rdraft)["slug"]
    seeds = [r[0] for r in _q(env, "SELECT ps.openalex_id FROM profile_seeds ps JOIN profiles p ON p.id = ps.profile_id WHERE p.slug = ?", live)]
    card = _q(env, "SELECT pc.openalex_id FROM profile_candidates pc JOIN profiles p ON p.id = pc.profile_id WHERE p.slug = ? LIMIT 1", live)[0][0]
    snap = tmp_path / "snap.db"
    _copy(_db_path(env), snap)
    # The same world with Live One open in Edit interest: every change to a
    # saved interest now goes through its copy (spec E), so the sweep runs
    # the copy's routes too, from this snapshot. The DB is put back after.
    copy = _ok(_request(app, "POST", f"/api/profiles/{live}/edit"))["slug"]
    snap_edit = tmp_path / "snap_edit.db"
    _copy(_db_path(env), snap_edit)
    _copy(snap, _db_path(env))
    return dict(live=live, draft=draft, rdraft=rdraft, rid=rid, seeds=seeds, card=card, snap=snap,
                copy=copy, snap_edit=snap_edit)


def _calls(w):
    live, draft, rid, s0, card = w["live"], w["draft"], w["rid"], w["seeds"][0], w["card"]
    P, R = "profile", "researcher"
    return [
        (P, live, "PATCH", f"/api/profiles/{live}", {"json": {"threshold": 0.2}}),
        (P, live, "PATCH", f"/api/profiles/{live}", {"json": {"years": {"year_from": 2019, "year_to": None}}}),
        (P, live, "POST", f"/api/profiles/{live}/refit", {}),
        (P, live, "POST", f"/api/profiles/{live}/refit?autotune=1", {}),
        (P, live, "POST", f"/api/profiles/{live}/dry-run", {"json": {}}),
        (P, live, "POST", f"/api/profiles/{live}/recompute-coherence", {}),
        (P, live, "POST", f"/api/profiles/{live}/recompute-topics", {}),
        (P, live, "PUT", f"/api/profiles/{live}/researcher-topics", {"json": {"enabled": True}}),
        (P, live, "PUT", f"/api/profiles/{live}/researcher-topics", {"json": {"enabled": False}}),
        (P, live, "POST", f"/api/profiles/{live}/seeds", {"json": {"openalex_ids": [s0]}}),
        (P, live, "DELETE", f"/api/profiles/{live}/seeds", {"params": {"openalex_id": s0}}),
        (P, live, "POST", f"/api/profiles/{live}/backfill", {"json": {"year_from": 2020}}),
        (P, live, "POST", f"/api/profiles/{live}/gather-now?days=7", {}),
        (P, live, "POST", "/api/radar/cards/save", {"json": {"card_id": card, "profile": live, "on": True}}),
        (P, live, "POST", "/api/radar/cards/dismiss", {"json": {"card_id": card, "profile": live, "on": True}}),
        (P, live, "DELETE", f"/api/profiles/{live}", {}),
        (P, live, "UPLOAD", live, {}),
        (P, draft, "POST", f"/api/profiles/draft/{draft}/coherence", {}),
        (P, draft, "POST", f"/api/profiles/draft/{draft}/dry-run", {"json": {"days": 30}}),
        (P, draft, "POST", f"/api/profiles/draft/{draft}/seeds", {"json": {"openalex_ids": [s0]}}),
        (P, draft, "POST", "/api/profiles", {"json": {"slug": draft, "threshold": 0.1, "selected_topic_ids": []}}),
        (P, draft, "DELETE", f"/api/profiles/draft/{draft}", {}),
        (P, draft, "UPLOAD", draft, {}),
        (P, w["rdraft"], "POST", "/api/profiles", {"json": {"slug": w["rdraft"], "threshold": 0.1, "selected_topic_ids": []}}),
        (R, rid, "PATCH", f"/api/researchers/{rid}", {"json": {"mark": "self"}}),
        (R, rid, "PUT", "/api/researchers/order", {"json": {"ids": [rid]}}),
        (R, rid, "POST", f"/api/researchers/{rid}/interests", {"json": {"name": "X"}}),
        (R, rid, "DELETE", f"/api/researchers/{rid}", {}),
        (R, rid, "POST", "/api/researchers/import", {"json": {"source": "prosopia", "ref": "sheffield-nathan"}}),
        (P, draft, "POST", "/api/import/prosopia", {"json": {"ref": "sheffield-nathan", "into": draft}}),
    ]


def _edit_calls(w):
    """The copy's routes (spec E), each run once with the copy deleted (the
    other tab's Discard) and once with the interest it copies deleted."""
    live, copy, s0 = w["live"], w["copy"], w["seeds"][0]
    calls = [
        ("PATCH", f"/api/profiles/{copy}", {"json": {"threshold": 0.2}}),
        ("PATCH", f"/api/profiles/{copy}", {"json": {"years": {"year_from": 2019, "year_to": None}}}),
        ("POST", f"/api/profiles/{copy}/recompute-coherence", {}),
        ("POST", f"/api/profiles/{copy}/recompute-topics", {}),
        ("PUT", f"/api/profiles/{copy}/researcher-topics", {"json": {"enabled": True}}),
        ("PUT", f"/api/profiles/{copy}/researcher-topics", {"json": {"enabled": False}}),
        ("POST", f"/api/profiles/{copy}/seeds", {"json": {"openalex_ids": [s0]}}),
        ("DELETE", f"/api/profiles/{copy}/seeds", {"params": {"openalex_id": s0}}),
        ("UPLOAD", copy, {}),
        ("POST", f"/api/profiles/draft/{copy}/save-edit", {}),
        ("DELETE", f"/api/profiles/draft/{copy}", {}),
    ]
    return [("profile", key, m, p, kw) for m, p, kw in calls for key in (copy, live)]


def _kill(kind, key):
    """What another tab's DELETE does, through the same service (with its
    refusals: a researcher mid-import, an interest mid-scan). Returns
    whether it deleted."""
    import os
    from rag_lib.api.services import profiles as profiles_service
    from rag_lib.api.services import researchers as researchers_service
    from rag_lib.api.services import wizard as wizard_service
    c = connect(os.environ["RADAR_DB_PATH"])
    try:
        if kind == "profile":
            row = c.execute("SELECT is_draft, edit_of FROM profiles WHERE slug = ? AND user_id = 1", (key,)).fetchone()
            if row is None:
                return False
            if row["edit_of"] is not None:
                from rag_lib.api.services import edit_copy as edit_copy_service
                edit_copy_service.discard_copy(c, user_id=1, slug=key)
                return True
            if row["is_draft"]:
                return bool(wizard_service.delete_draft(c, user_id=1, slug=key))
            return profiles_service.delete_profile(c, 1, key) is not None
        researchers_service.delete_researcher(c, 1, int(key))
        return True
    except (profiles_service.ProfileBusy, wizard_service.ImportInProgress,
            researchers_service.ResearcherImportRunning, researchers_service.ResearcherNotFound):
        return False
    finally:
        c.close()


def test_target_deleted_before_each_transaction(app, env, world, monkeypatch):
    import os
    from rag_lib.api import deps

    state = {"n": 0, "at": None, "kill": None, "fired": False}
    real_get_db = deps.get_db

    def traced_get_db(settings=None):
        conn = connect(os.environ["RADAR_DB_PATH"])

        def trace(sql):
            if not sql.startswith("BEGIN"):
                return
            state["n"] += 1
            if state["at"] is not None and state["n"] == state["at"] and not state["fired"]:
                state["fired"] = True
                state["deleted"] = state["kill"]()
        conn.set_trace_callback(trace)
        try:
            yield conn
        finally:
            conn.set_trace_callback(None)
            conn.close()

    app.dependency_overrides[real_get_db] = traced_get_db
    bad = []
    runs = [(world["snap"], c) for c in _calls(world)] + [(world["snap_edit"], c) for c in _edit_calls(world)]
    for snap, (kind, key, method, path, kw) in runs:
        for at in range(1, MAX_TX + 1):
            _copy(snap, _db_path(env))
            before = {r[0] for r in _q(env, "SELECT slug FROM profiles")}
            state.update(n=0, at=at, fired=False, deleted=False, kill=lambda kind=kind, key=key: _kill(kind, key))
            if method == "UPLOAD":
                r = _upload(app, f"sweep-{at}".encode(), "s.pdf", path)
            else:
                r = _request(app, method, path, **kw)
            if not state["fired"]:
                break  # the request has fewer transactions than ``at``
            open_runs = _q(env, "SELECT COUNT(*) FROM gather_runs WHERE finished_at IS NULL")[0][0]
            new = {r[0] for r in _q(env, "SELECT slug FROM profiles")} - before
            label = f"{method} {path} ({key} deleted before transaction {at})"
            if r.status_code >= 500:
                bad.append(f"{label}: {r.status_code} {r.text[:150]}")
            if open_runs:
                bad.append(f"{label}: {open_runs} run(s) left unfinished")
            if r.status_code >= 400 and new:
                bad.append(f"{label}: failed with {r.status_code} yet left {sorted(new)}")
            print(label, r.status_code, "deleted" if state["deleted"] else "delete refused")
    app.dependency_overrides.pop(real_get_db, None)
    assert not bad, "\n".join(bad)


def test_backstop_turns_a_missed_foreign_key_failure_into_404(app, env, world, monkeypatch):
    """A route the sweep does not reach still answers 404, not 500."""
    from rag_lib.api.services import wizard as wizard_service

    def gone(*a, **k):
        raise sqlite3.IntegrityError("FOREIGN KEY constraint failed")
    def dup(*a, **k):
        raise sqlite3.IntegrityError("UNIQUE constraint failed: profiles.slug")

    # Refit is a change: on a saved interest only its Edit-interest copy
    # takes it (spec E).
    with edit_saved(lambda m, p, **kw: _request(app, m, p, **kw), world["live"], save=False) as copy:
        monkeypatch.setattr(wizard_service, "refit_profile", gone)
        r = _request(app, "POST", f"/api/profiles/{copy}/refit")
        assert r.status_code == 404, r.text

        monkeypatch.setattr(wizard_service, "refit_profile", dup)
        r = _request(app, "POST", f"/api/profiles/{copy}/refit")
        assert r.status_code == 409, r.text
