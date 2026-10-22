"""Review 6 (a): deleting a saved interest while a request on it is mid-flight."""

from __future__ import annotations

from rag_lib.api.services import vault as vault_service
from rag_lib.db import connect
from rag_lib.db.repos import profiles as profiles_repo
from tests.test_review_round12_journeys import (  # noqa: F401
    env, app, _request, _ok, _go_live, _one, _upload,
)


def _delete_now(env, pid):
    conn = connect(env)
    try:
        profiles_repo.delete(conn, pid)
    finally:
        conn.close()


def test_scan_now_racing_a_delete_is_not_a_500(app, env, monkeypatch):
    """gather-now reads the row, then a DELETE lands (another request thread),
    then gather-now opens its run row for the vanished id."""
    from rag_lib.api.routers import profiles as profiles_router

    slug, _ = _go_live(app, env, name="Race")
    pid = _one(env, "SELECT id FROM profiles WHERE slug = ?", slug)["id"]
    real = profiles_router._refuse_if_scanning

    def _then_deleted(db, profile_id):
        real(db, profile_id)
        _delete_now(env, profile_id)      # the concurrent DELETE commits here

    monkeypatch.setattr(profiles_router, "_refuse_if_scanning", _then_deleted)
    try:
        res = _request(app, "POST", f"/api/profiles/{slug}/gather-now?days=7")
        code = res.status_code
    except Exception as exc:  # noqa: BLE001 -- ASGITransport re-raises app errors
        code = f"unhandled {type(exc).__name__}: {exc}"
    assert code in (404, 409), code


def test_upload_to_an_interest_deleted_mid_upload_says_so(app, env, monkeypatch):
    """The Seeds tab's PDF drop: the interest is deleted while the PDF is read."""
    slug, _ = _go_live(app, env, name="Upload Race", seeds_from_researcher=False)
    pid = _one(env, "SELECT id FROM profiles WHERE slug = ?", slug)["id"]
    real_ingest = vault_service.ingest_pdf

    def _ingest_then_delete(path):
        rec = real_ingest(path)
        _delete_now(env, pid)
        return rec

    monkeypatch.setattr(vault_service, "ingest_pdf", _ingest_then_delete)
    res = _upload(app, b"race-body", "race.pdf", slug)
    # 200 with the paper stored and no seed anywhere: the user was told it
    # was added to the interest.
    assert res.status_code != 200, res.json()
