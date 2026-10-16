"""Round 10: a seventh independent review's findings."""

from __future__ import annotations

from rag_lib.api.schemas import MAX_NAME
from rag_lib.api.services import prosopia as prosopia_service, wizard
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import profiles as profiles_repo
from tests.test_review_round4 import _Prof, _db


def test_a_name_an_import_derives_is_bounded_like_a_typed_one(tmp_path):
    from rag_lib.api import settings as settings_module

    class _Long(_Prof):
        def __init__(self):
            super().__init__()
            self.name = "N" * 320

    class _Client:
        def resolve_orcid(self, o):
            return "doe-jane"

        def fetch_profile(self, slug):
            return _Long()

    conn = connect(tmp_path / "r.db")
    apply_migrations(conn)
    plan = prosopia_service.prepare_import(conn, settings_module.get_settings(), user_id=1, ref="doe-jane",
                                           prosopia_client=_Client())
    assert len(plan.draft_name) == MAX_NAME


def test_a_suffix_keeps_a_name_within_the_bound(tmp_path):
    conn, uid = _db(tmp_path)
    name = "a" * MAX_NAME
    first = wizard.create_draft(conn, user_id=uid, name=name, embedding_model="placeholder-v1")
    second = wizard.create_draft(conn, user_id=uid, name=name, embedding_model="placeholder-v1")
    assert first["name"] == name
    assert len(second["name"]) == MAX_NAME and second["name"].endswith(" (2)")
    assert profiles_repo.get_by_slug(conn, uid, second["slug"])["name"] == second["name"]


def test_the_run_status_carries_its_last_progress_tick(tmp_path):
    """The client gives a run up on the server's staleness rule, which needs
    the tick the server keeps (progress_updated_at)."""
    from rag_lib.db.repos import gather_runs as gather_runs_repo
    from tests.test_review_round3 import _requests, app  # noqa: F401
    import pytest as _pt

    # Through the app fixture of round 3: a draft, a trial run with a tick.
    request = _pt.importorskip("tests.test_review_round3")
    appl = next(request.app.__wrapped__(tmp_path, _pt.MonkeyPatch())) if hasattr(request.app, "__wrapped__") else None
    assert appl is not None
    slug = _requests(appl, [("POST", "/api/profiles/draft", {"json": {"name": "T"}})])[0].json()["slug"]
    conn = connect(tmp_path / "radar.db")
    pid = int(conn.execute("SELECT id FROM profiles WHERE slug = ?", (slug,)).fetchone()[0])
    run = gather_runs_repo.start(conn, profile_id=pid, user_id=1, tier_used="dry_run")
    conn.execute("UPDATE gather_runs SET progress_updated_at = '2026-10-16 01:02:03' WHERE id = ?", (run,))
    conn.commit()
    conn.close()
    body = _requests(appl, [("GET", f"/api/profiles/draft/{slug}/dry-run/{run}", {})])[0].json()
    assert body["run"]["progress_updated_at"] == "2026-10-16 01:02:03"
