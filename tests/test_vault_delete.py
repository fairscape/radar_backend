"""Deleting an uploaded PDF from the Vault (FUNCTIONAL_SPEC F1)."""

from __future__ import annotations

import io

from rag_lib.db import connect
from rag_lib.db.repos import gather_runs as gather_runs_repo, users as users_repo
from tests.test_researchers import _request, env  # noqa: F401 -- fixture
from tests.test_review_round12_isolation import A, B, world  # noqa: F401 -- fixture


def _ok(r):
    assert r.status_code == 200, r.text
    return r.json()


def _q(env, sql, *args):
    conn = connect(env)
    try:
        return [tuple(r) for r in conn.execute(sql, args).fetchall()]
    finally:
        conn.close()


def _vault_ids(app, headers=A):
    return {d["id"] for d in _ok(_request(app, "GET", "/api/vault/docs", headers=headers))}


def _detail(app, slug):
    return _ok(_request(app, "GET", f"/api/profiles/{slug}/detail", headers=A))


def test_delete_takes_file_text_and_vectors_and_marks_the_seed(world):
    app, env, live, up = world["app"], world["env"], world["live"], world["upload"]
    path = _q(env, "SELECT local_path FROM paper_uploads WHERE openalex_id = ?", up)[0][0]
    # Fitted on its seeds, as a saved interest is.
    import json
    from rag_lib.api.services.wizard import FIT_SEEDS
    conn = connect(env)
    try:
        ids = sorted(r[0] for r in conn.execute(
            "SELECT ps.openalex_id FROM profile_seeds ps JOIN profiles p ON p.id = ps.profile_id WHERE p.slug = ?", (live,)))
        conn.execute("UPDATE profiles SET selector_config_json = ? WHERE slug = ?", (json.dumps({FIT_SEEDS: ids}), live))
        conn.commit()
    finally:
        conn.close()
    assert up in _vault_ids(app)
    # (Read from the rows: in this world the upload's vector and the other
    # seeds' differ in length, which the detail page cannot compare.)
    assert (up,) in _q(env, "SELECT ps.openalex_id FROM profile_seeds ps JOIN profiles p ON p.id = ps.profile_id"
                            " WHERE p.slug = ?", live)

    r = _ok(_request(app, "DELETE", f"/api/vault/docs/{up}", headers=A))
    assert r["interests"] == ["A live"]

    assert up not in _vault_ids(app)
    from pathlib import Path
    assert not Path(path).exists()
    assert _q(env, "SELECT 1 FROM paper_embeddings WHERE openalex_id = ?", up) == []
    assert _q(env, "SELECT 1 FROM paper_uploads WHERE openalex_id = ?", up) == []
    assert _q(env, "SELECT body_text FROM papers WHERE openalex_id = ? AND body_text IS NOT NULL", up) == []
    d = _detail(app, live)
    assert up not in {s["id"] for s in d["seeds"]}
    assert [x["id"] for x in d["deletedSeeds"]] == [up]
    assert d["deletedSeeds"][0]["title"]
    # The next scan refits it, on the seeds without the PDF.
    from rag_lib.api.services import wizard as wizard_service
    from rag_lib.scheduler import jobs

    pid, uid = _q(env, "SELECT id, user_id FROM profiles WHERE slug = ?", live)[0]
    refits = []
    real = wizard_service.refit_profile
    wizard_service.refit_profile = lambda conn, **kw: refits.append(kw["slug"])
    try:
        conn = connect(env)
        try:
            jobs._refit_if_fit_stale(conn, uid, pid, None, when="start")
        finally:
            conn.close()
    finally:
        wizard_service.refit_profile = real
    assert refits == [live]


def _scan(env, profile_id, user_id, seed_ids, error=None):
    conn = connect(env)
    try:
        g = gather_runs_repo.start(conn, profile_id=profile_id, user_id=user_id, tier_used="manual")
        gather_runs_repo.finish(conn, g, n_fetched=0, n_new=0, n_redup=0, error=error,
                                seed_ids=None if error else seed_ids)
    finally:
        conn.close()


def test_the_mark_goes_once_a_scan_has_scored_without_it(world):
    app, env, live, up = world["app"], world["env"], world["live"], world["upload"]
    _ok(_request(app, "DELETE", f"/api/vault/docs/{up}", headers=A))
    pid, uid = _q(env, "SELECT id, user_id FROM profiles WHERE slug = ?", live)[0]
    seeds = [s for (s,) in _q(env, "SELECT openalex_id FROM profile_seeds WHERE profile_id = ?", pid)]
    # A failed scan leaves it.
    _scan(env, pid, uid, seeds, error="OpenAlex unavailable")
    assert [x["id"] for x in _detail(app, live)["deletedSeeds"]] == [up]
    # A scan that started before the delete leaves it.
    conn = connect(env)
    try:
        g = gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="manual")
        conn.execute("UPDATE gather_runs SET started_at = datetime('now', '-1 hour') WHERE id = ?", (g,))
        conn.commit()
        gather_runs_repo.finish(conn, g, n_fetched=0, n_new=0, n_redup=0, seed_ids=seeds)
    finally:
        conn.close()
    assert [x["id"] for x in _detail(app, live)["deletedSeeds"]] == [up]
    conn = connect(env)
    try:
        conn.execute("UPDATE deleted_seeds SET deleted_at = datetime('now', '-1 minute')")
        conn.commit()
    finally:
        conn.close()
    _scan(env, pid, uid, seeds)
    assert _detail(app, live)["deletedSeeds"] == []


def test_only_ones_own_upload_and_never_an_interest_s_last_seed(world):
    app, env, live, up = world["app"], world["env"], world["live"], world["upload"]
    assert _request(app, "DELETE", f"/api/vault/docs/{up}", headers=B).status_code == 404
    assert _request(app, "DELETE", f"/api/vault/docs/{world['cards'][0]}", headers=A).status_code == 404
    pid = _q(env, "SELECT id FROM profiles WHERE slug = ?", live)[0][0]
    conn = connect(env)
    try:
        conn.execute("DELETE FROM profile_seeds WHERE profile_id = ? AND openalex_id != ?", (pid, up))
        conn.commit()
    finally:
        conn.close()
    r = _request(app, "DELETE", f"/api/vault/docs/{up}", headers=A)
    assert r.status_code == 409 and "only seed" in r.json()["detail"]
    assert up in _vault_ids(app)


def test_refused_while_a_scan_of_an_interest_it_seeds_runs(world):
    app, env, live, up = world["app"], world["env"], world["live"], world["upload"]
    pid, uid = _q(env, "SELECT id, user_id FROM profiles WHERE slug = ?", live)[0]
    conn = connect(env)
    try:
        gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="manual")
    finally:
        conn.close()
    r = _request(app, "DELETE", f"/api/vault/docs/{up}", headers=A)
    assert r.status_code == 409, r.text
    assert up in _vault_ids(app)


def test_another_user_s_upload_of_the_same_file_stays(world):
    app, env, up = world["app"], world["env"], world["upload"]
    b_draft = _ok(_request(app, "POST", "/api/profiles/draft", headers=B, json={"name": "B draft"}))["slug"]
    files = {"file": ("alpha.pdf", io.BytesIO(b"%PDF-1.4\nalice-bytes"), "application/pdf")}
    b_up = _ok(_request(app, "POST", "/api/vault/upload", headers=B, files=files, data={"profile_slug": b_draft}))["id"]
    assert b_up != up
    _ok(_request(app, "DELETE", f"/api/vault/docs/{up}", headers=A))
    assert b_up in _vault_ids(app, headers=B)
    conn = connect(env)
    try:
        b_uid = int(users_repo.get_by_email(conn, "bob@example.com")["id"])
        assert conn.execute("SELECT 1 FROM paper_uploads WHERE user_id = ? AND openalex_id = ?", (b_uid, b_up)).fetchone()
    finally:
        conn.close()
