"""FUNCTIONAL_SPEC CR.4 (review M3, repro V-02): the "deleted" mark on an
interest's seed list after the same PDF is uploaded again.

No rule "a current seed is never shown deleted" (the user: it would
mislead). Each upload is its own paper (CR.10), so the new upload has a new
id: the old one stays on the list, struck through and marked deleted, until
a successful scan has scored without it; the new one is an ordinary seed.
"""

from __future__ import annotations

import io

from rag_lib.db import connect
from rag_lib.db.repos import gather_runs as gather_runs_repo
from tests.edit_helpers import edit_saved
from tests.test_researchers import _request, env  # noqa: F401  (fixture)
from tests.test_review_round12_isolation import A, world  # noqa: F401  (fixture)

SAME_BYTES = b"%PDF-1.4\nalice-bytes"  # the world fixture's upload


def _q(env, sql, *a):
    c = connect(env)
    try:
        return [tuple(r) for r in c.execute(sql, a).fetchall()]
    finally:
        c.close()


def _same_width(env, profile_id):
    """The world fixture writes its three seed vectors by hand (64 wide)
    while uploads are embedded by the test embedder (128 wide); the detail
    page needs one width, as a real interest has (one model). Rewritten to
    the uploads' width, nothing else changed."""
    import numpy as np

    c = connect(env)
    try:
        width = c.execute("SELECT length(vector) FROM paper_embeddings WHERE openalex_id LIKE 'local:upload:%'").fetchone()[0]
        rng = np.random.default_rng(11)
        for (oa,) in c.execute("SELECT openalex_id FROM profile_seeds WHERE profile_id = ? AND openalex_id NOT LIKE 'local:%'",
                               (profile_id,)).fetchall():
            v = rng.standard_normal(width // 4).astype("float32")
            v /= np.linalg.norm(v)
            c.execute("UPDATE paper_embeddings SET vector = ? WHERE openalex_id = ?", (v.tobytes(), oa))
        c.commit()
    finally:
        c.close()


def _detail_json(app, slug):
    r = _request(app, "GET", f"/api/profiles/{slug}/detail", headers=A)
    assert r.status_code == 200, r.text
    return r.json()


def _detail(app, slug):
    d = _detail_json(app, slug)
    return [s["id"] for s in d["seeds"]], [s["id"] for s in d["deletedSeeds"]]


def _scan(env, profile_id, *, ok=True):
    """A scan as gather_for_profile closes one: a clean one passes the seeds
    it scored with; a failed one passes none (CR.5)."""
    c = connect(env)
    try:
        uid = c.execute("SELECT user_id FROM profiles WHERE id = ?", (profile_id,)).fetchone()[0]
        seeds = [r[0] for r in c.execute("SELECT openalex_id FROM profile_seeds WHERE profile_id = ?", (profile_id,))]
        # The delete is strictly older than this scan's start.
        c.execute("UPDATE deleted_seeds SET deleted_at = datetime('now', '-1 minute')")
        c.commit()
        g = gather_runs_repo.start(c, profile_id=profile_id, user_id=uid, tier_used="manual")
        if ok:
            gather_runs_repo.finish(c, g, n_fetched=0, n_new=0, n_redup=0, seed_ids=seeds)
        else:
            gather_runs_repo.finish(c, g, n_fetched=0, n_new=0, n_redup=0, error="OpenAlex refused. Nothing was changed.")
    finally:
        c.close()


def _reupload(app, live):
    files = {"file": ("alpha.pdf", io.BytesIO(SAME_BYTES), "application/pdf")}
    with edit_saved(lambda m, p, **kw: _request(app, m, p, headers=A, **kw), live) as copy:
        r = _request(app, "POST", "/api/vault/upload", files=files, data={"profile_slug": copy}, headers=A)
        assert r.status_code == 200, r.text
        return r.json()["id"]


def test_cr4_the_same_pdf_again_is_a_new_seed_and_the_old_mark_goes_after_a_clean_scan(world):
    app, env, live, live_id, old = world["app"], world["env"], world["live"], world["live_id"], world["upload"]
    _same_width(env, live_id)
    # The first file an hour old, so the two upload times differ.
    c = connect(env)
    c.execute("UPDATE paper_uploads SET uploaded_at = datetime('now', '-1 hour') WHERE openalex_id = ?", (old,))
    c.commit()
    old_at = c.execute("SELECT uploaded_at FROM paper_uploads WHERE openalex_id = ?", (old,)).fetchone()[0]
    c.close()
    before = {s["id"]: s for s in _detail_json(app, live)["seeds"]}
    assert before[old]["uploaded"] == old_at, "an upload seed shows its upload time, numbered or not"
    assert before[old]["version"] is None

    r = _request(app, "DELETE", f"/api/vault/docs/{old}", headers=A)
    assert r.status_code == 200, r.text
    seeds, deleted = _detail(app, live)
    assert old not in seeds and deleted == [old]

    new = _reupload(app, live)
    assert new != old
    seeds, deleted = _detail(app, live)
    assert new in seeds and old not in seeds
    assert deleted == [old], "the mark names the old upload, not the new one"
    # Option A (user 11-03): both carry their upload time, no numbers.
    d = _detail_json(app, live)
    mark = d["deletedSeeds"][0]
    fresh = next(s for s in d["seeds"] if s["id"] == new)
    assert mark["uploaded"] == old_at and mark["deletedAt"]
    assert fresh["uploaded"] and fresh["uploaded"] > old_at and fresh["version"] is None
    others = [s for s in d["seeds"] if not s["id"].startswith("local:upload:")]
    assert others and all(s["uploaded"] is None for s in others), "only uploads have an upload time"

    _scan(env, live_id, ok=False)
    assert _detail(app, live)[1] == [old], "a failed scan leaves the mark"

    _scan(env, live_id)
    seeds, deleted = _detail(app, live)
    assert deleted == [] and new in seeds
    assert _q(env, "SELECT openalex_id FROM deleted_seeds WHERE profile_id = ?", live_id) == []


def test_cr4_a_mark_from_before_the_scan_began_only(world):
    """A PDF deleted while a scan is running is still in that scan's
    scores: the mark stays until the next one."""
    app, env, live, live_id, old = world["app"], world["env"], world["live"], world["live_id"], world["upload"]
    _same_width(env, live_id)
    c = connect(env)
    try:
        uid = c.execute("SELECT user_id FROM profiles WHERE id = ?", (live_id,)).fetchone()[0]
        c.execute("UPDATE gather_runs SET finished_at = datetime('now') WHERE finished_at IS NULL")
        c.commit()
        g = gather_runs_repo.start(c, profile_id=live_id, user_id=uid, tier_used="manual")
        c.execute("UPDATE gather_runs SET started_at = datetime('now', '-1 minute') WHERE id = ?", (g,))
        c.commit()
    finally:
        c.close()
    # The delete itself refuses while a scan runs; the mark is written as
    # the delete does, after the scan began.
    c = connect(env)
    try:
        c.execute("INSERT INTO deleted_seeds (profile_id, openalex_id, title) VALUES (?, ?, 't')", (live_id, old))
        c.execute("DELETE FROM profile_seeds WHERE profile_id = ? AND openalex_id = ?", (live_id, old))
        c.commit()
        seeds = [r[0] for r in c.execute("SELECT openalex_id FROM profile_seeds WHERE profile_id = ?", (live_id,))]
        gather_runs_repo.finish(c, g, n_fetched=0, n_new=0, n_redup=0, seed_ids=seeds)
    finally:
        c.close()
    assert _detail(app, live)[1] == [old]
