"""A draft is a sandbox (FUNCTIONAL_SPEC D): what it brought into the
database goes when it is discarded, unless something else uses it."""

from __future__ import annotations

from pathlib import Path

from rag_lib.db import connect
from tests.test_private_upload_versions import W, _post, _upload, pdfs  # noqa: F401 -- fixture
from tests.test_researchers import env  # noqa: F401 -- fixture
from tests.test_review_round12_isolation import A, B, _request, world  # noqa: F401 -- fixture

TABLES = ("papers", "paper_embeddings", "paper_uploads", "paper_versions", "profile_seeds",
          "researchers", "researcher_papers", "draft_owned")


def _draft(world, name, headers=A) -> str:
    r = _request(world["app"], "POST", "/api/profiles/draft", json={"name": name}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()["slug"]


def _discard(world, slug, headers=A):
    r = _request(world["app"], "DELETE", f"/api/profiles/draft/{slug}", headers=headers)
    assert r.status_code == 200, r.text


def _q(world, sql, *a):
    conn = connect(world["env"])
    try:
        return [tuple(r) for r in conn.execute(sql, a).fetchall()]
    finally:
        conn.close()


def _snapshot(world):
    """Every row the sandbox could touch, and the files in the vault."""
    rows = {t: sorted(map(repr, _q(world, f"SELECT * FROM {t}"))) for t in TABLES}
    rows["files"] = sorted(str(p) for p in (Path(world["env"]).parent / "vault").rglob("*.pdf"))
    return rows


def _save(world, slug, headers=A):
    """The interest saved (the commit route's flip and clear; its first
    scan is the commit's own test)."""
    from rag_lib.db.repos import draft_owned

    conn = connect(world["env"])
    try:
        (pid,) = conn.execute("SELECT p.id FROM profiles p JOIN users u ON u.id = p.user_id "
                              "WHERE p.slug = ? AND u.email = ?", (slug, headers["X-User-Email"])).fetchone()
        conn.execute("UPDATE profiles SET is_draft = 0 WHERE id = ?", (pid,))
        draft_owned.clear(conn, pid)
        conn.commit()
    finally:
        conn.close()


def _vault(world, headers=A):
    r = _request(world["app"], "GET", "/api/vault/docs", headers=headers)
    assert r.status_code == 200, r.text
    return {d["id"]: d for d in r.json()}


def test_s1_discarding_a_draft_removes_its_uploads(pdfs):
    before, vault = _snapshot(pdfs), set(_vault(pdfs))
    d = _draft(pdfs, "S1")
    p = _upload(pdfs, b"matched P", A, d)
    m = _upload(pdfs, b"manuscript M", A, d)
    files = [r[0] for r in _q(pdfs, "SELECT local_path FROM paper_uploads WHERE openalex_id IN (?, ?)",
                              p["id"], m["id"])]
    assert len(files) == 2 and all(Path(f).is_file() for f in files)
    assert set(_vault(pdfs)) == vault | {p["id"], m["id"]}

    _discard(pdfs, d)
    # The versions, their vectors, upload rows and files, and W77's public
    # row, which this upload created and nothing else uses.
    assert _snapshot(pdfs) == before
    assert not any(Path(f).exists() for f in files)
    assert set(_vault(pdfs)) == vault


def test_s2_a_duplicate_upload_into_a_draft_is_not_the_drafts(pdfs):
    x = _draft(pdfs, "X")
    p = _upload(pdfs, b"matched P", A, x)
    _save(pdfs, x)
    before = _snapshot(pdfs)

    d = _draft(pdfs, "S2")
    again = _upload(pdfs, b"matched P", A, d)
    assert again["duplicate"] and again["id"] == p["id"]
    _discard(pdfs, d)
    assert _snapshot(pdfs) == before
    assert _q(pdfs, "SELECT s.openalex_id FROM profile_seeds s JOIN profiles p ON p.id = s.profile_id "
                    "WHERE p.slug = 'x'") == [(p["id"],)]


def test_s3_only_the_version_uploaded_in_the_draft_goes(pdfs):
    x = _draft(pdfs, "X")
    v1 = _upload(pdfs, b"matched P", A, x)
    _save(pdfs, x)
    before = _snapshot(pdfs)

    d = _draft(pdfs, "S3")
    v2 = _upload(pdfs, b"matched P revised", A, d)
    assert v2["canonicalId"] == W and v2["version"] == 2
    assert _vault(pdfs)[v1["id"]]["version"] == 1
    _discard(pdfs, d)
    assert _snapshot(pdfs) == before          # W77's public row stays: v1 is a copy of it
    assert _vault(pdfs)[v1["id"]]["version"] is None


def test_s4_an_upload_another_interest_uses_stays(pdfs):
    y = _draft(pdfs, "Y")
    _save(pdfs, y)
    d = _draft(pdfs, "S4")
    n = _upload(pdfs, b"note N", A, d)
    again = _upload(pdfs, b"note N", A, y)
    assert again["duplicate"] and again["id"] == n["id"]
    path = _q(pdfs, "SELECT local_path FROM paper_uploads WHERE openalex_id = ?", n["id"])[0][0]

    _discard(pdfs, d)
    assert n["id"] in _vault(pdfs) and Path(path).is_file()
    assert _q(pdfs, "SELECT 1 FROM paper_embeddings WHERE openalex_id = ?", n["id"])
    # Nothing is the discarded draft's any more, so a later discard of
    # some other draft cannot remove it either.
    assert _q(pdfs, "SELECT * FROM draft_owned") == []


def test_a_public_record_another_user_comes_to_use_stays(pdfs):
    vault_a, vault_b = set(_vault(pdfs)), set(_vault(pdfs, B))
    d = _draft(pdfs, "Mine")
    _upload(pdfs, b"matched P", A, d)
    b = _draft(pdfs, "Theirs", B)
    vb = _upload(pdfs, b"matched P by B", B, b)
    _discard(pdfs, d)
    assert _q(pdfs, "SELECT openalex_id FROM papers WHERE openalex_id = ?", W) == [(W,)]
    assert set(_vault(pdfs, B)) == vault_b | {vb["id"]} and set(_vault(pdfs)) == vault_a


def test_s11_a_draft_left_open_keeps_everything(pdfs):
    d = _draft(pdfs, "S11")
    p = _upload(pdfs, b"matched P", A, d)
    m = _upload(pdfs, b"manuscript M", A, d)
    before = _snapshot(pdfs)
    r = _request(pdfs["app"], "GET", f"/api/profiles/{d}/detail", headers=A)
    assert r.status_code == 200, r.text
    assert {s["id"] for s in r.json()["seeds"]} == {p["id"], m["id"]}
    assert {p["id"], m["id"]} <= set(_vault(pdfs))
    assert _snapshot(pdfs) == before


def test_saving_clears_what_the_draft_brought_in(pdfs):
    d = _draft(pdfs, "Saved")
    p = _upload(pdfs, b"matched P", A, d)
    assert len(_q(pdfs, "SELECT * FROM draft_owned")) == 2       # the version and W77's new public row
    _save(pdfs, d)
    assert _q(pdfs, "SELECT * FROM draft_owned") == []
    assert p["id"] in _vault(pdfs)


def test_an_upload_into_a_saved_interest_is_nobodys(pdfs):
    y = _draft(pdfs, "Y")
    _save(pdfs, y)
    _upload(pdfs, b"matched P", A, y)
    assert _q(pdfs, "SELECT * FROM draft_owned") == []


def test_a_discard_that_fails_keeps_the_draft_and_its_uploads(pdfs, monkeypatch):
    from rag_lib.db.repos import draft_owned

    d = _draft(pdfs, "Kept")
    _upload(pdfs, b"matched P", A, d)
    before = _snapshot(pdfs)

    def boom(conn, oa):
        raise RuntimeError("disk")
    monkeypatch.setattr(draft_owned, "_paper_in_use", boom)
    r = _request(pdfs["app"], "DELETE", f"/api/profiles/draft/{d}", headers=A)
    assert r.status_code == 500
    monkeypatch.undo()
    assert _snapshot(pdfs) == before
    assert _q(pdfs, "SELECT is_draft FROM profiles WHERE slug = ?", d) == [(1,)]


class _Gatherer:
    """A scan that finds ``papers``, or cannot reach OpenAlex (``fail``)."""

    last_tier_used = None

    def __init__(self, papers=(), fail=False):
        self.papers, self.fail = list(papers), fail

    def fetch(self, profile, since, *, limit=None):
        if self.fail:
            from rag_lib.openalex_status import OpenAlexUnavailable
            raise OpenAlexUnavailable("quota", detail="429")
        return list(self.papers)

    def fetch_years(self, profile, a, b, *, per_year):
        return self.fetch(profile, None)

    def cost(self):
        return {"api_calls": 1}


def _found(n=3):
    from rag_lib.paper import Paper

    return [Paper(doi=None, openalex_id=f"https://openalex.org/W55{i}", title=f"Found {i}",
                  abstract="hello world", year=2026, source="openalex_gatherer") for i in range(n)]


def _commit(world, slug, headers=A):
    return _request(world["app"], "POST", "/api/profiles",
                    json={"slug": slug, "threshold": 0.0, "selected_topic_ids": []}, headers=headers)


def _is_draft(world, slug):
    return _q(world, "SELECT is_draft FROM profiles WHERE slug = ?", slug)[0][0]


def test_s9_a_first_scan_that_fails_leaves_a_draft(pdfs, monkeypatch):
    from rag_lib.scheduler import jobs

    d = _draft(pdfs, "S9")
    p = _upload(pdfs, b"matched P", A, d)
    owned = _q(pdfs, "SELECT kind, item FROM draft_owned ORDER BY 1, 2")

    monkeypatch.setattr(jobs, "OpenAlexGatherer", lambda **_: _Gatherer(_found(), fail=True))
    r = _commit(pdfs, d)
    assert r.status_code == 503, r.text
    assert "still a draft" in r.json()["detail"] and "daily request limit" in r.json()["detail"]
    assert _is_draft(pdfs, d) == 1
    assert _q(pdfs, "SELECT kind, item FROM draft_owned ORDER BY 1, 2") == owned
    pid = _q(pdfs, "SELECT id FROM profiles WHERE slug = ?", d)[0][0]
    assert _q(pdfs, "SELECT COUNT(*) FROM profile_candidates WHERE profile_id = ?", pid) == [(0,)]
    assert _q(pdfs, "SELECT COUNT(*) FROM profile_schedules WHERE profile_id = ?", pid) == [(0,)]
    (err,) = _q(pdfs, "SELECT error FROM gather_runs WHERE profile_id = ? AND tier_used != 'dry_run' "
                      "ORDER BY id DESC LIMIT 1", pid)[0]
    assert err.startswith("OpenAlex's daily request limit")

    # Saving again, once OpenAlex answers, goes through.
    monkeypatch.setattr(jobs, "OpenAlexGatherer", lambda **_: _Gatherer(_found()))
    r = _commit(pdfs, d)
    assert r.status_code == 200, r.text
    assert r.json()["isDraft"] is False and r.json()["saveRunId"]
    assert p["id"] in _vault(pdfs)


def test_a_scan_that_fails_after_storing_papers_leaves_none_of_them(pdfs, monkeypatch):
    """Failing after the papers went in (the go-live itself, here): the
    draft keeps nothing of the scan."""
    from rag_lib.api.services import wizard
    from rag_lib.scheduler import jobs

    d = _draft(pdfs, "Late")
    _upload(pdfs, b"matched P", A, d)
    monkeypatch.setattr(jobs, "OpenAlexGatherer", lambda **_: _Gatherer(_found()))

    def boom(*a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(wizard, "go_live", boom)
    r = _commit(pdfs, d)
    assert r.status_code == 502, r.text
    pid = _q(pdfs, "SELECT id FROM profiles WHERE slug = ?", d)[0][0]
    assert _is_draft(pdfs, d) == 1
    assert _q(pdfs, "SELECT COUNT(*) FROM profile_candidates WHERE profile_id = ?", pid) == [(0,)]


def test_s10_a_save_goes_live_with_its_first_scan(pdfs, monkeypatch):
    from rag_lib.scheduler import jobs

    d = _draft(pdfs, "S10")
    p = _upload(pdfs, b"matched P", A, d)
    monkeypatch.setattr(jobs, "OpenAlexGatherer", lambda **_: _Gatherer(_found()))
    r = _commit(pdfs, d)
    assert r.status_code == 200, r.text
    assert r.json()["isDraft"] is False
    pid = _q(pdfs, "SELECT id FROM profiles WHERE slug = ?", d)[0][0]
    run = _q(pdfs, "SELECT finished_at IS NOT NULL, error, n_fetched FROM gather_runs WHERE id = ?",
             r.json()["saveRunId"])[0]
    assert run == (1, None, 3)
    assert _q(pdfs, "SELECT COUNT(*) FROM profile_candidates WHERE profile_id = ?", pid) == [(3,)]
    assert _q(pdfs, "SELECT * FROM draft_owned WHERE profile_id = ?", pid) == []
    assert _q(pdfs, "SELECT COUNT(*) FROM profile_schedules WHERE profile_id = ?", pid) == [(1,)]
    # Deleting the saved interest keeps its papers in the vault (D.4 Q3).
    r = _request(pdfs["app"], "DELETE", f"/api/profiles/{d}", headers=A)
    assert r.status_code in (200, 204), r.text
    assert p["id"] in _vault(pdfs)


def test_a_draft_being_saved_cannot_be_discarded_and_a_second_save_follows_the_first(pdfs):
    from rag_lib.db.repos import gather_runs as gather_runs_repo

    d = _draft(pdfs, "Busy")
    _upload(pdfs, b"matched P", A, d)
    conn = connect(pdfs["env"])
    try:
        pid = conn.execute("SELECT id FROM profiles WHERE slug = ?", (d,)).fetchone()[0]
        uid = conn.execute("SELECT user_id FROM profiles WHERE id = ?", (pid,)).fetchone()[0]
        run = gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="first_scan")
    finally:
        conn.close()
    r = _request(pdfs["app"], "DELETE", f"/api/profiles/draft/{d}", headers=A)
    assert r.status_code == 409 and "being saved" in r.json()["detail"]
    r = _commit(pdfs, d)
    assert r.status_code == 200 and r.json()["saveRunId"] == run and r.json()["isDraft"] is True


def test_an_uploaded_seed_says_so(pdfs):
    d = _draft(pdfs, "Up")
    p = _upload(pdfs, b"matched P", A, d)
    seeds = _request(pdfs["app"], "GET", f"/api/profiles/{d}/detail", headers=A).json()["seeds"]
    assert [(s["id"], s["source"]) for s in seeds] == [(p["id"], "upload")]


def test_while_openalex_is_paused_nothing_that_needs_it_starts(pdfs):
    """FUNCTIONAL_SPEC F2 Q2: saving, the trial, an import are refused at
    once (503), with nothing written; the upload already was."""
    from rag_lib import openalex_status

    d = _draft(pdfs, "Paused")
    _upload(pdfs, b"matched P", A, d)
    before = _snapshot(pdfs)
    runs = _q(pdfs, "SELECT COUNT(*) FROM gather_runs")
    openalex_status.pause("quota")
    for method, path, body in (
        ("POST", "/api/profiles", {"slug": d, "threshold": 0.5, "selected_topic_ids": []}),
        ("POST", f"/api/profiles/draft/{d}/dry-run", {"days": 30}),
        ("POST", "/api/import/prosopia", {"ref": "sheffield-nathan", "name": "Nope"}),
        ("POST", "/api/researchers/import", {"source": "prosopia", "ref": "sheffield-nathan"}),
    ):
        r = _request(pdfs["app"], method, path, json=body, headers=A)
        assert r.status_code == 503, (path, r.text)
        assert r.json()["openalex"]["available"] is False
    assert _snapshot(pdfs) == before and _q(pdfs, "SELECT COUNT(*) FROM gather_runs") == runs
    assert _is_draft(pdfs, d) == 1
