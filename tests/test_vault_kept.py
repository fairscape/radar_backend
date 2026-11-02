"""A paper leaves the vault only through the Vault's own delete
(FUNCTIONAL_SPEC 7, B2.2): deleting an interest or a researcher, or
removing a seed in Edit interest, leaves its papers in the vault. A draft
is a sandbox (D): what an abandoned draft brought in still goes."""

from __future__ import annotations

import re
from pathlib import Path

from rag_lib.db import connect
from tests.test_researchers import _request, env  # noqa: F401 -- fixture
from tests.test_review_round12_isolation import A, B, world  # noqa: F401 -- fixture


def _vault(app, headers=A, tag=None):
    path = "/api/vault/docs" + (f"?tag={tag}" if tag else "")
    r = _request(app, "GET", path, headers=headers)
    assert r.status_code == 200, r.text
    return {d["canonicalId"] or d["id"]: d for d in r.json()}


def _ok(r):
    assert r.status_code == 200, r.text
    return r.json()


def _seeds(env, slug):
    conn = connect(env)
    try:
        return [r[0] for r in conn.execute(
            "SELECT ps.openalex_id FROM profile_seeds ps JOIN profiles p ON p.id = ps.profile_id WHERE p.slug = ?",
            (slug,))]
    finally:
        conn.close()


def test_deleting_an_interest_leaves_its_papers_in_the_vault(world):
    app, live = world["app"], world["live"]
    saved = world["cards"][0]
    _ok(_request(app, "POST", "/api/radar/cards/save", headers=A,
                 json={"card_id": saved, "profile": live, "on": True}))
    seeds = _seeds(world["env"], live)
    before = _vault(app)
    assert set(seeds) | {saved} <= set(before)
    docs_before = _ok(_request(app, "GET", "/api/vault/stats", headers=A))["docs"]

    _ok(_request(app, "DELETE", f"/api/profiles/{live}", headers=A))

    after = _vault(app)
    assert set(after) == set(before)
    for oa in set(seeds) | {saved}:
        assert live not in after[oa]["tags"]
    assert _ok(_request(app, "GET", "/api/vault/stats", headers=A))["docs"] == docs_before
    assert live not in _ok(_request(app, "GET", "/api/vault/tags", headers=A))
    assert _vault(app, headers=B) == {}


def test_forgetting_a_researcher_leaves_their_papers_in_the_vault(world):
    app = world["app"]
    papers = {p["id"] for p in _ok(_request(app, "GET", f"/api/researchers/{world['rid']}", headers=A))["papers"]}
    # The draft built from them would keep the seeds in the vault by itself.
    _ok(_request(app, "DELETE", f"/api/profiles/draft/{world['draft']}", headers=A))
    before = _vault(app)
    assert papers <= set(before)

    _ok(_request(app, "DELETE", f"/api/researchers/{world['rid']}", headers=A))

    assert set(_vault(app)) == set(before)
    assert _vault(app, headers=B) == {}


def test_a_seed_removed_in_edit_interest_stays_in_the_vault(world):
    app, live = world["app"], world["live"]
    seed = next(s for s in _seeds(world["env"], live) if s != world["upload"])
    before = _vault(app)
    copy = _ok(_request(app, "POST", f"/api/profiles/{live}/edit", headers=A))["slug"]
    _ok(_request(app, "DELETE", f"/api/profiles/{copy}/seeds", headers=A, params={"openalex_id": seed}))
    _ok(_request(app, "POST", f"/api/profiles/draft/{copy}/save-edit", headers=A))

    assert seed not in _seeds(world["env"], live)
    after = _vault(app)
    assert set(after) == set(before)
    assert live not in after[seed]["tags"]
    assert seed not in _vault(app, tag=live)


def test_an_abandoned_draft_or_edit_keeps_nothing(world):
    app, live, plain = world["app"], world["live"], world["plain"]
    card = world["cards"][1]   # in the feed, never saved
    kept = _kept(world["env"])
    _ok(_request(app, "POST", f"/api/profiles/draft/{plain}/seeds", headers=A, json={"openalex_ids": [card]}))
    _ok(_request(app, "DELETE", f"/api/profiles/draft/{plain}", headers=A))
    assert card not in _vault(app)

    copy = _ok(_request(app, "POST", f"/api/profiles/{live}/edit", headers=A))["slug"]
    _ok(_request(app, "POST", f"/api/profiles/{copy}/seeds", headers=A, json={"openalex_ids": [card]}))
    _ok(_request(app, "DELETE", f"/api/profiles/draft/{copy}", headers=A))
    assert card not in _vault(app)

    assert _kept(world["env"]) == kept


def _kept(env):
    conn = connect(env)
    try:
        return sorted(tuple(r) for r in conn.execute("SELECT user_id, openalex_id FROM vault_kept"))
    finally:
        conn.close()


# Every statement that deletes what puts a paper in the vault, by file, with
# why it may: a user's delete keeps first (vault_kept), the rest only undo
# a draft, a copy or a failed request. A new one fails here until it is
# looked at and listed.
_VAULT_SOURCES = r"DELETE FROM (profiles|researchers|profile_seeds|researcher_papers|profile_candidates|paper_uploads)\b|(profiles_repo|researchers_repo)\.delete\(|detach_seed\("
_ALLOWED = {
    "rag_lib/api/services/profiles.py": (1, "delete_profile: keep_interest first"),
    "rag_lib/api/services/researchers.py": (2, "delete_researcher: keep_researcher first; a draft made in a failed request"),
    "rag_lib/api/services/edit_copy.py": (2, "save_copy: keep_interest first; then the copy itself"),
    "rag_lib/api/services/prosopia.py": (4, "a failed import taking back its own writes"),
    "rag_lib/api/services/wizard.py": (1, "a seed removed from a draft or a copy"),
    "rag_lib/db/repos/draft_owned.py": (2, "discarding a draft and what it brought in"),
    "rag_lib/db/repos/profiles.py": (3, "the repo functions themselves (delete, detach_seed)"),
    "rag_lib/db/repos/researchers.py": (1, "the repo function itself"),
    "rag_lib/scheduler/jobs.py": (1, "a draft whose first scan failed"),
    "rag_lib/api/services/vault.py": (2, "delete_upload: the Vault's own delete (F1)"),
}


def test_every_vault_source_delete_is_accounted_for():
    root = Path(__file__).resolve().parent.parent
    found = {}
    for path in sorted((root / "rag_lib").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if "/migrations/" in rel:
            continue
        n = len(re.findall(_VAULT_SOURCES, path.read_text()))
        if n:
            found[rel] = n
    assert found == {k: n for k, (n, _) in _ALLOWED.items()}
