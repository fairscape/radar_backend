"""Round 14 (reviewers b and c, independently): an interest named "ALL"
(acute lymphoblastic leukemia) got the slug "all", which the feed's
interest filter, the Vault's tag filter and its per-tag counts read as
"every interest". It is now reserved like "new" and suffixed."""

from __future__ import annotations

import importlib.util
import pathlib

from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import profiles as profiles_repo, users as users_repo
from tests.test_review_round12_journeys import env, app, _request, _ok, _upload  # noqa: F401


def test_an_interest_named_all_lists_only_its_own_seeds(app):
    _ok(_upload(app, b"unrelated-one", "u1.pdf"))
    _ok(_upload(app, b"unrelated-two", "u2.pdf"))
    d = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": "ALL"}))
    assert d["slug"] == "all-profile"
    _ok(_upload(app, b"the-real-seed", "seed.pdf", d["slug"]))
    docs = _ok(_request(app, "GET", "/api/vault/docs", params={"tag": d["slug"]}))
    assert len(docs) == 1                       # its seed, not the whole Vault
    tags = _ok(_request(app, "GET", "/api/vault/tags"))
    assert tags.get("all-profile") == 1 and tags.get("all") == 3


def test_0024_renames_stored_all_slugs_per_user(tmp_path):
    conn = connect(tmp_path / "r.db")
    apply_migrations(conn)
    a = int(users_repo.upsert(conn, "a@example.com")["id"])
    b = int(users_repo.upsert(conn, "b@example.com")["id"])
    # As stored before the slug was reserved; user a already has an all-profile.
    for uid, name, slug in ((a, "ALL", "all"), (a, "All profile", "all-profile"), (b, "All", "all")):
        conn.execute("INSERT INTO profiles (user_id, name, slug, embedding_model, topic_filters_json, n_seed) VALUES (?, ?, ?, 'm', '{}', 0)", (uid, name, slug))
    conn.commit()
    path = pathlib.Path(profiles_repo.__file__).parents[1] / "migrations" / "0024_reserve_all_slug.py"
    spec = importlib.util.spec_from_file_location("m0024", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.migrate(conn)
    conn.commit()
    got = {(r[0], r[1]) for r in conn.execute("SELECT user_id, slug FROM profiles")}
    assert got == {(a, "all-profile-2"), (a, "all-profile"), (b, "all-profile")}
    assert profiles_repo.slugify("ALL") == "all-profile"
    conn.close()


def test_a_researcher_id_too_large_for_sqlite_is_a_404(app):
    r = _request(app, "GET", f"/api/researchers/{2**64}/works")
    assert r.status_code == 404, r.text
