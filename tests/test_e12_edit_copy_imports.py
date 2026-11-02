"""ORCID, Prosopia and saved researcher profiles as seeds in Edit interest
(FUNCTIONAL_SPEC E.12).

The import goes into the copy Edit interest made (``into``), as it goes
into a new interest's draft: the interest is unchanged until Save
changes; discarding takes what the import brought in, but not a
researcher profile the user ticked to save (M.2).
"""

from __future__ import annotations

import pytest

from rag_lib.api.services import wizard
from rag_lib.db import connect
from rag_lib.embedders import placeholder_embed
from rag_lib.db.repos import embeddings as embeddings_repo
from rag_lib.db.repos import gather_runs as gather_runs_repo
from rag_lib.db.repos import profiles as profiles_repo
from rag_lib.researcher_topics import code as _c
from tests.test_orcid_import import ORCID, WORKS, StubOrcidOpenAlex
from tests.test_researchers import _app, _import, _request, env  # noqa: F401 -- fixture
from tests.test_seed_removal_topics import _paper

OWN = "https://openalex.org/W9001"
OWN_TOPIC = "https://openalex.org/T10009"


def _q(env, sql, *params):
    conn = connect(env)
    try:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


@pytest.fixture()
def saved(env):
    """demo's saved interest with one seed of its own, and the app with the
    ORCID stub (its works carry topic T10001)."""
    conn = connect(env)
    pid = profiles_repo.upsert(conn, user_id=1, name="Live", embedding_model="placeholder-v1",
                               n_seed=1, topic_filters={"topics": []}, threshold=0.5)
    conn.execute("UPDATE profiles SET is_draft = 0 WHERE id = ?", (pid,))
    _paper(conn, OWN, (OWN_TOPIC, "Nine"))
    profiles_repo.attach_seed(conn, pid, OWN)
    embeddings_repo.upsert(conn, OWN, "placeholder-v1", placeholder_embed(OWN))
    slug = profiles_repo.get(conn, pid)["slug"]
    wizard.aggregate_draft_topics(conn, user_id=1, slug=slug)
    conn.commit()
    conn.close()
    app = _app(None, StubOrcidOpenAlex(WORKS))
    copy = _request(app, "POST", f"/api/profiles/{slug}/edit").json()["slug"]
    return app, slug, copy


def _seeds(env, slug):
    return sorted(r[0].rsplit("/", 1)[-1] for r in _q(
        env, "SELECT s.openalex_id FROM profile_seeds s JOIN profiles p ON p.id = s.profile_id WHERE p.slug = ?", slug))


def _orcid_into(app, copy, ids, **body):
    r = _request(app, "POST", "/api/import/orcid", json={"orcid": ORCID, "openalex_ids": ids, "into": copy, **body})
    assert r.status_code == 200, r.text
    return _request(app, "GET", f"/api/import/prosopia/{r.json()['run_id']}").json()["result"]


def test_x41_an_import_lands_on_the_copy_only(env, saved):
    app, slug, copy = saved
    result = _orcid_into(app, copy, ["W1", "W2"])
    assert result["draft_slug"] == copy
    assert (result["drafted"], result["seeded"]) == (2, 2)
    assert _seeds(env, copy) == ["W1", "W2", "W9001"]
    assert _seeds(env, slug) == ["W9001"]
    assert [s["id"].rsplit("/", 1)[-1] for s in _request(app, "GET", f"/api/profiles/{slug}/detail").json()["seeds"]] == ["W9001"]
    assert _request(app, "GET", f"/api/profiles/draft/{copy}/edit-state").json() == {"changed": True}
    # Not a draft of the wizard's, nor an interest of the list.
    assert copy not in [d["slug"] for d in _request(app, "GET", "/api/profiles/drafts").json()]
    assert copy not in [p["key"] for p in _request(app, "GET", "/api/profiles").json()]


def test_x43_importing_the_same_person_again_adds_only_the_new_papers(env, saved):
    app, _, copy = saved
    _orcid_into(app, copy, ["W1"])
    result = _orcid_into(app, copy, ["W1", "W3"])
    assert (result["drafted"], result["seeded"]) == (2, 1)
    assert _seeds(env, copy) == ["W1", "W3", "W9001"]


def test_x42_the_copy_is_neither_saved_nor_discarded_while_an_import_fills_it(env, saved):
    app, slug, copy = saved
    conn = connect(env)
    cid = int(conn.execute("SELECT id FROM profiles WHERE slug = ?", (copy,)).fetchone()[0])
    run = gather_runs_repo.start(conn, profile_id=cid, user_id=1, since_date=None, filter_string=None,
                                 tier_used="orcid_import")
    conn.commit()
    conn.close()
    assert _request(app, "POST", f"/api/profiles/draft/{copy}/save-edit").status_code == 409
    _request(app, "DELETE", f"/api/profiles/draft/{copy}")
    assert _q(env, "SELECT COUNT(*) FROM profiles WHERE slug = ?", copy) == [(1,)]
    conn = connect(env)
    gather_runs_repo.finish(conn, run, n_fetched=0, n_new=0, n_redup=0, api_calls=0)
    conn.commit()
    conn.close()
    assert _request(app, "POST", f"/api/profiles/draft/{copy}/save-edit").status_code == 200


def test_x44_a_saved_researchers_papers_become_seeds_of_the_copy(env, saved):
    app, slug, copy = saved
    rid = _import(app, source="orcid", ref=ORCID).json()["researcher_id"]
    r = _request(app, "POST", f"/api/profiles/{copy}/seeds",
                 json={"openalex_ids": ["https://openalex.org/W2"], "source": "researcher", "researcher_id": rid})
    assert r.status_code == 200, r.text
    assert r.json()["attached"] == 1
    assert _seeds(env, copy) == ["W2", "W9001"]
    assert _seeds(env, slug) == ["W9001"]


def test_x45_discarding_takes_the_unticked_imports_papers(env, saved):
    app, slug, copy = saved
    _orcid_into(app, copy, ["W1", "W2"])
    assert _q(env, "SELECT COUNT(*) FROM papers")[0][0] == 3
    assert _request(app, "DELETE", f"/api/profiles/draft/{copy}").status_code in (200, 204)
    assert _seeds(env, slug) == ["W9001"]
    assert _q(env, "SELECT openalex_id FROM papers") == [(OWN,)]
    assert _q(env, "SELECT COUNT(*) FROM profiles WHERE edit_of IS NOT NULL") == [(0,)]


def test_x45_a_ticked_researcher_profile_stays_when_the_edit_is_discarded(env, saved):
    app, slug, copy = saved
    _orcid_into(app, copy, ["W1", "W2"], save_profile=True)
    assert _q(env, "SELECT COUNT(*) FROM researchers") == [(1,)]
    assert _request(app, "DELETE", f"/api/profiles/draft/{copy}").status_code in (200, 204)
    assert _seeds(env, slug) == ["W9001"]
    assert _q(env, "SELECT COUNT(*) FROM researchers") == [(1,)]
    assert [x["n_papers"] for x in _request(app, "GET", "/api/researchers").json()] == [2]


def test_x46_save_keeps_the_seeds_and_their_new_topics_are_off_and_new(env, saved):
    app, slug, copy = saved
    _orcid_into(app, copy, ["W1", "W2"])
    topics = {_c(t["id"]): t for t in _request(app, "GET", f"/api/profiles/{copy}/topics/preview").json()}
    assert topics["T10001"]["on"] is False
    assert topics["T10001"].get("new") is True
    r = _request(app, "POST", f"/api/profiles/draft/{copy}/save-edit")
    assert r.status_code == 200, r.text
    assert _seeds(env, slug) == ["W1", "W2", "W9001"]
    detail = _request(app, "GET", f"/api/profiles/{slug}/detail").json()
    assert "seeds" in detail, detail
    assert {s["id"].rsplit("/", 1)[-1] for s in detail["seeds"]} == {"W1", "W2", "W9001"}
