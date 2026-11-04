"""API edges after the 2026-11-02 review (FUNCTIONAL_SPEC CR.11 L11, L12, L16)."""

from __future__ import annotations

import pytest

from tests.test_interest_terms import KEY, SLUG, _request as _terms_request, app  # noqa: F401 -- fixture
from tests.test_researchers import _app, _request, env  # noqa: F401 -- fixture

B = {"X-User-Email": "b@example.org"}


# --- L16: a body that holds a secret is never echoed in a 422 ---

@pytest.mark.parametrize("body", [
    [KEY],                              # the body is not an object: loc ("body",), input = all of it
    {"api_key": [KEY]},                 # the key in the wrong shape (already covered before L16)
    {"api_key": {"nested": KEY}},
    KEY,                                # a bare string body
])
def test_l16_a_422_on_a_route_with_a_secret_echoes_none_of_the_body(app, body):
    resp = _terms_request(app, "POST", f"/api/profiles/{SLUG}/terms/generate", json=body)
    assert resp.status_code == 422, resp.text
    assert KEY not in resp.text
    assert all("input" not in e and "ctx" not in e for e in resp.json()["detail"])


def test_l16_other_routes_still_say_what_was_wrong(env):
    app = _app(None, None)
    r = _request(app, "PATCH", "/api/profiles/whatever", json={"threshold": "high"})
    assert r.status_code == 422
    assert any(e.get("input") == "high" for e in r.json()["detail"])


def test_l16_the_secret_scan_reads_nested_models():
    from pydantic import BaseModel, SecretStr

    from rag_lib.api.app import _holds_secret

    class Inner(BaseModel):
        token: SecretStr

    class Outer(BaseModel):
        name: str
        inner: list[Inner] | None = None

    class Keyed(BaseModel):
        api_key: str | None = None

    class Plain(BaseModel):
        name: str
        n: int | None = None

    assert _holds_secret(Outer) and _holds_secret(Keyed) and _holds_secret(SecretStr | None)
    assert not _holds_secret(Plain) and not _holds_secret(str)


# --- L12: an empty X-User-Email is a bad header, never the default user ---

@pytest.mark.parametrize("value", ["", "   "])
def test_l12_an_empty_identity_header_is_400(env, value):
    app = _app(None, None)
    r = _request(app, "GET", "/api/users/me", headers={"X-User-Email": value})
    assert r.status_code == 400, r.text
    assert r.json()["detail"] == "X-User-Email is empty"


def test_l12_no_header_is_still_the_default_user_when_auth_is_off(env):
    app = _app(None, None)
    assert _request(app, "GET", "/api/users/me").status_code == 200


# --- L11: someone else's interest and no interest read the same ---

def test_l11_another_users_interest_and_a_missing_key_are_both_404(env):
    from rag_lib.db import connect
    from rag_lib.db.repos import profiles as profiles_repo

    conn = connect(env)
    pid = profiles_repo.upsert(conn, user_id=1, name="Mine", embedding_model="placeholder-v1",
                               n_seed=0, topic_filters={"topics": []}, threshold=0.5)
    conn.execute("UPDATE profiles SET is_draft = 0 WHERE id = ?", (pid,))
    slug = profiles_repo.get(conn, pid)["slug"]
    conn.commit()
    conn.close()
    app = _app(None, None)
    assert _request(app, "GET", f"/api/profiles/{slug}").status_code == 200
    assert _request(app, "GET", f"/api/profiles/{slug}/detail").status_code == 200
    seen = []
    for key, headers in ((slug, B), ("no-such-interest", {})):
        for path in (f"/api/profiles/{key}", f"/api/profiles/{key}/detail"):
            r = _request(app, "GET", path, headers=headers)
            seen.append((r.status_code, r.json()))
    assert seen == [(404, {"detail": "No such interest."})] * 4
