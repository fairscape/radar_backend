"""``GET /api/users/me/prosopia``: the signed-in person's own profile.

Sign-in runs through the fake issuer from ``test_api_oidc``; the Prosopia
read (``httpx.get`` in the users router) is monkeypatched. No ``self_rid``,
a 404, or Prosopia down must be a 200 with no profile, never an error.
"""

from __future__ import annotations

import httpx
import pytest

from rag_lib.api.routers import users as users_router
from rag_lib.db import connect
from rag_lib.db.repos import researchers as researchers_repo

from tests.test_api_oidc import _session, _sign_in, app, issuer  # noqa: F401  (fixtures)

RID = "local:nathan-sheffield-a1b2c3"
RECORD = {
    "slug": "sheffield-nathan",
    "rid": RID,
    "fields": {"name": "Nathan C. Sheffield", "affiliation": "UVA", "paper_count": 82},
}


@pytest.fixture()
def prosopia(monkeypatch):
    """Fake profile read. Set ``.reply`` to a Response or an exception."""
    fake = type("Fake", (), {})()
    fake.reply = httpx.Response(200, json=RECORD)
    fake.urls = []

    def fake_get(url, headers=None, timeout=None):
        fake.urls.append(url)
        if isinstance(fake.reply, Exception):
            raise fake.reply
        return fake.reply

    monkeypatch.setattr(users_router.httpx, "get", fake_get)
    return fake


def _mine(app, before=None):
    """Sign in, run ``before()`` if given, then load the profile."""
    async def steps(c):
        await _sign_in(c)
        if before:
            before()
        return await c.get("/api/users/me/prosopia")
    resp = _session(app, steps)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_no_self_rid_is_empty(app, issuer, prosopia):
    assert _mine(app)["profile"] is None
    assert prosopia.urls == []


def test_self_rid_returns_profile_and_researcher(app, issuer, prosopia, tmp_path):
    issuer.claims = {"self_rid": RID}
    rid = {}

    def import_researcher():
        conn = connect(tmp_path / "radar.db")
        uid = conn.execute(
            "SELECT id FROM users WHERE email = ?", ("alice@example.org",)
        ).fetchone()["id"]
        rid["id"] = researchers_repo.upsert(
            conn, user_id=uid, source="prosopia", key="sheffield-nathan", name="N"
        )
        conn.close()

    body = _mine(app, before=import_researcher)
    assert prosopia.urls[0].endswith("/api/v1/profiles/local%3Anathan-sheffield-a1b2c3")
    assert body["profile"]["slug"] == "sheffield-nathan"
    assert body["profile"]["name"] == "Nathan C. Sheffield"
    assert body["profile"]["paper_count"] == 82
    assert body["researcher_id"] == rid["id"]


@pytest.mark.parametrize(
    "reply",
    [httpx.Response(404, json={"detail": "not found"}), httpx.ConnectError("down")],
)
def test_no_public_profile_or_outage_is_empty(app, issuer, prosopia, reply):
    issuer.claims = {"self_rid": RID}
    prosopia.reply = reply
    assert _mine(app)["profile"] is None
