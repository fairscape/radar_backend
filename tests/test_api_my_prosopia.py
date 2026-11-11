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


def test_another_users_import_is_not_mine(app, issuer, prosopia, tmp_path):
    """A has imported the profile; B signs in with the same self_rid (the
    same Prosopia profile) and sees it as not imported, not A's researcher.
    The cross-user route sweep lists this route here (round12_isolation)."""
    def a_imports():
        conn = connect(tmp_path / "radar.db")
        a = conn.execute("INSERT INTO users (email, mailto) VALUES ('a@example.org', 'a@example.org')").lastrowid
        researchers_repo.upsert(conn, user_id=a, source="prosopia", key="sheffield-nathan", name="N")
        conn.commit()
        conn.close()

    issuer.claims = {"self_rid": RID, "email": "b@example.org"}
    body = _mine(app, before=a_imports)
    assert body["profile"]["slug"] == "sheffield-nathan"
    assert body["researcher_id"] is None and body["imported_at"] is None


# --- B5-3 (frontend review 2026-11-07): "already stored" and the import's
# refusal of a second profile name the same person ----------------------

ORCID = "0000-0002-1825-009X"


def _orcid_record(rid):
    return {**RECORD, "rid": rid}


def _store_by_orcid(tmp_path, orcid=ORCID):
    out = {}

    def store():
        conn = connect(tmp_path / "radar.db")
        uid = conn.execute("SELECT id FROM users WHERE email = ?", ("alice@example.org",)).fetchone()["id"]
        out["uid"] = uid
        out["id"] = researchers_repo.upsert(
            conn, user_id=uid, source="orcid", key=orcid, name="N by ORCID", orcid=orcid)
        conn.close()
    return out, store


@pytest.mark.parametrize("rid", [ORCID, "https://orcid.org/0000-0002-1825-009x"])
def test_a_person_stored_by_orcid_is_this_profile(app, issuer, prosopia, tmp_path, rid):
    """Stored from their ORCID, the person is the one this Prosopia profile is
    about: the card opens that researcher instead of offering an Import the
    server refuses (409). The rid in any ORCID form."""
    issuer.claims = {"self_rid": rid}
    prosopia.reply = httpx.Response(200, json=_orcid_record(rid))
    out, store = _store_by_orcid(tmp_path)
    body = _mine(app, before=store)
    assert body["researcher_id"] == out["id"]


def test_another_orcid_is_not_this_profile(app, issuer, prosopia, tmp_path):
    issuer.claims = {"self_rid": ORCID}
    prosopia.reply = httpx.Response(200, json=_orcid_record(ORCID))
    _, store = _store_by_orcid(tmp_path, orcid="0000-0001-5109-3700")
    assert _mine(app, before=store)["researcher_id"] is None


def test_card_and_refusal_agree(app, issuer, prosopia, tmp_path):
    """Class check: whoever the card calls already stored is whom the import
    refuses as a second profile, and the other way round."""
    from rag_lib.api.services import researchers as researchers_service

    issuer.claims = {"self_rid": ORCID}
    prosopia.reply = httpx.Response(200, json=_orcid_record(ORCID))
    out, store = _store_by_orcid(tmp_path)
    body = _mine(app, before=store)
    conn = connect(tmp_path / "radar.db")
    try:
        with pytest.raises(researchers_service.ResearcherExists) as refused:
            researchers_service.refuse_a_second_profile(
                conn, user_id=out["uid"], source="prosopia", key=RECORD["slug"], orcid=ORCID, refresh_of=None)
    finally:
        conn.close()
    assert refused.value.researcher_id == body["researcher_id"]
