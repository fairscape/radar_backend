"""Sign-in through OpenID Connect (Prosopia and Google).

The issuers are faked in-process: discovery, the JWKS lookup and the token
endpoint are monkeypatched, and the id_token is signed with an RSA key
made here. Everything else -- cookies, PKCE, the identity dependency --
runs for real.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from rag_lib.api import oidc
from rag_lib.api import settings as settings_module
from rag_lib.db import apply_migrations, connect

ISSUER = "https://prosopia.test"
GOOGLE = oidc.GOOGLE_ISSUER
REDIRECT = "http://test/auth/login/callback"

_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _id_token(nonce: str, **over) -> str:
    now = int(time.time())
    claims = {
        "iss": ISSUER,
        "aud": "bioxplorer",
        "sub": "p_123",
        "email": "Alice@Example.org",
        "email_verified": True,
        "nonce": nonce,
        "iat": now,
        "exp": now + 300,
        **over,
    }
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, _KEY, algorithm="RS256")


@pytest.fixture()
def issuer(monkeypatch):
    """Fake issuer. ``issuer.claims`` overrides what the next id_token says."""
    state = SimpleNamespace(claims={}, token_requests=[])
    monkeypatch.setattr(oidc, "discover", lambda iss: {
        "issuer": iss,
        "authorization_endpoint": f"{iss}/oauth/authorize",
        "token_endpoint": f"{iss}/oauth/token",
        "jwks_uri": f"{iss}/oauth/jwks.json",
    })
    monkeypatch.setattr(
        oidc,
        "_jwks_client",
        lambda uri: SimpleNamespace(
            get_signing_key_from_jwt=lambda tok: SimpleNamespace(key=_KEY.public_key())
        ),
    )

    def fake_post(url, data, auth, timeout):
        state.token_requests.append(data)
        # The code carries the nonce so the token can echo it.
        nonce = data["code"].removeprefix("code-")
        return httpx.Response(200, json={"id_token": _id_token(nonce, **state.claims)})

    monkeypatch.setattr(oidc.httpx, "post", fake_post)
    return state


@pytest.fixture()
def app(tmp_path, monkeypatch, issuer):
    db = tmp_path / "radar.db"
    for k, v in {
        "RADAR_DB_PATH": str(db),
        "RADAR_VAULT_DIR": str(tmp_path / "vault"),
        "RADAR_CHROMA_DIR": str(tmp_path / "chroma"),
        "RADAR_LOG_JSON": "false",
        "RADAR_SCHEDULER_ENABLED": "false",
        "RADAR_PROSOPIA_BASE_URL": ISSUER,
        "BIOXPLORER_PROSOPIA_CLIENT_SECRET": "s3cret",
        "BIOXPLORER_GOOGLE_CLIENT_ID": "g-client",
        "BIOXPLORER_GOOGLE_CLIENT_SECRET": "g-s3cret",
        "BIOXPLORER_OIDC_REDIRECT_URI": REDIRECT,
        "BIOXPLORER_SESSION_SECRET": "session-key-for-tests-only-32bytes",
    }.items():
        monkeypatch.setenv(k, v)
    settings_module.get_settings.cache_clear()
    conn = connect(db)
    apply_migrations(conn)
    conn.close()

    from rag_lib.api.app import create_app

    yield create_app()
    settings_module.get_settings.cache_clear()


def _session(app, steps):
    """Run ``steps(client)`` in one client so cookies carry between calls."""
    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await steps(client)
    return asyncio.run(_run())


async def _sign_in(client, provider="prosopia") -> httpx.Response:
    q = await _login_params(client, provider)
    return await client.get(
        "/auth/login/callback", params={"code": f"code-{q['nonce']}", "state": q["state"]}
    )


async def _login_params(client, provider="prosopia") -> dict:
    start = await client.get(f"/auth/login/{provider}")
    return {k: v[0] for k, v in parse_qs(urlparse(start.headers["location"]).query).items()}


def test_sign_in_and_out(app, issuer):
    async def steps(c):
        q = await _login_params(c)
        cb = await c.get(
            "/auth/login/callback",
            params={"code": f"code-{q['nonce']}", "state": q["state"]},
        )
        me = await c.get("/api/users/me")
        await c.get("/auth/logout")
        return q, cb, me, await c.get("/api/users/me")
    q, cb, me, after = _session(app, steps)
    assert q["scope"] == "openid email profile profiles:own"
    assert cb.status_code == 302
    assert me.status_code == 200, me.text
    assert me.json()["email"] == "alice@example.org"
    assert after.status_code == 401
    challenge = q["code_challenge"]
    # The verifier sent to the token endpoint matches the challenge sent
    # to the authorize endpoint.
    verifier = issuer.token_requests[0]["code_verifier"]
    digest = hashlib.sha256(verifier.encode()).digest()
    assert challenge == base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


@pytest.mark.parametrize("iss", [GOOGLE, "accounts.google.com"])
def test_google_sign_in(app, issuer, iss):
    """Google signs some id_tokens with the bare host as issuer."""
    issuer.claims = {"iss": iss, "aud": "g-client", "sub": "g_1"}
    async def steps(c):
        await _sign_in(c, "google")
        return await c.get("/api/users/me"), await c.get("/auth/config")
    me, config = _session(app, steps)
    assert me.json()["email"] == "alice@example.org"
    assert config.json() == {"providers": ["google", "prosopia"], "signed_in_with": "google"}


def test_wrong_state_is_400(app):
    async def steps(c):
        q = await _login_params(c)
        return await c.get(
            "/auth/login/callback",
            params={"code": f"code-{q['nonce']}", "state": "forged"},
        )
    assert _session(app, steps).status_code == 400


@pytest.mark.parametrize(
    "claims",
    [{"email": None}, {"email_verified": False}, {"aud": "someone-else"}, {"sub": None}],
    ids=["no-email", "unverified-email", "wrong-audience", "no-sub"],
)
def test_bad_id_token_is_refused(app, issuer, claims):
    issuer.claims = claims
    async def steps(c):
        return await _sign_in(c)
    assert _session(app, steps).status_code == 400


@pytest.mark.parametrize("cookie", [None, "forged", "flow"])
def test_no_valid_session_is_401(app, cookie):
    """The X-User-Email header is ignored; only a real session cookie counts."""
    async def steps(c):
        if cookie == "forged":
            claims = {"kind": "session", "email": "mallory@example.org", "sub": "x",
                      "exp": int(time.time()) + 60}
            c.cookies.set(oidc.SESSION_COOKIE, jwt.encode(claims, "not-the-session-key-but-32-bytes-long", algorithm="HS256"))
        elif cookie == "flow":
            # A login-flow cookie is signed with the same key but is not a session.
            await c.get("/auth/login/prosopia")
            c.cookies.set(oidc.SESSION_COOKIE, c.cookies.get(oidc.FLOW_COOKIE))
        return await c.get("/api/users/me", headers={"X-User-Email": "mallory@example.org"})
    assert _session(app, steps).status_code == 401
