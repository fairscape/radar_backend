"""Sign-in buttons replace Cloudflare Access (records/21).

What has to hold once nothing in front of the site vouches for a header:

- identity is the session cookie only; X-User-Email and the Cf-Access
  header name nobody, with or without a cookie next to them;
- a deployment that requires auth but has no sign-in provider refuses to
  start, unless RADAR_TRUST_EMAIL_HEADER says a proxy overwrites the
  header (the test site) -- a lost client secret must not reopen it;
- the settings that make the cookie safe are checked at start: secret
  length, Secure behind https, no header trust next to sign-in;
- CORS no longer answers every origin with credentials;
- a write the browser marks cross-site, or from a foreign Origin, is
  refused before any route runs;
- the live Caddyfile drops the identity headers instead of copying them.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from rag_lib.api import oidc
from rag_lib.api import settings as settings_module
from rag_lib.api.settings import Settings
from rag_lib.db import apply_migrations, connect

ROOT = Path(__file__).resolve().parents[1]
SITE = "https://radar.example.org"
SECRET = "k" * 40

SIGN_IN = {
    "BIOXPLORER_GOOGLE_CLIENT_ID": "g-client",
    "BIOXPLORER_GOOGLE_CLIENT_SECRET": "g-secret",
    "BIOXPLORER_OIDC_REDIRECT_URI": f"{SITE}/auth/login/callback",
    "BIOXPLORER_SESSION_SECRET": SECRET,
    "BIOXPLORER_COOKIE_SECURE": "true",
}


def _clear(monkeypatch):
    for key in Settings.model_fields:
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------- settings

@pytest.mark.parametrize("trust", [None, "false"])
def test_auth_required_without_sign_in_refuses_to_start(monkeypatch, trust):
    _clear(monkeypatch)
    monkeypatch.setenv("RADAR_REQUIRE_AUTH", "true")
    if trust is not None:
        monkeypatch.setenv("RADAR_TRUST_EMAIL_HEADER", trust)
    with pytest.raises(ValidationError, match="no sign-in provider"):
        Settings(_env_file=None)


def test_a_lost_client_secret_refuses_to_start(monkeypatch):
    """The live .env with its Google secret gone: stop, do not trust headers."""
    _clear(monkeypatch)
    monkeypatch.setenv("RADAR_REQUIRE_AUTH", "true")
    for k, v in SIGN_IN.items():
        if k not in ("BIOXPLORER_GOOGLE_CLIENT_ID", "BIOXPLORER_GOOGLE_CLIENT_SECRET"):
            monkeypatch.setenv(k, v)
    with pytest.raises(ValidationError, match="no sign-in provider"):
        Settings(_env_file=None)


def test_header_model_has_to_be_asked_for(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("RADAR_REQUIRE_AUTH", "true")
    monkeypatch.setenv("RADAR_TRUST_EMAIL_HEADER", "true")
    s = Settings(_env_file=None)
    assert oidc.providers(s) == {}


def test_single_user_mode_still_starts(monkeypatch):
    _clear(monkeypatch)
    s = Settings(_env_file=None)
    assert s.RADAR_REQUIRE_AUTH is False and oidc.providers(s) == {}


def test_live_sign_in_settings_start(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("RADAR_REQUIRE_AUTH", "true")
    for k, v in SIGN_IN.items():
        monkeypatch.setenv(k, v)
    assert list(oidc.providers(Settings(_env_file=None))) == ["google"]


@pytest.mark.parametrize("change, message", [
    ({"BIOXPLORER_SESSION_SECRET": "short"}, "at least 32"),
    ({"BIOXPLORER_SESSION_SECRET": "k" * 31}, "at least 32"),
    ({"BIOXPLORER_COOKIE_SECURE": "false"}, "COOKIE_SECURE"),
    ({"RADAR_TRUST_EMAIL_HEADER": "true"}, "unset RADAR_TRUST_EMAIL_HEADER"),
])
def test_unsafe_sign_in_settings_refuse_to_start(monkeypatch, change, message):
    _clear(monkeypatch)
    monkeypatch.setenv("RADAR_REQUIRE_AUTH", "true")
    for k, v in {**SIGN_IN, **change}.items():
        monkeypatch.setenv(k, v)
    with pytest.raises(ValidationError, match=message):
        Settings(_env_file=None)


def test_redirect_uri_without_a_provider_refuses_to_start(monkeypatch):
    """Secrets lost and RADAR_REQUIRE_AUTH gone too: not the shared demo user."""
    _clear(monkeypatch)
    monkeypatch.setenv("BIOXPLORER_OIDC_REDIRECT_URI", SIGN_IN["BIOXPLORER_OIDC_REDIRECT_URI"])
    with pytest.raises(ValidationError, match="no sign-in provider is"):
        Settings(_env_file=None)


def test_plain_http_redirect_may_use_an_insecure_cookie(monkeypatch):
    """The test site over ssh -L is http://localhost; Secure would drop it."""
    _clear(monkeypatch)
    for k, v in {**SIGN_IN, "BIOXPLORER_OIDC_REDIRECT_URI": "http://localhost:8051/auth/login/callback",
                 "BIOXPLORER_COOKIE_SECURE": "false"}.items():
        monkeypatch.setenv(k, v)
    assert Settings(_env_file=None).BIOXPLORER_COOKIE_SECURE is False


# ---------------------------------------------------------------- the app

def _make_app(tmp_path, monkeypatch, env: dict):
    _clear(monkeypatch)
    db = tmp_path / "radar.db"
    base = {
        "RADAR_DB_PATH": str(db),
        "RADAR_VAULT_DIR": str(tmp_path / "vault"),
        "RADAR_CHROMA_DIR": str(tmp_path / "chroma"),
        "RADAR_LOG_JSON": "false",
        "RADAR_SCHEDULER_ENABLED": "false",
        "RADAR_REQUIRE_AUTH": "true",
    }
    for k, v in {**base, **env}.items():
        monkeypatch.setenv(k, v)
    settings_module.get_settings.cache_clear()
    conn = connect(db)
    apply_migrations(conn)
    conn.close()
    from rag_lib.api.app import create_app
    return create_app()


@pytest.fixture()
def app(tmp_path, monkeypatch):
    yield _make_app(tmp_path, monkeypatch, {**SIGN_IN, "RADAR_CORS_ORIGINS": '["http://localhost:5173"]'})
    settings_module.get_settings.cache_clear()


def _name() -> str:
    return oidc.session_cookie_name(settings_module.get_settings())


def _cookie(email: str) -> str:
    return oidc.sign_session(settings_module.get_settings(), email, "sub-" + email, idp="google")


def _run(app, steps):
    async def go():
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url=SITE) as c:
                return await steps(c)
    return asyncio.run(go())


FORGED = [
    {"X-User-Email": "alice@example.org"},
    {"Cf-Access-Authenticated-User-Email": "alice@example.org"},
    {"X-User-Email": "alice@example.org", "Cf-Access-Authenticated-User-Email": "alice@example.org"},
    {"X-User-Email": ""},
]


@pytest.mark.parametrize("headers", FORGED)
def test_identity_headers_name_nobody(app, headers):
    async def steps(c):
        return await c.get("/api/users/me", headers=headers)
    r = _run(app, steps)
    assert r.status_code == 401 and r.json()["detail"] == "not signed in"


@pytest.mark.parametrize("headers", FORGED[:3])
def test_identity_headers_do_not_override_the_cookie(app, headers):
    """Bob, signed in, sends Alice's address in the headers: still Bob."""
    async def steps(c):
        c.cookies.set(_name(), _cookie("bob@example.org"))
        return await c.get("/api/users/me", headers=headers)
    r = _run(app, steps)
    assert r.status_code == 200 and r.json()["email"] == "bob@example.org"


def test_two_cookies_two_accounts(app):
    async def steps(c):
        out = []
        for who in ("alice@example.org", "bob@example.org"):
            c.cookies.set(_name(), _cookie(who))
            out.append((await c.get("/api/users/me")).json())
        return out
    a, b = _run(app, steps)
    assert (a["email"], b["email"]) == ("alice@example.org", "bob@example.org")
    assert a["id"] != b["id"]


# ------------------------------------------------------------ cross-site

@pytest.mark.parametrize("headers, refused", [
    ({}, False),                                                  # curl, old browsers
    ({"Sec-Fetch-Site": "same-origin", "Origin": SITE}, False),   # the SPA
    ({"Sec-Fetch-Site": "none"}, False),                          # typed by the user
    ({"Origin": "http://localhost:5173"}, False),                 # vite dev, listed in CORS
    ({"Sec-Fetch-Site": "cross-site", "Origin": "https://evil.example"}, True),
    ({"Sec-Fetch-Site": "same-site", "Origin": "https://other.example.org"}, True),
    ({"Origin": "https://evil.example"}, True),                   # no Sec-Fetch-*
    ({"Sec-Fetch-Site": "cross-site"}, True),
    # Another *.example.org subdomain: SameSite=Lax sends the cookie, and
    # the browser may leave Origin out of the check -- the site header
    # alone has to refuse it.
    ({"Sec-Fetch-Site": "same-site"}, True),
    ({"Sec-Fetch-Site": "same-site", "Origin": SITE}, True),
    ({"Origin": "null"}, True),                                   # sandboxed iframe / data:
])
def test_cross_site_writes_are_refused(app, headers, refused):
    async def steps(c):
        c.cookies.set(_name(), _cookie("alice@example.org"))
        return await c.patch("/api/users/me", json={"mailto": "a@example.org"}, headers=headers)
    r = _run(app, steps)
    if refused:
        assert r.status_code == 403 and r.json()["detail"] == "cross-site request refused"
    else:
        assert r.status_code == 200


def test_cross_site_write_changes_nothing(app):
    async def steps(c):
        c.cookies.set(_name(), _cookie("alice@example.org"))
        before = (await c.get("/api/users/me")).json()["mailto"]
        r = await c.patch("/api/users/me", json={"mailto": "evil@example.org"},
                          headers={"Sec-Fetch-Site": "cross-site", "Origin": "https://evil.example"})
        after = (await c.get("/api/users/me")).json()["mailto"]
        return r.status_code, before, after
    code, before, after = _run(app, steps)
    assert code == 403 and before == after != "evil@example.org"


def test_cross_site_reads_are_not_blocked_but_not_shared(app):
    """A GET is answered (links must work); CORS keeps it from a foreign page."""
    async def steps(c):
        c.cookies.set(_name(), _cookie("alice@example.org"))
        return await c.get("/api/users/me", headers={"Origin": "https://evil.example"})
    r = _run(app, steps)
    assert r.status_code == 200
    assert "access-control-allow-origin" not in r.headers


def test_cors_answers_only_listed_origins_with_credentials(app):
    async def steps(c):
        out = {}
        for origin in ("http://localhost:5173", "https://evil.example"):
            out[origin] = await c.options("/api/users/me", headers={
                "Origin": origin, "Access-Control-Request-Method": "PATCH"})
        return out
    out = _run(app, steps)
    ok = out["http://localhost:5173"].headers
    assert ok.get("access-control-allow-origin") == "http://localhost:5173"
    assert ok.get("access-control-allow-credentials") == "true"
    assert "access-control-allow-origin" not in out["https://evil.example"].headers


def test_cors_wildcard_never_carries_credentials(tmp_path, monkeypatch):
    app = _make_app(tmp_path, monkeypatch, {"RADAR_REQUIRE_AUTH": "false", "RADAR_CORS_ORIGINS": '["*"]'})
    try:
        async def steps(c):
            return await c.options("/api/users/me", headers={
                "Origin": "https://evil.example", "Access-Control-Request-Method": "PATCH"})
        r = _run(app, steps)
        assert r.headers.get("access-control-allow-credentials") != "true"
    finally:
        settings_module.get_settings.cache_clear()


def test_header_model_has_no_cross_site_guard_to_get_in_the_way(tmp_path, monkeypatch):
    """The test site (header model) keeps working with no Origin rules."""
    app = _make_app(tmp_path, monkeypatch, {"RADAR_TRUST_EMAIL_HEADER": "true"})
    try:
        async def steps(c):
            return await c.patch("/api/users/me", json={"mailto": "a@example.org"},
                                 headers={"X-User-Email": "a@example.org", "Origin": "http://localhost:8051"})
        assert _run(app, steps).status_code == 200
    finally:
        settings_module.get_settings.cache_clear()


# ------------------------------------------------------------- Caddyfile

def _blocks(text: str) -> dict[str, str]:
    """The body of each ``handle <path>`` block in the live Caddyfile."""
    out = {}
    for m in re.finditer(r"^\thandle (/\S+) \{\n(.*?)^\t\}", text, re.S | re.M):
        out[m.group(1)] = m.group(2)
    return out


def test_live_caddyfile_drops_identity_headers():
    text = (ROOT / "deploy" / "Caddyfile").read_text()
    code = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    # Nothing copies a client header into the identity any more.
    assert not re.search(r"header_up\s+X-User-Email\s+\S", code)
    assert "Cf-Access-Authenticated-User-Email}" not in code
    blocks = _blocks(code)
    for path in ("/api/*", "/auth/*"):
        for h in ("X-User-Email", "Cf-Access-Authenticated-User-Email", "Cf-Access-Jwt-Assertion"):
            assert f"header_up -{h}" in blocks[path], (path, h)
    assert "resp_headers>Set-Cookie delete" in code
    assert "request>headers>Cookie delete" in code


# --------------------------------------------------------- cookie name

def test_secure_session_cookie_is_host_prefixed(app):
    """Behind https the cookie is __Host-: no other subdomain can plant one."""
    assert _name() == "__Host-bioxplorer_session"

    async def steps(c):
        # The bare name a subdomain could set with Domain=example.org is
        # not read at all.
        c.cookies.set(oidc.SESSION_COOKIE, _cookie("mallory@example.org"))
        return await c.get("/api/users/me")
    assert _run(app, steps).status_code == 401


def test_logout_clears_the_secure_cookie(app):
    async def steps(c):
        return await c.get("/auth/logout")
    r = _run(app, steps)
    assert r.status_code == 302 and r.headers["location"] == "/"
    sc = r.headers["set-cookie"]
    assert sc.startswith("__Host-bioxplorer_session=") and "Max-Age=0" in sc
    assert "Secure" in sc and "Path=/" in sc


def test_logout_answers_on_the_header_model_too(tmp_path, monkeypatch):
    """The frontend sends every Sign out to /auth/logout; never a 404 page."""
    app = _make_app(tmp_path, monkeypatch, {"RADAR_TRUST_EMAIL_HEADER": "true"})
    try:
        async def steps(c):
            return await c.get("/auth/logout")
        r = _run(app, steps)
        assert r.status_code == 302 and r.headers["location"] == "/"
    finally:
        settings_module.get_settings.cache_clear()


# ------------------------------------------------------------- id_token

def test_google_asks_only_for_the_address(monkeypatch):
    _clear(monkeypatch)
    for k, v in SIGN_IN.items():
        monkeypatch.setenv(k, v)
    assert oidc.providers(Settings(_env_file=None))["google"].scope == "openid email"


@pytest.mark.parametrize("skew, ok", [(30, True), (55, True), (120, False)])
def test_id_token_allows_a_minute_of_clock_skew(monkeypatch, skew, ok):
    import time
    from types import SimpleNamespace

    import jwt
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    iss = "https://issuer.test"
    monkeypatch.setattr(oidc, "discover", lambda i: {"issuer": iss, "jwks_uri": iss + "/jwks"})
    monkeypatch.setattr(oidc, "_jwks_client", lambda uri: SimpleNamespace(
        get_signing_key_from_jwt=lambda tok: SimpleNamespace(key=key.public_key())))
    now = int(time.time()) + skew          # the issuer's clock runs ahead
    tok = jwt.encode({"iss": iss, "aud": "c", "sub": "s", "nonce": "n",
                      "iat": now, "nbf": now, "exp": now + 300}, key, algorithm="RS256")
    p = oidc.Provider("prosopia", iss, "c", "secret", "openid email")
    if ok:
        assert oidc.verify_id_token(p, tok, "n")["sub"] == "s"
    else:
        with pytest.raises(oidc.OIDCError):
            oidc.verify_id_token(p, tok, "n")


# ------------------------------------------------------------- renewal

def _aged_cookie(monkeypatch, email: str, days: float) -> str:
    import time as _time
    real = _time.time()
    monkeypatch.setattr(oidc.time, "time", lambda: real - days * 86400)
    try:
        return _cookie(email)
    finally:
        monkeypatch.setattr(oidc.time, "time", lambda: real)


def _set_cookie_token(r) -> str | None:
    for line in r.headers.get_list("set-cookie"):
        if line.startswith(_name() + "="):
            return line.split(";")[0].split("=", 1)[1]
    return None


@pytest.mark.parametrize("age_days, renewed", [(1, False), (6.9, False), (7.1, True), (13.5, True)])
def test_session_is_renewed_on_use_past_half_its_life(app, monkeypatch, age_days, renewed):
    old = _aged_cookie(monkeypatch, "alice@example.org", age_days)

    async def steps(c):
        c.cookies.set(_name(), old)
        return await c.get("/api/users/me")
    r = _run(app, steps)
    assert r.status_code == 200
    token = _set_cookie_token(r)
    if not renewed:
        assert token is None
        return
    s = oidc.read_session(settings_module.get_settings(), token)
    assert s["email"] == "alice@example.org" and s["idp"] == "google"
    assert s["exp"] - s["iat"] == 14 * 86400 and s["iat"] > oidc.read_session(settings_module.get_settings(), old)["iat"]
    line = next(l for l in r.headers.get_list("set-cookie") if l.startswith(_name()))
    assert "HttpOnly" in line and "Secure" in line and "Path=/" in line and "Max-Age=1209600" in line


def test_expired_session_is_not_renewed(app, monkeypatch):
    old = _aged_cookie(monkeypatch, "alice@example.org", 15)

    async def steps(c):
        c.cookies.set(_name(), old)
        return await c.get("/api/users/me")
    r = _run(app, steps)
    assert r.status_code == 401 and _set_cookie_token(r) is None


def test_failed_request_does_not_renew(app, monkeypatch):
    old = _aged_cookie(monkeypatch, "alice@example.org", 10)

    async def steps(c):
        c.cookies.set(_name(), old)
        return await c.get("/api/profiles/no-such-interest")
    r = _run(app, steps)
    assert r.status_code == 404 and _set_cookie_token(r) is None


def test_logout_is_not_undone_by_renewal(app, monkeypatch):
    """/auth/logout clears the cookie; a renewal next to it would undo that."""
    old = _aged_cookie(monkeypatch, "alice@example.org", 10)

    async def steps(c):
        c.cookies.set(_name(), old)
        return await c.get("/auth/logout")
    r = _run(app, steps)
    mine = [l for l in r.headers.get_list("set-cookie") if l.startswith(_name() + "=")]
    # Exactly one: the clearing one, no fresh token next to it.
    assert len(mine) == 1 and "Max-Age=0" in mine[0]
