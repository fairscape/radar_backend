"""OpenID Connect sign-in (authorization code flow with PKCE) against
Google and/or Prosopia; each is on when its client secret is set.

No server-side store: two HS256 cookies under ``BIOXPLORER_SESSION_SECRET``
carry everything. ``bioxplorer_oidc`` is the in-flight login (provider, state,
nonce, PKCE verifier; ten minutes). ``bioxplorer_session`` is who is signed in
(email, ``sub``, ``idp``, and ``self_rid`` when Prosopia sends it).
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
from dataclasses import dataclass
from functools import lru_cache

import httpx
import jwt

from .settings import Settings

SESSION_COOKIE = "bioxplorer_session"
FLOW_COOKIE = "bioxplorer_oidc"
FLOW_SECONDS = 600

_SESSION_KIND = "session"
_FLOW_KIND = "oidc_flow"

class OIDCError(Exception):
    """A sign-in step failed in a way the person should be told about."""


@dataclass(frozen=True)
class Provider:
    name: str
    issuer: str
    client_id: str
    client_secret: str
    scope: str


GOOGLE_ISSUER = "https://accounts.google.com"


def providers(settings: Settings) -> dict[str, Provider]:
    """The providers that are on, in the order the login page shows them."""
    out: dict[str, Provider] = {}
    if settings.BIOXPLORER_GOOGLE_CLIENT_SECRET:
        out["google"] = Provider(
            "google", GOOGLE_ISSUER,
            settings.BIOXPLORER_GOOGLE_CLIENT_ID or "", settings.BIOXPLORER_GOOGLE_CLIENT_SECRET,
            "openid email profile",
        )
    if settings.BIOXPLORER_PROSOPIA_CLIENT_SECRET:
        # profiles:own adds self_rid, the person's own profile.
        out["prosopia"] = Provider(
            "prosopia", settings.RADAR_PROSOPIA_BASE_URL.rstrip("/"),
            settings.BIOXPLORER_PROSOPIA_CLIENT_ID, settings.BIOXPLORER_PROSOPIA_CLIENT_SECRET,
            "openid email profile profiles:own",
        )
    return out


@lru_cache(maxsize=4)
def discover(issuer: str) -> dict:
    url = issuer.rstrip("/") + "/.well-known/openid-configuration"
    try:
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise OIDCError(f"could not reach {issuer}: {exc}") from None


@lru_cache(maxsize=4)
def _jwks_client(jwks_uri: str) -> jwt.PyJWKClient:
    # PyJWKClient fetches with urllib, whose default "Python-urllib" user
    # agent Cloudflare answers with 403 in front of prosopia.databio.org.
    return jwt.PyJWKClient(jwks_uri, headers={"User-Agent": "bioxplorer-backend"})


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def new_flow(provider: str) -> dict:
    """Fresh state, nonce and PKCE verifier for one login attempt."""
    return {
        "provider": provider,
        "state": secrets.token_urlsafe(24),
        "nonce": secrets.token_urlsafe(24),
        "verifier": secrets.token_urlsafe(48),
    }


def code_challenge(verifier: str) -> str:
    return _b64url(hashlib.sha256(verifier.encode()).digest())


def _sign(settings: Settings, payload: dict, seconds: int) -> str:
    now = int(time.time())
    return jwt.encode(
        {**payload, "iat": now, "exp": now + seconds},
        settings.BIOXPLORER_SESSION_SECRET,
        algorithm="HS256",
    )


def _unsign(settings: Settings, token: str | None, kind: str) -> dict | None:
    if not token:
        return None
    try:
        claims = jwt.decode(token, settings.BIOXPLORER_SESSION_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    return claims if claims.get("kind") == kind else None


def sign_flow(settings: Settings, flow: dict) -> str:
    return _sign(settings, {**flow, "kind": _FLOW_KIND}, FLOW_SECONDS)


def read_flow(settings: Settings, token: str | None) -> dict | None:
    return _unsign(settings, token, _FLOW_KIND)


def sign_session(
    settings: Settings,
    email: str,
    sub: str,
    *,
    idp: str,
    self_rid: object = None,
) -> str:
    """The session cookie. ``self_rid`` is the rid of the person's own
    Prosopia profile, stored only when Prosopia sends one."""
    payload = {"kind": _SESSION_KIND, "email": email, "sub": sub, "idp": idp}
    if isinstance(self_rid, str) and self_rid.strip():
        payload["self_rid"] = self_rid.strip()
    return _sign(settings, payload, settings.BIOXPLORER_SESSION_DAYS * 86400)


def read_session(settings: Settings, token: str | None) -> dict | None:
    return _unsign(settings, token, _SESSION_KIND)


def authorize_url(settings: Settings, p: Provider, flow: dict) -> str:
    meta = discover(p.issuer)
    params = {
        "response_type": "code",
        "client_id": p.client_id,
        "redirect_uri": settings.BIOXPLORER_OIDC_REDIRECT_URI,
        "scope": p.scope,
        "state": flow["state"],
        "nonce": flow["nonce"],
        "code_challenge": code_challenge(flow["verifier"]),
        "code_challenge_method": "S256",
    }
    return str(httpx.URL(meta["authorization_endpoint"], params=params))


def exchange_code(settings: Settings, p: Provider, code: str, verifier: str) -> str:
    """Trade the code for tokens; return the raw id_token."""
    meta = discover(p.issuer)
    try:
        resp = httpx.post(
            meta["token_endpoint"],
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": settings.BIOXPLORER_OIDC_REDIRECT_URI,
                "code_verifier": verifier,
            },
            auth=(p.client_id, p.client_secret),
            timeout=15,
        )
    except httpx.HTTPError as exc:
        raise OIDCError(f"could not reach {p.issuer}: {exc}") from None
    if resp.status_code != 200:
        raise OIDCError(f"token exchange failed ({resp.status_code})")
    id_token = resp.json().get("id_token")
    if not id_token:
        raise OIDCError("the provider returned no id_token")
    return id_token


def verify_id_token(p: Provider, id_token: str, nonce: str) -> dict:
    meta = discover(p.issuer)
    try:
        key = _jwks_client(meta["jwks_uri"]).get_signing_key_from_jwt(id_token)
        claims = jwt.decode(
            id_token,
            key.key,
            algorithms=["RS256"],
            audience=p.client_id,
            options={"verify_iss": False, "require": ["exp", "iat", "iss", "sub", "aud"]},
        )
    except jwt.PyJWTError as exc:
        raise OIDCError(f"id_token rejected: {exc}") from None
    ok = {meta["issuer"]}
    if p.name == "google":
        ok.add("accounts.google.com")  # Google's documented second issuer
    if claims["iss"] not in ok:
        raise OIDCError("id_token issuer does not match the provider")
    if claims.get("nonce") != nonce:
        raise OIDCError("id_token nonce does not match this login")
    return claims
