"""Sign-in routes, mounted at ``/auth`` (outside ``/api``).

  GET /auth/config            → ``{"providers": [...], "signed_in_with": idp | null}``
  GET /auth/login/{provider}  → 302 to that provider's authorize endpoint
  GET /auth/login/callback    → 302 to ``/`` with the session cookie set,
                                or to ``/?signin_error=<code>`` when it failed
  GET /auth/logout            → 302 to ``/`` with the session cookie cleared

Login and callback 404 when no provider is on, so a deployment on the
``X-User-Email`` header model exposes nothing new. Logout always answers:
the frontend sends every Sign out there, and clearing a cookie that is not
there is harmless.

A failed callback lands on the login page with one of a fixed set of
codes, never the provider's own text (that came back verbatim in a bare
JSON page before, and any link could put words in it). The reason is
logged.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated

import structlog
from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse

from rag_lib.db.repos import users

from .. import oidc
from ..deps import get_db
from ..settings import Settings, get_settings

router = APIRouter()
log = structlog.get_logger("rag_lib.api.routers.auth")

# What the login page knows how to say (LoginPage SIGNIN_ERRORS).
SIGNIN_ERRORS = ("cancelled", "expired", "unverified", "failed")


def _back_with(code: str, reason: str) -> RedirectResponse:
    log.warning("auth.signin_failed", code=code, reason=reason[:300])
    resp = RedirectResponse(f"/?signin_error={code}", status_code=302)
    resp.delete_cookie(oidc.FLOW_COOKIE, path="/auth")
    return resp

def _require_oidc(settings: Annotated[Settings, Depends(get_settings)]) -> Settings:
    if not oidc.providers(settings):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return settings


@router.get("/config")
def auth_config(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> dict:
    on = list(oidc.providers(settings))
    session = oidc.request_session(settings, request) if on else None
    return {
        "providers": on,
        "signed_in_with": session.get("idp") if session else None,
    }


@router.get("/login/callback")
def callback(
    settings: Annotated[Settings, Depends(_require_oidc)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    bioxplorer_oidc: Annotated[str | None, Cookie()] = None,
) -> RedirectResponse:
    if error:
        return _back_with("cancelled", f"provider error {error!r}")
    flow = oidc.read_flow(settings, bioxplorer_oidc)
    if flow is None:
        return _back_with("expired", "no valid flow cookie")
    if not code or not state or state != flow["state"]:
        return _back_with("expired", "state missing or does not match")
    p = oidc.providers(settings).get(flow.get("provider", ""))
    if p is None:
        return _back_with("failed", f"provider {flow.get('provider')!r} is off")
    try:
        id_token = oidc.exchange_code(settings, p, code, flow["verifier"])
        claims = oidc.verify_id_token(p, id_token, flow["nonce"])
    except oidc.OIDCError as exc:
        return _back_with("failed", str(exc))
    email = (claims.get("email") or "").strip().lower()
    if not email or claims.get("email_verified") is not True:
        return _back_with("unverified", f"{p.name}: no verified email")
    users.upsert(db, email)

    resp = RedirectResponse("/", status_code=302)
    oidc.set_session_cookie(settings, resp, oidc.sign_session(
        settings,
        email,
        str(claims["sub"]),
        idp=p.name,
        self_rid=claims.get("self_rid"),
    ))
    resp.delete_cookie(oidc.FLOW_COOKIE, path="/auth")
    return resp


@router.get("/login/{provider}")
def login(
    provider: str,
    settings: Annotated[Settings, Depends(_require_oidc)],
) -> RedirectResponse:
    p = oidc.providers(settings).get(provider)
    if p is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    flow = oidc.new_flow(p.name)
    try:
        url = oidc.authorize_url(settings, p, flow)
    except oidc.OIDCError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from None
    resp = RedirectResponse(url, status_code=302)
    resp.set_cookie(
        oidc.FLOW_COOKIE,
        oidc.sign_flow(settings, flow),
        max_age=oidc.FLOW_SECONDS,
        httponly=True,
        samesite="lax",
        secure=settings.BIOXPLORER_COOKIE_SECURE,
        path="/auth",
    )
    return resp


@router.get("/logout")
def logout(settings: Annotated[Settings, Depends(get_settings)]) -> RedirectResponse:
    resp = RedirectResponse("/", status_code=302)
    secure = settings.BIOXPLORER_COOKIE_SECURE
    resp.delete_cookie(oidc.session_cookie_name(settings), path="/", secure=secure, httponly=True, samesite="lax")
    return resp
