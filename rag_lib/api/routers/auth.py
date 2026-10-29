"""Sign-in routes, mounted at ``/auth`` (outside ``/api``).

  GET /auth/config            → ``{"providers": [...], "signed_in_with": idp | null}``
  GET /auth/login/{provider}  → 302 to that provider's authorize endpoint
  GET /auth/login/callback    → 302 to ``/`` with the session cookie set
  GET /auth/logout            → 302 to ``/`` with the session cookie cleared

The other routes 404 when no provider is on, so a deployment on the
``X-User-Email`` header model exposes nothing new.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Cookie, Depends, HTTPException, status
from fastapi.responses import RedirectResponse

from rag_lib.db.repos import users

from .. import oidc
from ..deps import get_db
from ..settings import Settings, get_settings

router = APIRouter()

def _require_oidc(settings: Annotated[Settings, Depends(get_settings)]) -> Settings:
    if not oidc.providers(settings):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return settings


def _bad(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


@router.get("/config")
def auth_config(
    settings: Annotated[Settings, Depends(get_settings)],
    bioxplorer_session: Annotated[str | None, Cookie()] = None,
) -> dict:
    on = list(oidc.providers(settings))
    session = oidc.read_session(settings, bioxplorer_session) if on else None
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
        raise _bad(f"sign-in was not completed: {error}")
    flow = oidc.read_flow(settings, bioxplorer_oidc)
    if flow is None:
        raise _bad("this sign-in expired or was started elsewhere; start again")
    if not code or not state or state != flow["state"]:
        raise _bad("sign-in state does not match; start again")
    p = oidc.providers(settings).get(flow.get("provider", ""))
    if p is None:
        raise _bad("that sign-in provider is no longer available; start again")
    try:
        id_token = oidc.exchange_code(settings, p, code, flow["verifier"])
        claims = oidc.verify_id_token(p, id_token, flow["nonce"])
    except oidc.OIDCError as exc:
        raise _bad(str(exc)) from None
    email = (claims.get("email") or "").strip().lower()
    if not email or claims.get("email_verified") is not True:
        raise _bad(f"BioXplorer needs a verified email from {p.name.title()}; add one there and try again")
    users.upsert(db, email)

    resp = RedirectResponse("/", status_code=302)
    resp.set_cookie(
        oidc.SESSION_COOKIE,
        oidc.sign_session(
            settings,
            email,
            str(claims["sub"]),
            idp=p.name,
            self_rid=claims.get("self_rid"),
        ),
        max_age=settings.BIOXPLORER_SESSION_DAYS * 86400,
        httponly=True,
        samesite="lax",
        secure=settings.BIOXPLORER_COOKIE_SECURE,
        path="/",
    )
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
def logout(settings: Annotated[Settings, Depends(_require_oidc)]) -> RedirectResponse:
    resp = RedirectResponse("/", status_code=302)
    resp.delete_cookie(oidc.SESSION_COOKIE, path="/")
    return resp
