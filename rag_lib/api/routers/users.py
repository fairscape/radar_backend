"""Users routes — Phase 12.

Three endpoints:

  GET   /api/users/me            → ``User`` for the authenticated email.
  PATCH /api/users/me            → ``User`` (after applying ``mailto`` change).
  GET   /api/users/me/prosopia   → ``MyProsopia``: the person's self
                                   profile on Prosopia, if public.
  POST  /api/users/me/lease      → ``{editing}``: this tab takes or keeps
                                   the account's right to change things.
  POST  /api/users/me/lease/release → this tab lets go of it (W.A).

The acting user is always resolved by ``get_current_user`` (the sign-in
session cookie; the header only on a header-model setup); this router never accepts a user id in the
URL. There is no admin surface and no per-user list endpoint — Phase 12
is intentionally minimal.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated, Any
from urllib.parse import quote

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status

from rag_lib.db.repos import researchers as researchers_repo
from rag_lib.prosopia_client import looks_like_orcid, normalize_rid

from .. import oidc, write_lease
from ..deps import get_current_user, get_db
from ..schemas import MyProsopia, MyProsopiaProfile, User, UserUpdateRequest
from ..settings import Settings, get_settings

log = structlog.get_logger("rag_lib.api.routers.users")

router = APIRouter()


def _row_to_user(row: sqlite3.Row) -> User:
    return User(
        id=int(row["id"]),
        email=row["email"],
        mailto=row["mailto"],
        created_at=row["created_at"],
    )


@router.get("/me", response_model=User)
def get_me(
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
) -> User:
    return _row_to_user(user)


@router.patch("/me", response_model=User)
def update_me(
    body: UserUpdateRequest,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> User:
    """Update the authenticated user's mailto.

    The ``mailto`` is what the gatherer sends as OpenAlex's polite-pool
    contact — must look like an email. Empty / whitespace-only payloads
    are rejected so a typo can't silently break the next gather.
    """
    if body.mailto is None:
        # Treat as no-op; explicit clear isn't supported because the
        # gatherer needs *some* mailto to be polite-pool eligible.
        return _row_to_user(user)
    new_mailto = body.mailto.strip()
    if "@" not in new_mailto or " " in new_mailto:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"mailto must be an email address, got {body.mailto!r}",
        )
    from rag_lib.db.repos import users as users_repo

    updated = users_repo.upsert(db, user["email"], mailto=new_mailto)
    return _row_to_user(updated)


def _lookup_profile(base_url: str, rid: str) -> dict[str, Any] | None:
    """The public record of profile ``rid``, or None (404, error, outage).

    This feeds an optional card, so a Prosopia outage must look like "no
    profile", never like a broken page.
    """
    try:
        resp = httpx.get(
            f"{base_url}/api/v1/profiles/{quote(rid, safe='')}",
            # Cloudflare in front of prosopia.databio.org 403s default
            # Python user agents.
            headers={"User-Agent": "bioxplorer-backend"},
            timeout=10,
        )
        return resp.json() if resp.status_code == 200 else None
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("my_prosopia.lookup_failed", rid=rid, error=str(exc))
        return None


@router.get("/me/prosopia", response_model=MyProsopia)
def get_my_prosopia(
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    request: Request,
) -> MyProsopia:
    """The person's self profile on Prosopia, if it is public.

    Keyed on the ``self_rid`` claim from sign-in (scope ``profiles:own``):
    the rid of the profile the Prosopia account says is about itself. No
    session, no ``self_rid``, or no public profile all answer 200 with
    nulls; the frontend then simply shows no card.
    """
    session = oidc.request_session(settings, request) if oidc.providers(settings) else None
    if not session or not session.get("self_rid"):
        return MyProsopia()
    base = settings.RADAR_PROSOPIA_BASE_URL.rstrip("/")
    record = _lookup_profile(base, session["self_rid"])
    if not record or not record.get("slug"):
        return MyProsopia()
    slug = record["slug"]
    fields = record.get("fields") or {}
    profile = MyProsopiaProfile(
        slug=slug,
        name=fields.get("name") or slug,
        affiliation=fields.get("affiliation"),
        paper_count=fields.get("paper_count"),
        url=f"{base}/{slug}",
    )
    # The same person as the import's refusal of a second profile: this
    # Prosopia profile, or a researcher with the same ORCID under either
    # source. By slug alone the card offered Import for someone stored from
    # their ORCID, and the import then refused with a 409 (B5-3).
    # The record's rid only, as the import reads it (review R-5).
    rid = normalize_rid(record.get("rid") or fields.get("rid"))
    row = researchers_repo.find_person(db, int(user["id"]), "prosopia", slug, rid if looks_like_orcid(rid) else None)
    return MyProsopia(
        profile=profile,
        researcher_id=int(row["id"]) if row else None,
        imported_at=row["imported_at"] if row else None,
    )


@router.post("/me/lease")
def claim_lease(
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> dict:
    """This tab asks for the account's right to change things, or says it
    still has it (every 30 s; a view-only tab asks every 15 s). ``editing``:
    whether it holds it now (FUNCTIONAL_SPEC W.A). However long the tab is
    left unused, it keeps the right (user 2026-11-05, second round); an
    unused tab's requests only stop renewing the sign-in (app.py)."""
    return {"editing": write_lease.claim(db, int(user["id"]))}


@router.post("/me/lease/release")
def release_lease(
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    grace: bool = False,
) -> dict:
    """This tab lets go. Signing out: the right is free at once.
    A page closing or reloading (``?grace=1``): free after a few seconds, so
    the same tab reloaded keeps it (FC-1). A tab that does not hold it
    changes nothing."""
    write_lease.release(db, int(user["id"]), grace=grace)
    return {"ok": True}
