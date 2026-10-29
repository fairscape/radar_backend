"""Users routes — Phase 12.

Three endpoints:

  GET   /api/users/me            → ``User`` for the authenticated email.
  PATCH /api/users/me            → ``User`` (after applying ``mailto`` change).
  GET   /api/users/me/prosopia   → ``MyProsopia``: the person's self
                                   profile on Prosopia, if public.

The acting user is always resolved by ``get_current_user`` from the
``X-User-Email`` header; this router never accepts a user id in the
URL. There is no admin surface and no per-user list endpoint — Phase 12
is intentionally minimal.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated, Any
from urllib.parse import quote

import httpx
import structlog
from fastapi import APIRouter, Cookie, Depends, HTTPException, status

from rag_lib.db.repos import researchers as researchers_repo

from .. import oidc
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
    bioxplorer_session: Annotated[str | None, Cookie()] = None,
) -> MyProsopia:
    """The person's self profile on Prosopia, if it is public.

    Keyed on the ``self_rid`` claim from sign-in (scope ``profiles:own``):
    the rid of the profile the Prosopia account says is about itself. No
    session, no ``self_rid``, or no public profile all answer 200 with
    nulls; the frontend then simply shows no card.
    """
    session = oidc.read_session(settings, bioxplorer_session) if oidc.providers(settings) else None
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
    row = researchers_repo.get_by_key(db, int(user["id"]), "prosopia", slug)
    return MyProsopia(
        profile=profile,
        researcher_id=int(row["id"]) if row else None,
        imported_at=row["imported_at"] if row else None,
    )
