"""Users routes — Phase 12.

Two endpoints:

  GET   /api/users/me   → ``User`` for the authenticated email.
  PATCH /api/users/me   → ``User`` (after applying ``mailto`` change).

The acting user is always resolved by ``get_current_user`` from the
``X-User-Email`` header; this router never accepts a user id in the
URL. There is no admin surface and no per-user list endpoint — Phase 12
is intentionally minimal.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ..deps import get_current_user, get_db
from ..schemas import User, UserUpdateRequest

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
