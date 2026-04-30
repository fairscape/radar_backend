"""FastAPI dependencies.

Three dependencies the rest of the service hangs off:

  - ``get_settings`` — process-wide ``Settings`` singleton.
  - ``get_db`` — per-request SQLite connection (WAL + FK pragmas applied
    by ``rag_lib.db.connect``); closed on request teardown.
  - ``get_current_user`` — Phase 12 reads ``X-User-Email`` from the
    request and upserts a row in ``users``. Falls back to user 1
    (``demo@example.com``) when the header is absent and
    ``RADAR_REQUIRE_AUTH`` is False; 401s when the header is absent and
    auth is required.

Identity is email-only: there are no tokens or passwords. The header is
trusted — the surface is the same as a public API behind a reverse
proxy, which is the demo's threat model. CSRF is N/A (no cookies).
"""

from __future__ import annotations

import re
import sqlite3
from typing import Annotated, Iterator

from fastapi import Depends, Header, HTTPException, Request, status

from rag_lib.db import connect
from rag_lib.db.repos import users

from .settings import Settings, get_settings


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def get_db(
    settings: Annotated[Settings, Depends(get_settings)],
) -> Iterator[sqlite3.Connection]:
    conn = connect(settings.RADAR_DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def get_scheduler(request: Request):
    """Return the live ``BackgroundScheduler`` from app state.

    503s when the scheduler is disabled (``RADAR_SCHEDULER_ENABLED=false``)
    so callers of ``gather-now`` / schedule-update endpoints get an
    actionable error instead of a vague NoneType failure.
    """
    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="scheduler disabled (set RADAR_SCHEDULER_ENABLED=true)",
        )
    return scheduler


def get_current_user(
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    x_user_email: Annotated[str | None, Header(alias="X-User-Email")] = None,
) -> sqlite3.Row:
    """Resolve the acting user from ``X-User-Email``.

    Behavior:

    - Header present → lowercase, validate it looks like an email,
      ``users.upsert(email)`` to ensure the row exists, return it.
      ``upsert`` defaults ``mailto = email`` on first insert.
    - Header absent + ``RADAR_REQUIRE_AUTH=False`` → fall back to the
      demo user (id 1). 500s if that row is missing (migrations
      not run / DB wiped without re-init).
    - Header absent + ``RADAR_REQUIRE_AUTH=True`` → 401.

    The trust model: the header is the identity. There's no signature
    or session — the demo's threat surface is "anyone who can reach the
    backend can claim any email". Phase 13's docker-compose puts the
    backend behind localhost only; production deployments must add a
    real auth proxy on top.
    """
    if x_user_email:
        email = x_user_email.strip().lower()
        if not _EMAIL_RE.match(email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"X-User-Email is not a valid email: {x_user_email!r}",
            )
        return users.upsert(db, email)

    if settings.RADAR_REQUIRE_AUTH:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-User-Email header is required",
        )

    row = users.get(db, 1)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="default user (id=1) missing — run `python -m cli.db init`",
        )
    return row
