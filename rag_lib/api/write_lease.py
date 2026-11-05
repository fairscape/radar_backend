"""One tab at a time changes an account (FUNCTIONAL_SPEC W.A, user 2026-11-04:
"两个页面可以同时登录，只是不能同时修改，第一个页面锁住修改的权限，第二个页面只能看已有内容").

An account has one right to change things, held by one tab -- of any browser,
on any device (``write_leases``, migration 0042). The tab that opens the site
first claims it (``claim``) and says every 30 s it is still there; it lets go
when it leaves (``release``: closed, reloaded, signed out). Every other tab is
view-only: each request that would change something is refused with
``EditingElsewhere`` (409, ``editingElsewhere``) by one guard on every API
router (``guard``), so no route can be left out. The guard only checks: the
right is taken and kept only by asking for it (``claim``, as a page opens
and every 30 s), never by a change. A holder not heard from for ``STALE_AFTER_S``
(2 minutes, as the draft and edit locks) has left; the next tab to ask takes
it over. There is no "take over" (user 11-04), and an open tab keeps the
right however long it is left unused: only closing it or signing out lets go
(user 2026-11-05, second round). A reload keeps the right (``RELEASE_GRACE_S``).

Reads are never refused, nor the three POSTs that only read or tidy up
(``EXEMPT``). Work outside a request (a scan, an import) has no tab and is
never refused: the tab that started it held the right then.

Which tab a request comes from is the ``X-Radar-Tab`` header, read by
``draft_lock.request_tab`` for every request. A request without it is a tab
of its own (""), refused like any other while a tab holds the account.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Annotated

import structlog
from fastapi import Depends, Request

from . import draft_lock
from .deps import get_current_user, get_db
from .settings import Settings, get_settings

log = structlog.get_logger("rag_lib.api.write_lease")

# As the draft and edit locks; the page says it is there every 30 s.
STALE_AFTER_S = 120

# A page that is closing or reloading lets go with ``grace``: for this long
# it is still the holder, so the same tab reloaded takes it straight back
# (user 2026-11-05: "刷新不算离开", review FC-1), and its own last requests
# (Edit interest's discard-edit) still land after the release (BL-2). A
# closed tab's right then reaches the next tab after these seconds.
RELEASE_GRACE_S = 5

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

# POSTs a view-only tab still needs (W.A R3), and the lease's own routes.
EXEMPT = tuple(re.compile(p) for p in (
    r"^/api/users/me/lease(?:/release)?$",
    # An interest's dry-run scores (the Threshold tab's chart): reads only.
    # Not a draft's dry run (/draft/<slug>/dry-run), which starts a job.
    r"^/api/profiles/(?!draft/|drafts/)[^/]+/dry-run$",
    # This tab's own left-behind edit copies, and leaving a draft: tidying
    # up after itself, which a view-only tab must be able to do too.
    r"^/api/profiles/drafts/discard-edits$",
    r"^/api/profiles/draft/[^/]+/release$",
))

# A page keeps what it has open -- its new interest's draft, its Edit
# interest copy -- even once the tab has lost the right to change things
# (asleep past 2 minutes, closing after its release). Reviews FC-2, BL-2,
# R1, R2; user 2026-11-05: "草稿如果在编辑，读取，写入期间，应该不允许其他的
# 页面resume". So these pass for a draft or copy this tab holds
# (``edit_tab``), and only for that one:
# - saying it is still there (draft hold, copy alive), so no other tab
#   resumes the draft or drops the copy while the page is open;
# - leaving it (draft release is exempt already; dropping its own copy), so
#   the tab that has the right can edit that interest at once.
# A view-only tab still cannot hold, keep or drop another tab's, or change
# anything in it.
_OWN = (
    ("POST", re.compile(r"^/api/profiles/draft/([^/]+)/hold$"), False),
    ("POST", re.compile(r"^/api/profiles/draft/([^/]+)/edit-alive$"), True),
    ("POST", re.compile(r"^/api/profiles/draft/([^/]+)/discard-edit$"), True),
    ("DELETE", re.compile(r"^/api/profiles/draft/([^/]+)$"), True),
)


def _keeps_its_own(conn: sqlite3.Connection, user_id: int, method: str, path: str) -> bool:
    tab = draft_lock._tab.get()
    if tab is None:
        return False
    for m, pat, copy_only in _OWN:
        hit = pat.match(path) if method == m else None
        if hit is None:
            continue
        from urllib.parse import unquote

        row = conn.execute(
            "SELECT 1 FROM profiles WHERE user_id = ? AND slug = ? AND is_draft = 1"
            " AND COALESCE(edit_tab, '') = ? AND COALESCE(edit_tab, '') != ''"
            + (" AND edit_of IS NOT NULL" if copy_only else " AND edit_of IS NULL"),
            (int(user_id), unquote(hit.group(1)), tab or "")).fetchone()
        return row is not None
    return False


class EditingElsewhere(RuntimeError):
    """Another tab or device holds this account's right to change things."""

    def __init__(self) -> None:
        super().__init__("This account is being changed in another tab or on another device; this tab can only "
                         "view. It can change things once that one is closed or signed out (within 2 minutes "
                         "if it was closed without signing out).")


def _holder(conn: sqlite3.Connection, user_id: int):
    """The tab holding ``user_id``'s lease, or None when nobody does (never held, released, or stale)."""
    row = conn.execute(
        "SELECT tab, seen_at >= datetime('now', ?) AS fresh FROM write_leases WHERE user_id = ?",
        (f"-{STALE_AFTER_S} seconds", int(user_id))).fetchone()
    if row is None or not row["fresh"]:
        return None
    return row["tab"]


def claim(conn: sqlite3.Connection, user_id: int, tab: str | None = None) -> bool:
    """Take or keep ``user_id``'s lease for ``tab`` (default: this request's).
    True when it holds it now; False while another tab does."""
    tab = draft_lock._tab.get() if tab is None else tab
    tab = tab or ""
    if conn.in_transaction:
        conn.commit()
    conn.execute("BEGIN IMMEDIATE")      # two tabs asking at once: one gets it
    try:
        holder = _holder(conn, user_id)
        if holder is not None and holder != tab:
            conn.rollback()
            return False
        conn.execute(
            "INSERT INTO write_leases (user_id, tab, seen_at) VALUES (?, ?, datetime('now'))"
            " ON CONFLICT(user_id) DO UPDATE SET tab = excluded.tab, seen_at = excluded.seen_at",
            (int(user_id), tab))
        conn.commit()
        if holder is None:
            log.info("write_lease.claimed", user_id=int(user_id))
        return True
    except BaseException:
        conn.rollback()
        raise


def release(conn: sqlite3.Connection, user_id: int, tab: str | None = None, *, grace: bool = False) -> None:
    """``tab`` (default: this request's) lets go of ``user_id``'s lease. A tab
    that does not hold it changes nothing.

    ``grace`` (a page closing or reloading): the tab stays the holder for
    ``RELEASE_GRACE_S`` more seconds -- its last-seen time is set back so the
    lease goes stale then -- so a reload keeps the right. Without it (sign
    out) the right is free at once."""
    tab = draft_lock._tab.get() if tab is None else tab
    if grace:
        conn.execute(
            "UPDATE write_leases SET seen_at = datetime('now', ?) WHERE user_id = ? AND tab = ?"
            " AND seen_at > datetime('now', ?)",
            (f"-{STALE_AFTER_S - RELEASE_GRACE_S} seconds", int(user_id), tab or "",
             f"-{STALE_AFTER_S - RELEASE_GRACE_S} seconds"))
    else:
        conn.execute("DELETE FROM write_leases WHERE user_id = ? AND tab = ?", (int(user_id), tab or ""))
    conn.commit()


def check(conn: sqlite3.Connection, user_id: int) -> None:
    """Refuse a change from this request's tab while another tab holds the
    account. Only checks: a change never takes the lease (user 11-04: the
    right goes by the order the pages opened, through ``claim`` alone), nor
    renews it: only ``claim`` does. Work outside a request passes.

    Why not "whoever changes something while nobody holds it takes it": a
    closing page's last request (Edit interest's discard-edit) reached the
    server after its release and took the right back, and a request without
    a tab header held it for 2 minutes (review WA-R1, WA-R5)."""
    tab = draft_lock._tab.get()
    if tab is None:
        return
    tab = tab or ""
    holder = _holder(conn, user_id)
    if holder is None:
        return
    if holder != tab:
        log.info("write_lease.refused", user_id=int(user_id))
        raise EditingElsewhere()
    # A change does not renew the lease: only ``claim`` does. Otherwise a
    # closing page's last request, inside the release grace, would hold the
    # right for 2 more minutes (WA-R1, FC-1).


def guard(
    request: Request,
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """Router dependency on every /api router that changes things. Who is
    asking is resolved here only for a change, the way the route itself
    does: a read stays exactly as it was."""
    if request.method.upper() in SAFE_METHODS:
        return
    path = request.url.path
    if any(p.match(path) for p in EXEMPT):
        return
    user = get_current_user(request, db, settings, request.headers.get("X-User-Email"))
    if _keeps_its_own(db, int(user["id"]), request.method.upper(), path):
        return
    check(db, int(user["id"]))
