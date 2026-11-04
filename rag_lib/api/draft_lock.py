"""One tab at a time on a new interest's draft (FUNCTIONAL_SPEC W.L, user
2026-11-04: "只要一个草稿被一个标签页先resume了，就跟edit interest一样，其他的操作一概被锁，连resume都不行").

As Edit interest's copy (E.7 R5), first come first served: the wizard tab
that opens a draft (new or resumed) holds it (``hold``), says every 30 s it
is still there, and lets go when it leaves (``release``). While it is heard
from, every other request that would change the draft is refused with
``DraftOpenElsewhere`` -- another tab's Resume included. A holder not heard
from for ``STALE_AFTER_S`` (2 minutes, as E.7) has left it behind and no
longer holds it. Unlike an edit copy, the draft itself stays: leaving the
wizard keeps it to resume later.

The holder is kept in the columns Edit interest uses for its copy's tab
(``profiles.edit_tab``, ``edit_seen_at``, migration 0034); on a new
interest's draft (``edit_of`` NULL) nothing else reads them.

Which tab a request comes from is the ``X-Radar-Tab`` header the page sends
on every request (``request_tab``, set for the request in ``_tab``). Work
outside a request -- an import or a scan job filling the draft -- has no
tab and is never refused: the tab that started it holds the draft.
"""

from __future__ import annotations

import sqlite3
from contextvars import ContextVar

import structlog
from fastapi import Request

log = structlog.get_logger("rag_lib.api.draft_lock")

TAB_HEADER = "X-Radar-Tab"
# As Edit interest (E.7 R5, the user's Q1); the page says it is there every 30 s.
STALE_AFTER_S = 120

# The tab of the request being served; None outside a request.
_tab: ContextVar[str | None] = ContextVar("radar_tab", default=None)


class DraftOpenElsewhere(RuntimeError):
    """The draft is open in another tab or window that is still there."""

    def __init__(self, name: str):
        super().__init__(f"'{name}' is open in another tab or window. Finish or leave it there first; "
                         "a tab closed without leaving frees it within 2 minutes.")
        self.name = name


async def request_tab(request: Request) -> None:
    """App-wide dependency: the request's tab, for ``check``. A request with
    no header is a tab of its own ("") -- refused like any other while a
    tab holds the draft."""
    _tab.set((request.headers.get(TAB_HEADER) or "").strip()[:100])


def _is_new_draft(row) -> bool:
    return row is not None and bool(row["is_draft"]) and row["edit_of"] is None


def _held_by_other(conn: sqlite3.Connection, row, tab: str) -> bool:
    holder = row["edit_tab"] or ""
    if not holder or holder == tab:
        return False
    seen = row["edit_seen_at"]
    if seen is None:
        return False
    return not conn.execute("SELECT ? < datetime('now', ?)", (seen, f"-{STALE_AFTER_S} seconds")).fetchone()[0]


def check(conn: sqlite3.Connection, row) -> None:
    """Refuse a change to draft ``row`` from this request's tab while
    another tab holds it. Anything but a new interest's draft, and work
    outside a request, pass."""
    tab = _tab.get()
    if tab is None or not _is_new_draft(row):
        return
    if _held_by_other(conn, row, tab):
        log.info("draft_lock.refused", profile_id=int(row["id"]))
        raise DraftOpenElsewhere(row["name"])


def check_id(conn: sqlite3.Connection, profile_id: int) -> None:
    check(conn, conn.execute("SELECT * FROM profiles WHERE id = ?", (int(profile_id),)).fetchone())


def check_slug(conn: sqlite3.Connection, user_id: int, slug: str | None) -> None:
    if slug:
        check(conn, conn.execute("SELECT * FROM profiles WHERE user_id = ? AND slug = ?",
                                 (int(user_id), slug)).fetchone())


def hold(conn: sqlite3.Connection, *, user_id: int, slug: str) -> None:
    """This request's tab opens draft ``slug``, or says it is still there:
    it holds the draft from now. ``DraftOpenElsewhere`` while another tab
    holds it; ``LookupError`` when it is no draft of a new interest (saved,
    or deleted, meanwhile)."""
    tab = _tab.get() or ""
    if conn.in_transaction:
        conn.commit()
    conn.execute("BEGIN IMMEDIATE")      # two tabs opening at once: one holds it
    try:
        row = conn.execute("SELECT * FROM profiles WHERE user_id = ? AND slug = ?", (int(user_id), slug)).fetchone()
        if not _is_new_draft(row):
            raise LookupError(f"draft '{slug}' not found")
        if _held_by_other(conn, row, tab):
            raise DraftOpenElsewhere(row["name"])
        conn.execute("UPDATE profiles SET edit_tab = ?, edit_seen_at = datetime('now') WHERE id = ?",
                     (tab, int(row["id"])))
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def release(conn: sqlite3.Connection, *, user_id: int, slug: str) -> None:
    """This request's tab leaves draft ``slug``: it is free at once. A tab
    that does not hold it changes nothing. ``LookupError`` when it is no
    draft of a new interest of this user (saved or deleted meanwhile, an
    Edit interest copy, or another user's), as ``hold``."""
    tab = _tab.get() or ""
    row = conn.execute("SELECT * FROM profiles WHERE user_id = ? AND slug = ?", (int(user_id), slug)).fetchone()
    if not _is_new_draft(row):
        raise LookupError(f"draft '{slug}' not found")
    conn.execute(
        "UPDATE profiles SET edit_tab = NULL, edit_seen_at = NULL"
        " WHERE user_id = ? AND slug = ? AND is_draft = 1 AND edit_of IS NULL AND edit_tab = ?",
        (int(user_id), slug, tab))
    conn.commit()
