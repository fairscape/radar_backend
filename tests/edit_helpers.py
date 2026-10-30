"""Changing a saved interest the way the site does: on a copy, saved in one
go (FUNCTIONAL_SPEC E). The routes that change an interest answer 409 for
a saved one; tests that set a saved interest up, or change it, go through
these.

Saving does not refit or rescore (E.4 Q5); a scan does. ``refit`` runs
what the scan would, for a test that needs the interest fitted on its new
seeds without a scan.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Callable, Iterator


@contextmanager
def edit_saved(call: Callable, slug: str, *, save: bool = True) -> Iterator[str]:
    """Open Edit interest on saved interest ``slug`` over the API; yields
    the copy's slug, which every change inside goes to; saves on the way
    out (or discards, ``save=False``). ``call(method, path, **kw)`` returns
    an httpx/TestClient response."""
    r = call("POST", f"/api/profiles/{slug}/edit")
    assert r.status_code == 200, r.text
    copy = r.json()["slug"]
    yield copy
    if save:
        r = call("POST", f"/api/profiles/draft/{copy}/save-edit")
        assert r.status_code == 200, r.text
    else:
        r = call("DELETE", f"/api/profiles/draft/{copy}")
        assert r.status_code == 200, r.text


@contextmanager
def edit_saved_db(conn: sqlite3.Connection, user_id: int, slug: str, *, save: bool = True) -> Iterator[str]:
    """``edit_saved`` through the services."""
    from rag_lib.api.services import edit_copy

    copy = edit_copy.open_copy(conn, user_id=user_id, slug=slug)["slug"]
    yield copy
    if save:
        edit_copy.save_copy(conn, user_id=user_id, slug=copy)
    else:
        edit_copy.discard_copy(conn, user_id=user_id, slug=copy)


def refit(conn: sqlite3.Connection, user_id: int, slug: str) -> None:
    """Fit the interest on its seeds and score its pool again: what the
    next scan does first (``jobs._refit_if_fit_stale``)."""
    from rag_lib.api.services import wizard

    wizard.refit_profile(conn, user_id=user_id, slug=slug)
