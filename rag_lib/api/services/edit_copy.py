"""Editing a saved interest on a copy (FUNCTIONAL_SPEC E).

Opening Edit interest copies the interest into a draft row that points at
it (``profiles.edit_of``). Every step of the edit page changes the copy;
the interest, its feed and its scans are left alone. Save changes writes
the copy's seeds, topics, threshold, years, researcher and research topics
back onto the interest in one transaction and deletes the copy. Leaving
without saving discards the copy with what it brought in that nothing
else uses (``draft_owned``, as a new interest's draft).

What the save does not do: scan, refit or rescore. The recommendations
stay as the last scan left them, and the edit page says they no longer
match until the user scans (E.4 Q5); the scan refits on the new seeds
(``jobs._refit_if_fit_stale``) and scores the whole pool again.

One tab at a time, first come first served (E.7 R5): the copy belongs to
the tab that opened it (``edit_tab``), which says every 30 s it is still
there (``alive``). Edit interest in another tab is refused (``BeingEdited``)
while the copy is heard from; a copy not heard from for ``STALE_AFTER_S``
(the tab closed without leaving, the computer slept) is left behind and
is discarded the next time anything looks for it.

The copy is a draft, so everything that keys on drafts treats it as one:
not listed, scanned or scored; an upload into it is the draft's. Only
its topic list follows the saved interest's rules (``wizard.is_new_draft``).
"""

from __future__ import annotations

import json
import secrets
import sqlite3

import structlog

from ...db.repos import (
    draft_owned as draft_owned_repo,
    embeddings as embeddings_repo,
    gather_runs as gather_runs_repo,
    profiles as profiles_repo,
    researchers as researchers_repo,
)
from ...openalex_tiers import enabled_topic_ids
from . import wizard as wizard_service

log = structlog.get_logger("rag_lib.api.services.edit_copy")

# Copied onto the copy, and the user's part of them written back on save.
# The fit (centroid, selector config, coherence, seed band) is copied for
# the steps that read it and never written back: the scan refits.
_COPIED = (
    "embedding_model", "centroid", "threshold", "topic_filters_json", "selector_config_json",
    "gatherer_config_json", "coherence_median", "coherence_iqr", "coherence_bimodal", "n_seed",
    "reranker_config_json", "seed_sim_min", "seed_sim_median", "seed_sim_max", "researcher_id",
    "interest_terms_json", "year_from", "year_to", "researcher_doc_json",
)
_SAVED = ("threshold", "topic_filters_json", "interest_terms_json", "researcher_id",
          "researcher_doc_json", "year_from", "year_to")


# A copy not heard from for this long is left behind (E.7 R5, user's Q1).
STALE_AFTER_S = 120

CLOSED = ("this edit is no longer open (this page went unheard for over 2 minutes, or the interest was "
          "deleted elsewhere); its changes were discarded. Open Edit interest again")


class BeingEdited(RuntimeError):
    """Edit interest refused: another tab or window has the interest open
    in Edit interest and is still there (E.7 R5, first come first served)."""

    def __init__(self, slug: str):
        super().__init__(f"'{slug}' is being edited in another tab or window. Save or leave the edit there "
                         "first; a tab closed without leaving frees it within 2 minutes.")
        self.slug = slug


def copy_of(conn: sqlite3.Connection, profile_id: int) -> sqlite3.Row | None:
    """The copy of interest ``profile_id`` open in Edit interest, if any,
    heard from or not."""
    return conn.execute("SELECT * FROM profiles WHERE edit_of = ? ORDER BY id LIMIT 1",
                        (int(profile_id),)).fetchone()


def is_stale(conn: sqlite3.Connection, row: sqlite3.Row) -> bool:
    """Whether copy ``row`` is left behind: not heard from for
    ``STALE_AFTER_S``, and no import is filling it."""
    if gather_runs_repo.has_unfinished_import(conn, int(row["id"])):
        return False
    seen = row["edit_seen_at"]
    if seen is None:
        return True
    return bool(conn.execute("SELECT ? < datetime('now', ?)", (seen, f"-{STALE_AFTER_S} seconds")).fetchone()[0])


def open_copy_of(conn: sqlite3.Connection, profile_id: int) -> sqlite3.Row | None:
    """The copy of interest ``profile_id`` someone is editing: one left
    behind goes first, so it does not hold the interest (its scans, its
    backfill) after the tab that had it is gone."""
    row = copy_of(conn, profile_id)
    if row is not None and is_stale(conn, row):
        draft_owned_repo.discard(conn, int(row["id"]))
        log.info("edit_copy.stale_discarded", copy_id=int(row["id"]), profile_id=int(profile_id))
        return copy_of(conn, profile_id)
    return row


def _require_copy(conn: sqlite3.Connection, user_id: int, slug: str) -> sqlite3.Row:
    row = profiles_repo.get_by_slug(conn, user_id, slug)
    if row is None or not wizard_service.is_edit_copy(row):
        # Discarded meanwhile -- left behind and found by another tab, or
        # its interest deleted. Said in the user's terms.
        raise LookupError(CLOSED)
    return row


def alive(conn: sqlite3.Connection, *, user_id: int, slug: str, tab: str) -> None:
    """The edit page of copy ``slug`` is still there: every 30 s, and when
    its tab comes back into view. ``LookupError`` when the copy is gone;
    ``BeingEdited`` when another tab has it (a duplicated tab)."""
    row = _require_copy(conn, user_id, slug)
    if (row["edit_tab"] or "") != (tab or ""):
        raise BeingEdited(row["slug"])
    conn.execute("UPDATE profiles SET edit_seen_at = datetime('now') WHERE id = ?", (int(row["id"]),))
    conn.commit()


def discard_leftovers(conn: sqlite3.Connection, user_id: int, tab: str = "") -> list[str]:
    """Discard the user's copies left behind: not heard from for 2 minutes,
    or this tab's own (it is not on an edit page now: closed or reloaded
    without leaving). Called when the site opens and when Edit interest
    opens. Another tab's copy still heard from stays (E.7 R5), as does one
    an import is still filling. Returns the slugs discarded."""
    gone: list[str] = []
    for row in conn.execute("SELECT * FROM profiles WHERE user_id = ? AND edit_of IS NOT NULL ORDER BY id",
                            (int(user_id),)).fetchall():
        if gather_runs_repo.has_unfinished_import(conn, int(row["id"])):
            log.info("edit_copy.leftover_importing", profile_id=int(row["id"]))
            continue
        if (row["edit_tab"] or "") != (tab or "") and not is_stale(conn, row):
            continue
        draft_owned_repo.discard(conn, int(row["id"]))
        gone.append(row["slug"])
    if gone:
        log.info("edit_copy.leftovers_discarded", user_id=user_id, n=len(gone))
    return gone


def open_copy(conn: sqlite3.Connection, *, user_id: int, slug: str, tab: str = "") -> sqlite3.Row:
    """Copy saved interest ``slug`` for Edit interest in tab ``tab``;
    returns the copy's row.

    The copies left behind go first (unsaved changes are not kept, E.4
    Q3), this tab's own among them (a reload). ``BeingEdited`` while
    another tab has the interest open and is heard from (first come first
    served, E.7 R5); ``LookupError`` for no such saved interest;
    ``wizard.ImportInProgress`` when a copy of it is still being filled
    by an import.
    """
    row = profiles_repo.get_by_slug(conn, user_id, slug)
    if row is None or row["is_draft"]:
        raise LookupError(f"no saved interest '{slug}'")
    discard_leftovers(conn, user_id, tab)
    cols = ", ".join(_COPIED)
    for _attempt in range(5):
        # A slug of its own each time: reused, a tab left on an earlier copy
        # (or a late discard of one) acted on the new copy under that slug.
        copy_slug, n = wizard_service._unique_slug(
            conn, user_id, f"{slug}-edit-{secrets.token_hex(4)}", f"{row['name']} (editing)")
        name = f"{row['name']} (editing)" if n == 0 else wizard_service._suffixed(f"{row['name']} (editing)", n)
        try:
            if conn.in_transaction:
                conn.commit()
            conn.execute("BEGIN IMMEDIATE")
            # Under the lock: a copy of the interest still there (another
            # tab's, heard from) or made since the leftovers went (another
            # tab opening at the same moment) has it first, and this open is
            # refused. This tab's own, or one left behind, goes. One an
            # import still fills stays, and this open is refused.
            left = copy_of(conn, int(row["id"]))
            if left is not None:
                conn.rollback()
                # Another tab's first: said as that, import or not.
                if (left["edit_tab"] or "") != (tab or "") and not is_stale(conn, left):
                    raise BeingEdited(slug)
                if gather_runs_repo.has_unfinished_import(conn, int(left["id"])):
                    raise wizard_service.ImportInProgress(left["slug"])
                draft_owned_repo.discard(conn, int(left["id"]))     # its own transaction; then again
                continue
            cur = conn.execute(
                f"INSERT INTO profiles (user_id, name, slug, is_draft, edit_of, edit_tab, edit_seen_at, {cols})"
                f" SELECT user_id, ?, ?, 1, id, ?, datetime('now'), {cols} FROM profiles WHERE id = ? AND is_draft = 0",
                (name, copy_slug, tab or "", int(row["id"])),
            )
            if cur.rowcount == 0:
                conn.rollback()
                raise LookupError(f"no saved interest '{slug}'")    # deleted meanwhile
            copy_id = int(cur.lastrowid)
            conn.execute(
                "INSERT INTO profile_seeds (profile_id, openalex_id, added_at, source, source_researcher_id)"
                " SELECT ?, openalex_id, added_at, source, source_researcher_id FROM profile_seeds WHERE profile_id = ?",
                (copy_id, int(row["id"])),
            )
            conn.commit()
            break
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            # The slug taken meanwhile, or (the unique edit_of index) a copy
            # made meanwhile: try again, which refuses it or discards it.
            if "UNIQUE" not in str(exc):
                raise
    else:
        raise ValueError(f"could not open '{slug}' for editing")
    log.info("edit_copy.opened", profile_id=int(row["id"]), copy_id=copy_id)
    return profiles_repo.get(conn, copy_id)


def discard_copy(conn: sqlite3.Connection, *, user_id: int, slug: str) -> None:
    """Leave Edit interest without saving: the copy goes, with what it
    brought in that nothing else uses; the interest is as it was."""
    row = _require_copy(conn, user_id, slug)
    if gather_runs_repo.has_unfinished_import(conn, int(row["id"])):
        raise wizard_service.ImportInProgress(slug)
    draft_owned_repo.discard(conn, int(row["id"]))
    log.info("edit_copy.discarded", copy_id=int(row["id"]), profile_id=int(row["edit_of"]))


def save_copy(conn: sqlite3.Connection, *, user_id: int, slug: str) -> str:
    """Save changes: write the copy onto the interest it copies, delete the
    copy, and return the interest's slug. No scan, refit or rescore (E.4
    Q5) -- except that a new threshold or year range re-decides which
    stored papers are in the feed, as changing them always has (E.6 R10).

    Refused (nothing written) with ``wizard.ImportInProgress`` while an
    import fills the copy, ``wizard.ScanRunning`` while a scan of the
    interest runs (it would record the old topics and seeds as the ones
    it searched, E.6 R11), ``wizard.NoTopicOn`` when no topic would be
    on, and ``ValueError`` when no seed is left that can be placed.
    """
    copy = _require_copy(conn, user_id, slug)
    copy_id, profile_id = int(copy["id"]), int(copy["edit_of"])
    original = profiles_repo.get(conn, profile_id)
    if original is None or original["is_draft"] or int(original["user_id"]) != int(user_id):
        raise LookupError(f"the interest '{slug}' was copied from is gone")
    if gather_runs_repo.has_unfinished_import(conn, copy_id):
        raise wizard_service.ImportInProgress(slug)
    if gather_runs_repo.running_scan(conn, profile_id) is not None:
        raise wizard_service.ScanRunning(
            f"a scan of '{original['slug']}' is still running; save when it has finished")
    seeds = profiles_repo.list_seed_openalex_ids(conn, copy_id)
    if not seeds:
        raise ValueError("an interest needs at least one seed; add one before saving")
    wizard_service.embed_missing(conn, copy["embedding_model"], seeds)
    if not any(embeddings_repo.has(conn, s, copy["embedding_model"]) for s in seeds):
        raise ValueError("none of the seeds can be placed (no vector); add one that can before saving")
    topics = (profiles_repo.topic_filters(conn, copy_id) or {}).get("topics") or []
    if topics and not enabled_topic_ids({"topics": topics}):
        raise wizard_service.NoTopicOn("switch on at least one topic: a scan searches by them")

    seeds_before = sorted(profiles_repo.list_seed_openalex_ids(conn, profile_id))
    if conn.in_transaction:
        conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        copy = conn.execute("SELECT * FROM profiles WHERE id = ?", (copy_id,)).fetchone()
        if copy is None:
            raise LookupError(CLOSED)
        researcher_id = copy["researcher_id"]
        # Deleted while the copy was open: the link goes, the copy of the
        # profile it took stays (E.6 R9).
        if researcher_id is not None and researchers_repo.get(conn, int(researcher_id)) is None:
            researcher_id = None
        values = {k: copy[k] for k in _SAVED}
        values["researcher_id"] = researcher_id
        conn.execute(
            "UPDATE profiles SET " + ", ".join(f"{k} = ?" for k in _SAVED) + ","
            " n_seed = (SELECT COUNT(*) FROM profile_seeds WHERE profile_id = ?), updated_at = datetime('now')"
            " WHERE id = ? AND is_draft = 0",
            (*[values[k] for k in _SAVED], copy_id, profile_id),
        )
        conn.execute("DELETE FROM profile_seeds WHERE profile_id = ?", (profile_id,))
        conn.execute(
            "INSERT INTO profile_seeds (profile_id, openalex_id, added_at, source, source_researcher_id)"
            " SELECT ?, openalex_id, added_at, source, source_researcher_id FROM profile_seeds WHERE profile_id = ?",
            (profile_id, copy_id),
        )
        # What the copy brought in is the interest's now: no discard removes it.
        draft_owned_repo.clear(conn, copy_id)
        # An import into the copy is the researcher's import history.
        conn.execute("UPDATE gather_runs SET profile_id = NULL WHERE profile_id = ? AND researcher_id IS NOT NULL",
                     (copy_id,))
        conn.execute("DELETE FROM profiles WHERE id = ?", (copy_id,))
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    if (copy["threshold"] != original["threshold"] or copy["year_from"] != original["year_from"]
            or copy["year_to"] != original["year_to"]):
        from ...scheduler.jobs import rescore_pool

        # Saved by now: a failure here must not answer "not saved" (a retry
        # then found no copy). The next scan rescores the pool anyway.
        try:
            rescore_pool(conn, profile_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("edit_copy.rescore_after_save_failed", profile_id=profile_id, reason=str(exc)[:160])
    log.info("edit_copy.saved", profile_id=profile_id, copy_id=copy_id,
             seeds_changed=sorted(seeds) != seeds_before)
    return original["slug"]


def changed(conn: sqlite3.Connection, *, user_id: int, slug: str) -> bool:
    """Whether the copy differs from the interest it copies: what the edit
    page asks before letting the user leave."""
    copy = _require_copy(conn, user_id, slug)
    original = profiles_repo.get(conn, int(copy["edit_of"]))
    if original is None:
        return False
    if any(_norm(k, copy[k]) != _norm(k, original[k]) for k in _SAVED):
        return True
    return (sorted(profiles_repo.list_seed_openalex_ids(conn, int(copy["id"])))
            != sorted(profiles_repo.list_seed_openalex_ids(conn, int(original["id"]))))


def _norm(key: str, value):
    """JSON columns compared as data, not text."""
    if key.endswith("_json") and isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value
