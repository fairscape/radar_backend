"""What a draft interest brought into the database (FUNCTIONAL_SPEC D).

A draft is a sandbox. A PDF uploaded into it (its version and, when
OpenAlex matched it, the public record if that was new), a researcher its
import created and the papers that import stored are recorded here as
the draft's. Discarding the draft -- deleting it, "start over", an import
that failed -- removes them with it; saving it clears the record, and they
stay like anything else in the database.

What the draft only used is never the draft's: a paper or researcher
already in the database, a file uploaded before. And an item the draft
brought in is kept when something else has come to use it meanwhile:
another interest's seed, a feed row, feedback, a researcher's paper list,
another file of the same work, an interest linked to the researcher.

Only ``own`` and ``clear`` are called outside this module; neither commits.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import structlog

log = structlog.get_logger("rag_lib.db.repos.draft_owned")


def own(conn: sqlite3.Connection, profile_id: int, kind: str, item: str | int) -> None:
    """Record ``item`` as brought in by draft ``profile_id``; nothing when it
    is not (or no longer) a draft. Not committed."""
    conn.execute(
        """INSERT OR IGNORE INTO draft_owned (profile_id, kind, item)
           SELECT ?, ?, ? WHERE EXISTS (SELECT 1 FROM profiles WHERE id = ? AND is_draft = 1)""",
        (int(profile_id), kind, str(item), int(profile_id)),
    )


def clear(conn: sqlite3.Connection, profile_id: int) -> None:
    """The draft is saved: what it brought in stays. Not committed."""
    conn.execute("DELETE FROM draft_owned WHERE profile_id = ?", (int(profile_id),))


def disown(conn: sqlite3.Connection, kind: str, item: str | int) -> None:
    """``item`` is now wanted for itself (a researcher imported again from
    the Profiles page): no draft's discard removes it. Not committed."""
    conn.execute("DELETE FROM draft_owned WHERE kind = ? AND item = ?", (kind, str(item)))


def owned(conn: sqlite3.Connection, profile_id: int) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {"paper": [], "researcher": []}
    for kind, item in conn.execute(
        "SELECT kind, item FROM draft_owned WHERE profile_id = ? ORDER BY added_at, rowid", (int(profile_id),)
    ):
        out[kind].append(item)
    return out


_PAPER_USES = (
    "SELECT 1 FROM profile_seeds WHERE openalex_id = :id",
    "SELECT 1 FROM profile_candidates WHERE openalex_id = :id",
    "SELECT 1 FROM feedback_events WHERE openalex_id = :id",
    "SELECT 1 FROM researcher_papers WHERE openalex_id = :id",
    "SELECT 1 FROM paper_versions WHERE canonical_openalex_id = :id AND version_id != :id",
    "SELECT 1 FROM draft_owned WHERE kind = 'paper' AND item = :id",
)


def _paper_in_use(conn: sqlite3.Connection, openalex_id: str) -> bool:
    return any(conn.execute(sql, {"id": openalex_id}).fetchone() for sql in _PAPER_USES)


def _researcher_in_use(conn: sqlite3.Connection, researcher_id: int) -> bool:
    return any(conn.execute(sql, (researcher_id,)).fetchone() for sql in (
        "SELECT 1 FROM profiles WHERE researcher_id = ?",
        "SELECT 1 FROM draft_owned WHERE kind = 'researcher' AND item = CAST(? AS TEXT)",
    ))


def discard(conn: sqlite3.Connection, profile_id: int) -> dict[str, list[str]]:
    """Delete draft ``profile_id`` and what it brought in that nothing else uses.

    One transaction, so a failure leaves the draft and all of it as they
    were. The files of deleted uploads are removed after the commit.
    Returns what was deleted: ``{"paper": [...], "researcher": [...], "files": [...]}``.
    """
    profile_id = int(profile_id)
    if not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")
    try:
        items = owned(conn, profile_id)
        # profiles_repo.delete, uncommitted: its seeds, feed, feedback and
        # this draft's record (draft_owned) go by cascade.
        conn.execute("UPDATE gather_runs SET profile_id = NULL WHERE profile_id = ? AND researcher_id IS NOT NULL",
                     (profile_id,))
        conn.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))

        gone: dict[str, list[str]] = {"paper": [], "researcher": [], "files": []}
        for rid in items["researcher"]:
            if not _researcher_in_use(conn, int(rid)):
                # Its paper links and its own runs (the import's, detached
                # from the draft just above) go by cascade.
                conn.execute("DELETE FROM researchers WHERE id = ?", (int(rid),))
                gone["researcher"].append(rid)
        # Versions before the public records: a version is the use that
        # keeps its public record.
        for oa in sorted(items["paper"], key=lambda i: not i.startswith("local:upload:")):
            if _paper_in_use(conn, oa):
                continue
            paths = {r[0] for r in conn.execute(
                "SELECT local_path FROM papers WHERE openalex_id = ? AND local_path IS NOT NULL "
                "UNION SELECT local_path FROM paper_uploads WHERE openalex_id = ? AND local_path IS NOT NULL",
                (oa, oa))}
            # Its vectors, upload row, version record and embed queue entry
            # go by cascade.
            conn.execute("DELETE FROM papers WHERE openalex_id = ?", (oa,))
            gone["paper"].append(oa)
            gone["files"].extend(sorted(paths))
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    for path in gone["files"]:
        still = conn.execute(
            "SELECT 1 FROM papers WHERE local_path = ? UNION SELECT 1 FROM paper_uploads WHERE local_path = ?",
            (path, path)).fetchone()
        if still is None:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError as exc:
                log.warning("draft_owned.file_left", path=path, reason=str(exc)[:160])
    if gone["paper"] or gone["researcher"]:
        log.info("draft_owned.discarded", profile_id=profile_id,
                 n_papers=len(gone["paper"]), n_researchers=len(gone["researcher"]))
    return gone
