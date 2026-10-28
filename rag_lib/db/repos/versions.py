"""Each user's upload of a PDF is its own paper: a private version.

A version is a ``papers`` row under ``local:upload:<user_id>:<sha256>``
(migration 0029). It holds what was read from that user's file -- body
text, embeddings, UMLS concepts, content topics -- and a copy of the
public record's metadata when OpenAlex matched it. The public row
(``https://openalex.org/W...``) is bibliographic only: the same work
uploaded by two users, or twice by one user with different bytes, never
shares text or vectors.

``paper_versions`` records the owner, the file hash and the public work
the version is a copy of (``canonical_openalex_id``, NULL when OpenAlex
had nothing). ``parsed`` is 1 once the row holds this file's own text,
0 while it waits for a parse (a migrated upload whose text could not be
proven to be this file's), -1 when its file could not be read; such a
version is scored on its metadata, like an imported paper.
"""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger("rag_lib.db.repos.versions")

PREFIX = "local:upload:"


def version_id(user_id: int, file_hash: str) -> str:
    return f"{PREFIX}{int(user_id)}:{file_hash}"


def is_version(openalex_id: str | None) -> bool:
    return bool(openalex_id) and str(openalex_id).startswith(PREFIX)


def owner(conn: sqlite3.Connection, paper_id: str | None) -> int | None:
    """The user a version belongs to, as ``paper_versions`` records it.
    None for any other paper, and for a version id nobody registered."""
    row = conn.execute("SELECT user_id FROM paper_versions WHERE version_id = ?", (paper_id,)).fetchone()
    return int(row[0]) if row is not None else None


def register(
    conn: sqlite3.Connection, *, user_id: int, paper_id: str,
    canonical_id: str | None, file_hash: str, parsed: int = 1,
    title_from_file: bool = False,
) -> None:
    """Record ``paper_id`` as ``user_id``'s version. Not committed."""
    conn.execute(
        """
        INSERT INTO paper_versions (version_id, user_id, canonical_openalex_id, file_hash, parsed, title_from_file)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (version_id) DO UPDATE SET
          canonical_openalex_id = COALESCE(excluded.canonical_openalex_id, canonical_openalex_id),
          parsed = excluded.parsed,
          title_from_file = excluded.title_from_file
        """,
        (paper_id, int(user_id), canonical_id, file_hash, int(parsed), int(bool(title_from_file))),
    )


def _title_key(title: str | None) -> str:
    """A title as compared for "the same paper": case, accents, spacing and
    punctuation do not count."""
    decomposed = unicodedata.normalize("NFKD", title or "")
    ascii_ = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", ascii_.lower()).strip()


def version_numbers(conn: sqlite3.Connection, user_id: int) -> dict[str, tuple[int, str]]:
    """``{version_id: (n, uploaded_at)}`` for ``user_id``'s uploads that are
    one of two or more files of the same paper; n counts from 1 in upload
    order. Only names: nothing is merged or replaced.

    The same paper: the same OpenAlex work; or, for a file OpenAlex has no
    record of, the same title (``_title_key``) as another of the user's
    files -- not when that title is only the file's name (the PDF had none)
    or "Untitled PDF". Papers that came in any other way (ORCID, a search)
    are not uploads and take no number. Two files uploaded within the same
    second (``uploaded_at`` has no finer grain) are in the order they were
    registered.
    """
    rows = conn.execute(
        """
        SELECT v.version_id, v.canonical_openalex_id, v.title_from_file, p.title,
               COALESCE(u.uploaded_at, p.first_seen_at, '') AS at
        FROM paper_versions v
        JOIN papers p ON p.openalex_id = v.version_id
        LEFT JOIN paper_uploads u ON u.user_id = v.user_id AND u.openalex_id = v.version_id
        WHERE v.user_id = ?
        ORDER BY at, v.rowid
        """,
        (int(user_id),),
    ).fetchall()
    group_of: dict[str, str] = {}
    by_title: dict[str, str] = {}
    # Matched files first: an unmatched file with a matched one's title
    # joins that work's group.
    for vid, canonical, from_file, title, _at in sorted(rows, key=lambda r: r[1] is None):
        key = _title_key(title)
        named = bool(key) and not from_file and key != "untitled pdf"
        if canonical is not None:
            group_of[vid] = canonical
            if named:
                by_title.setdefault(key, canonical)
        elif named:
            group_of[vid] = by_title.setdefault(key, f"title:{key}")
        else:
            group_of[vid] = f"file:{vid}"
    members: dict[str, list[tuple[str, str]]] = {}
    for vid, _c, _f, _t, at in rows:
        members.setdefault(group_of[vid], []).append((vid, at))
    return {vid: (n, at) for group in members.values() if len(group) > 1
            for n, (vid, at) in enumerate(group, start=1)}


def canonical_id(conn: sqlite3.Connection, paper_id: str | None) -> str | None:
    """The public work ``paper_id`` is a version of, or None."""
    if not is_version(paper_id):
        return None
    row = conn.execute(
        "SELECT canonical_openalex_id FROM paper_versions WHERE version_id = ?", (paper_id,),
    ).fetchone()
    return row[0] if row else None


def canonical_ids(conn: sqlite3.Connection, paper_ids) -> dict[str, str]:
    """``{version_id: canonical}`` for those of ``paper_ids`` that have one."""
    ids = [p for p in paper_ids if is_version(p)]
    out: dict[str, str] = {}
    for i in range(0, len(ids), 500):
        part = ids[i:i + 500]
        out.update(
            (r[0], r[1]) for r in conn.execute(
                f"SELECT version_id, canonical_openalex_id FROM paper_versions "
                f"WHERE canonical_openalex_id IS NOT NULL AND version_id IN ({','.join('?' * len(part))})",
                part,
            )
        )
    return out


def has_own_text(conn: sqlite3.Connection, paper_id: str) -> bool:
    """False for a version still without its file's text (waiting, or its
    file was unreadable): uploading the file again reads it."""
    row = conn.execute("SELECT parsed FROM paper_versions WHERE version_id = ?", (paper_id,)).fetchone()
    return row is None or row[0] == 1


def is_unmatched(conn: sqlite3.Connection, paper_id: str | None) -> bool:
    """A paper OpenAlex has no record of: its topics come from its text.

    The ``local:`` prefix used to say this on its own; a version of a
    matched work is ``local:`` too and has OpenAlex's topics.
    """
    if not paper_id or not str(paper_id).startswith("local:"):
        return False
    return canonical_id(conn, paper_id) is None


def same_work_seeds(conn: sqlite3.Connection, profile_id: int, paper_id: str) -> list[str]:
    """The profile's seeds that are ``paper_id``'s work under another id:
    its public record, and every uploaded version of it (an earlier file,
    a revised one). An unmatched upload is a work of its own."""
    work = canonical_id(conn, paper_id) if is_version(paper_id) else paper_id
    if work is None:
        return []
    return [r[0] for r in conn.execute(
        """
        SELECT s.openalex_id FROM profile_seeds s
        LEFT JOIN paper_versions v ON v.version_id = s.openalex_id
        WHERE s.profile_id = ? AND s.openalex_id != ?
          AND (s.openalex_id = ? OR v.canonical_openalex_id = ?)
        ORDER BY s.openalex_id
        """,
        (profile_id, paper_id, work, work),
    )]


def _stale_fit(raw: str | None, ids: set[str]) -> str | None:
    """``raw`` fit record no longer matching seeds ``ids``, or None if it
    is unchanged. Drops them from the recorded fit seed ids, so they no
    longer match the seeds and a scan's start-of-run check refits
    (``jobs._refit_if_fit_stale``); a fit with no record is compared by
    seed count there, which would not see this, so its matrix goes."""
    from ...api.services.wizard import FIT_SEEDS

    try:
        cfg = json.loads(raw) if raw else None
    except ValueError:
        return None
    if not ids or not isinstance(cfg, dict):
        return None
    if isinstance(cfg.get(FIT_SEEDS), list):
        kept = [i for i in cfg[FIT_SEEDS] if i not in ids]
        if len(kept) == len(cfg[FIT_SEEDS]):
            return None
        cfg[FIT_SEEDS] = kept
    elif cfg.pop("seed_matrix", None) is None:
        return None
    return json.dumps(cfg)


def public_fields(paper: dict[str, Any]) -> dict[str, Any]:
    """What of an uploaded paper's record may go on the shared public row."""
    out = dict(paper)
    for key in ("body_text", "local_path", "file_hash", "uploaded_by_user_id", "n_pages"):
        out.pop(key, None)
    return out


def rename_ids(value: Any, renames: dict[str, str]) -> Any:
    """``value`` with every string equal to an old id replaced by its new
    one. Two old ids renamed to one new id leave it in a list once."""
    if isinstance(value, str):
        return renames.get(value, value)
    if isinstance(value, list):
        out = [rename_ids(v, renames) for v in value]
        if any(isinstance(v, str) and v in renames for v in value):
            seen: set = set()
            out = [v for v in out if not isinstance(v, str) or not (v in seen or seen.add(v))]
        return out
    if isinstance(value, dict):
        return {k: rename_ids(v, renames) for k, v in value.items()}
    return value


def parse_pending(conn: sqlite3.Connection, *, limit: int = 20) -> int:
    """Read the text of versions still waiting for it (``parsed = 0``).

    Only a migrated upload can be waiting: its stored text came from
    another user's file, or from a different file of the same user, so
    0029 kept the metadata and not the text. Runs from the periodic UMLS
    sweep, never inside a request. The file must still hash to the
    version's hash; a missing or changed file marks the version -1 and it
    stays scored on its metadata until the user uploads the PDF again.
    Interests using a version that gained text are left with a stale fit,
    so their next scan refits them. Returns how many were read.
    """
    from ...vault import compute_file_hash, ingest_pdf

    rows = conn.execute(
        """
        SELECT v.version_id, v.file_hash, u.local_path
        FROM paper_versions v
        LEFT JOIN paper_uploads u ON u.openalex_id = v.version_id AND u.user_id = v.user_id
        WHERE v.parsed = 0
        ORDER BY v.version_id LIMIT ?
        """,
        (int(limit),),
    ).fetchall()
    n = 0
    for vid, digest, path in rows:
        record = None
        try:
            if path and Path(path).is_file() and compute_file_hash(Path(path)) == digest:
                record = ingest_pdf(path)
        except Exception as exc:  # noqa: BLE001 -- pdfplumber raises many kinds
            log.warning("versions.parse_failed", version_id=vid, reason=str(exc)[:160])
        if conn.in_transaction:
            conn.commit()
        if record is None:
            log.warning("versions.file_unavailable", version_id=vid)
            conn.execute("UPDATE paper_versions SET parsed = -1 WHERE version_id = ? AND parsed = 0", (vid,))
            conn.commit()
            continue
        cur = conn.execute("UPDATE paper_versions SET parsed = 1 WHERE version_id = ? AND parsed = 0", (vid,))
        if cur.rowcount:
            # The title stays: it is OpenAlex's (or the upload's), and
            # ingest_pdf falls back to the file name, here the hash.
            conn.execute(
                """UPDATE papers SET body_text = ?, n_pages = COALESCE(?, n_pages),
                   umls_concepts_json = NULL, umls_mapped_topics_json = NULL,
                   topics_json = CASE WHEN json_valid(topics_json)
                     AND json_extract(topics_json, '$.source') = 'pdf_content'
                     THEN NULL ELSE topics_json END
                   WHERE openalex_id = ?""",
                (record.body_text, record.n_pages, vid),
            )
            conn.execute("DELETE FROM paper_embeddings WHERE openalex_id = ?", (vid,))
            forget_fit(conn, vid)
            n += 1
        conn.commit()
    return n


def forget_fit(conn: sqlite3.Connection, paper_id: str) -> None:
    """Make the fit of every interest seeded with ``paper_id`` stale.

    See ``_stale_fit``. Not committed.
    """
    for pid, raw in conn.execute(
        """SELECT p.id, p.selector_config_json FROM profiles p
           JOIN profile_seeds s ON s.profile_id = p.id WHERE s.openalex_id = ?""",
        (paper_id,),
    ).fetchall():
        cfg = _stale_fit(raw, {paper_id})
        if cfg is not None:
            conn.execute("UPDATE profiles SET selector_config_json = ? WHERE id = ?", (cfg, pid))


def embed_queued(conn: sqlite3.Connection, *, default_model: str, limit: int = 50) -> int:
    """Embed the public papers 0029 left without a vector (``papers_to_embed``)
    from their metadata, under each model that uses them: the model of
    every interest seeding or holding them as a candidate, and the default
    one for a researcher's papers (the interest suggestions). Runs from
    the periodic sweep. Returns how many vectors were made."""
    from ...api.services.wizard import embed_missing
    from ...scoring.rescore import rescore_candidates

    n = 0
    # Oldest attempt first: a paper whose rescore keeps failing goes to the
    # back, and the others are not stuck behind it.
    for (pid,) in conn.execute(
        "SELECT openalex_id FROM papers_to_embed ORDER BY attempted_at IS NOT NULL, attempted_at, openalex_id LIMIT ?",
        (int(limit),),
    ).fetchall():
        conn.execute("UPDATE papers_to_embed SET attempted_at = datetime('now') WHERE openalex_id = ?", (pid,))
        models = {r[0] for r in conn.execute(
            """SELECT pr.embedding_model FROM profile_candidates c JOIN profiles pr ON pr.id = c.profile_id
               WHERE c.openalex_id = :id
               UNION SELECT pr.embedding_model FROM profile_seeds s JOIN profiles pr ON pr.id = s.profile_id
               WHERE s.openalex_id = :id""", {"id": pid})}
        if conn.execute("SELECT 1 FROM researcher_papers WHERE openalex_id = ?", (pid,)).fetchone():
            models.add(default_model)
        for model in sorted(m for m in models if m):
            n += embed_missing(conn, model, [pid])
        conn.commit()
        # A candidate's score was made from the old vector: score it again.
        # Off the queue only once every interest holding it is rescored; a
        # failure leaves it there for the next sweep.
        try:
            for (prof,) in conn.execute(
                """SELECT DISTINCT c.profile_id FROM profile_candidates c JOIN profiles pr ON pr.id = c.profile_id
                   WHERE c.openalex_id = ? AND pr.is_draft = 0""", (pid,),
            ).fetchall():
                rescore_candidates(conn, int(prof))
        except Exception as exc:  # noqa: BLE001 -- retried next sweep
            if conn.in_transaction:
                conn.rollback()
            log.warning("versions.rescore_failed", openalex_id=pid, reason=str(exc)[:160])
            continue
        conn.execute("DELETE FROM papers_to_embed WHERE openalex_id = ?", (pid,))
        conn.commit()
    return n


# A version's bibliographic fields follow its public record. Not title or
# abstract: they are part of what the version's own vector was made from.
_FOLLOWED = ("doi", "year", "venue", "publication_date", "authors_json", "pdf_url", "oa_status")


def follow_public(conn: sqlite3.Connection, public_id: str) -> None:
    """Copy what the public record ``public_id`` now has of the followed
    fields onto the users' versions of it. Not committed."""
    if is_version(public_id) or str(public_id).startswith("local:"):
        return
    sets = ", ".join(f"{c} = COALESCE((SELECT {c} FROM papers WHERE openalex_id = :pub), {c})" for c in _FOLLOWED)
    try:
        conn.execute(
            f"UPDATE papers SET {sets} WHERE openalex_id IN "
            "(SELECT version_id FROM paper_versions WHERE canonical_openalex_id = :pub)",
            {"pub": public_id},
        )
    except sqlite3.OperationalError as exc:
        if "no such table" not in str(exc):     # a database before 0029
            raise
