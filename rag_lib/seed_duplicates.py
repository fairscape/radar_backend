"""Two seeds of one interest that may be one paper (FUNCTIONAL_SPEC M.4).

Compared only among one interest's seeds (one user's):

- Two OpenAlex works with one DOI are one paper: the second is not added
  as a seed (:func:`doi_twin`, called by ``profiles_repo.attach_seed``).
- Seeds with the same title (case, punctuation and spacing aside) are
  marked, both kept, and listed together (:func:`marks`, :func:`grouped`):
  "title" (title duplicate) where one of them is an uploaded PDF or an
  unmatched Prosopia record, whose title is all there is to compare;
  "possible" (possible duplicate) where all are OpenAlex works, such as a
  preprint and its published version. The user removes one; until then
  both count, as any two seeds.

Titles are compared as ``paper.title_key`` does everywhere else: one
shorter than ``MIN_TITLE_KEY_LEN`` ("Editorial", "Correction") is not
evidence of the same paper and is not compared.
"""

from __future__ import annotations

import sqlite3
from typing import Iterable, TypeVar

from .db.repos.versions import PREFIX as UPLOAD_PREFIX
from .paper import title_key

#: A seed whose title is all there is to compare it by.
TITLE_DUPLICATE = "title"
#: Two OpenAlex works with one title.
POSSIBLE_DUPLICATE = "possible"

# An unmatched Prosopia record (``prosopia_seeds.synthetic_id``).
_PROSOPIA_RECORD = "prosopia:"

T = TypeVar("T")


def doi_key(doi: str | None) -> str:
    """A DOI as compared: case and a ``https://doi.org/`` (or ``doi:``) prefix aside."""
    d = (doi or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi:"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    return d.strip()


def _title_only(paper_id: str) -> bool:
    return paper_id.startswith((UPLOAD_PREFIX, _PROSOPIA_RECORD))


def _seed_rows(conn: sqlite3.Connection, profile_id: int) -> list[sqlite3.Row]:
    """Each seed with its own title and DOI, or its public work's (a
    Prosopia copy keeps the work's DOI on the work)."""
    return conn.execute(
        """
        SELECT s.openalex_id AS id, COALESCE(p.title, cp.title) AS title,
               COALESCE(p.doi, cp.doi) AS doi
        FROM profile_seeds s
        LEFT JOIN papers p ON p.openalex_id = s.openalex_id
        LEFT JOIN paper_versions v ON v.version_id = s.openalex_id
        LEFT JOIN papers cp ON cp.openalex_id = v.canonical_openalex_id
        WHERE s.profile_id = ?
        ORDER BY s.added_at, s.openalex_id
        """,
        (profile_id,),
    ).fetchall()


def doi_twin(conn: sqlite3.Connection, profile_id: int, paper_id: str) -> str | None:
    """The seed of ``profile_id`` that is OpenAlex work ``paper_id`` under
    another id with the same DOI, or None. An uploaded file or an
    unmatched Prosopia record is never one: it is compared by title only."""
    if not paper_id or _title_only(paper_id):
        return None
    row = conn.execute(
        """
        SELECT COALESCE(p.doi, cp.doi) FROM papers p
        LEFT JOIN paper_versions v ON v.version_id = p.openalex_id
        LEFT JOIN papers cp ON cp.openalex_id = v.canonical_openalex_id
        WHERE p.openalex_id = ?
        """,
        (paper_id,),
    ).fetchone()
    key = doi_key(row[0]) if row else ""
    if not key:
        return None
    for r in _seed_rows(conn, profile_id):
        if r["id"] != paper_id and not _title_only(r["id"]) and doi_key(r["doi"]) == key:
            return r["id"]
    return None


def marks(conn: sqlite3.Connection, profile_id: int) -> dict[str, dict]:
    """``{seed id: {"kind", "group"}}`` for the seeds sharing a title with
    another seed; ``group`` is the same for each of one title. Seeds
    without a mark are not listed."""
    by_title: dict[str, list[str]] = {}
    for r in _seed_rows(conn, profile_id):
        key = title_key(r["title"])
        if key:
            by_title.setdefault(key, []).append(r["id"])
    out: dict[str, dict] = {}
    for ids in by_title.values():
        if len(ids) < 2:
            continue
        kind = TITLE_DUPLICATE if any(_title_only(i) for i in ids) else POSSIBLE_DUPLICATE
        for i in ids:
            out[i] = {"kind": kind, "group": ids[0]}
    return out


def grouped(items: Iterable[T], marks_: dict[str, dict], id_of) -> list[T]:
    """``items`` in their order, but each marked one moved up to just after
    the first of its group: the copies are listed together."""
    items = list(items)
    groups: dict[str, list[T]] = {}
    for it in items:
        m = marks_.get(id_of(it))
        if m:
            groups.setdefault(m["group"], []).append(it)
    out: list[T] = []
    placed: set[str] = set()
    for it in items:
        m = marks_.get(id_of(it))
        if not m:
            out.append(it)
        elif m["group"] not in placed:
            placed.add(m["group"])
            out.extend(groups[m["group"]])
    return out
