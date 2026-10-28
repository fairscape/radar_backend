"""papers repo.

Content-addressed by ``openalex_id``. A single paper exists once even if
it is a seed in profile A and a candidate in profile B. The dict shape
mirrors ``rag_lib.paper.Paper.to_dict()``; Phase 2's ``db_store`` does
the conversion so this module stays import-free of the dataclasses.
"""

from __future__ import annotations

import datetime as _dt
import json
import sqlite3
from typing import Any


def plausible_year(year: Any) -> int | None:
    """``year``, or None when it cannot be a publication year yet.

    OpenAlex carries works dated 2029, 2040, 2050 -- placeholders and typos.
    Stored, an interest whose years run "to now" let them into its feed as
    the newest papers (round 19). Next year stays: journals date issues
    ahead. An unknown year passes every year filter, so the paper stays.
    """
    try:
        y = int(year)
    except (TypeError, ValueError):
        return None
    return y if 1000 <= y <= _dt.date.today().year + 1 else None


def get_by_openalex_id(
    conn: sqlite3.Connection, openalex_id: str
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM papers WHERE openalex_id = ?", (openalex_id,)
    ).fetchone()


def get_by_doi(conn: sqlite3.Connection, doi: str) -> sqlite3.Row | None:
    """The public paper with this DOI. A user's uploaded version carries
    its work's DOI too and is never what a DOI resolves to."""
    return conn.execute(
        "SELECT * FROM papers p WHERE p.doi = ?"
        " AND NOT EXISTS (SELECT 1 FROM paper_versions v WHERE v.version_id = p.openalex_id)"
        " ORDER BY p.openalex_id LIMIT 1",
        (doi,),
    ).fetchone()


def upsert(conn: sqlite3.Connection, paper: dict[str, Any]) -> str:
    """Insert (or update on conflict) a paper row.

    Required keys: ``openalex_id``, ``title``, ``source``.
    Optional: ``doi``, ``abstract``, ``year``, ``venue``,
    ``publication_date``, ``primary_topic`` + ``topics`` (encoded as
    ``topics_json``), ``local_path``, ``body_text``, the Phase 6
    vault fields ``file_hash``, ``uploaded_by_user_id``, ``n_pages``,
    and ``authors`` (list, serialized into ``authors_json``), and the
    OpenAlex open-access fields ``pdf_url`` and ``oa_status``.

    Returns the ``openalex_id`` (caller-friendly when chaining).
    """
    if not paper.get("openalex_id"):
        raise ValueError("papers.upsert requires openalex_id")

    topics_json = _encode_topics(paper)
    authors_json = _encode_authors(paper)
    before = conn.execute(
        "SELECT title, abstract, body_text, umls_concepts_json FROM papers WHERE openalex_id = ?",
        (paper["openalex_id"],),
    ).fetchone()
    fields = (
        paper["openalex_id"],
        paper.get("doi"),
        paper.get("title") or "",
        paper.get("abstract"),
        plausible_year(paper.get("year")),
        paper.get("venue"),
        paper.get("publication_date"),
        topics_json,
        paper.get("source") or "unknown",
        paper.get("local_path"),
        paper.get("body_text"),
        paper.get("file_hash"),
        paper.get("uploaded_by_user_id"),
        paper.get("n_pages"),
        authors_json,
        paper.get("pdf_url"),
        paper.get("oa_status"),
    )

    conn.execute(
        """
        INSERT INTO papers (
          openalex_id, doi, title, abstract, year, venue, publication_date,
          topics_json, source, local_path, body_text,
          file_hash, uploaded_by_user_id, n_pages, authors_json,
          pdf_url, oa_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(openalex_id) DO UPDATE SET
          doi                 = COALESCE(excluded.doi, doi),
          -- OpenAlex returns works with no title; a scan of another interest
          -- then blanked a seed's title for every user of the paper.
          title               = COALESCE(NULLIF(excluded.title, ''), title),
          -- NULLIF: a gather passes '' for "OpenAlex has none" (and for
          -- body text it never has), which COALESCE took as a value -- a
          -- re-fetched seed lost its PDF body text or its Prosopia abstract.
          abstract            = COALESCE(NULLIF(excluded.abstract, ''), abstract),
          year                = COALESCE(excluded.year, year),
          venue               = COALESCE(excluded.venue, venue),
          publication_date    = COALESCE(excluded.publication_date, publication_date),
          topics_json         = COALESCE(excluded.topics_json, topics_json),
          -- The first upload's file and uploader stay; each user's own
          -- upload is a paper_uploads row (migration 0020).
          local_path          = COALESCE(local_path, excluded.local_path),
          body_text           = COALESCE(NULLIF(excluded.body_text, ''), body_text),
          file_hash           = COALESCE(file_hash, excluded.file_hash),
          uploaded_by_user_id = COALESCE(uploaded_by_user_id, excluded.uploaded_by_user_id),
          n_pages             = COALESCE(excluded.n_pages, n_pages),
          authors_json        = COALESCE(excluded.authors_json, authors_json),
          pdf_url             = COALESCE(excluded.pdf_url, pdf_url),
          oa_status           = COALESCE(excluded.oa_status, oa_status)
        """,
        fields,
    )
    if before is not None and before[3] is not None:     # extracted before
        _forget_umls_if_text_changed(
            conn, paper, dict(zip(("title", "abstract", "body_text"), before[:3])))
    from .versions import follow_public

    follow_public(conn, paper["openalex_id"])
    conn.commit()
    return paper["openalex_id"]


def _forget_umls_if_text_changed(conn: sqlite3.Connection, paper: dict[str, Any], before) -> None:
    """Clear a paper's UMLS results when the text they were read from changed.

    Every extraction path only picks papers whose concepts are NULL, and an
    update never touched them, so a seed imported without an abstract
    (concepts from the title alone) kept them after its PDF's body text or
    a later abstract arrived. NULL sends it back to extraction. Compared
    on the text extraction reads, mirroring the COALESCEs above, so a
    re-import of the same record changes nothing.
    """
    from ...umls.text import umls_input_text

    def merged(key):     # as the upsert's COALESCE(NULLIF(new, ''), old)
        return paper.get(key) if paper.get(key) else before[key]

    old = umls_input_text(before["title"], before["abstract"], before["body_text"])
    new = umls_input_text(merged("title"), merged("abstract"), merged("body_text"))
    if new != old:
        conn.execute(
            "UPDATE papers SET umls_concepts_json = NULL, umls_mapped_topics_json = NULL "
            "WHERE openalex_id = ?",
            (paper["openalex_id"],),
        )


def get_by_user_hash(
    conn: sqlite3.Connection, user_id: int, file_hash: str
) -> sqlite3.Row | None:
    """Look up an existing user-uploaded paper by content hash.

    The vault upload path uses this for idempotent re-upload of the
    same bytes by the same user.
    """
    return conn.execute(
        """
        SELECT p.* FROM papers p
        JOIN paper_uploads u ON u.openalex_id = p.openalex_id
        WHERE u.user_id = ? AND u.file_hash = ?
        LIMIT 1
        """,
        (user_id, file_hash),
    ).fetchone()


def record_upload(
    conn: sqlite3.Connection, *, user_id: int, openalex_id: str,
    file_hash: str, local_path: str | None, n_pages: int | None,
) -> None:
    """This user uploaded this paper (``paper_uploads``; migration 0020).

    Per user, so a second user uploading the same paper adds their own row
    instead of taking the paper over. A re-upload of a different file of
    the same paper by the same user replaces their file.
    """
    conn.execute(
        """
        INSERT INTO paper_uploads (user_id, openalex_id, file_hash, local_path, n_pages)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (user_id, openalex_id) DO UPDATE SET
          file_hash  = excluded.file_hash,
          local_path = excluded.local_path,
          n_pages    = excluded.n_pages
        """,
        (user_id, openalex_id, file_hash, local_path, n_pages),
    )
    conn.commit()


def decode_authors(row: sqlite3.Row | None) -> list[str]:
    if row is None or row["authors_json"] is None:
        return []
    return list(json.loads(row["authors_json"]))


def _encode_topics(paper: dict[str, Any]) -> str | None:
    if "topics_json" in paper and paper["topics_json"] is not None:
        return paper["topics_json"]
    primary = paper.get("primary_topic")
    topics = paper.get("topics")
    if primary is None and not topics:
        return None
    return json.dumps({"primary_topic": primary, "topics": topics or []})


def _encode_authors(paper: dict[str, Any]) -> str | None:
    if "authors_json" in paper and paper["authors_json"] is not None:
        return paper["authors_json"]
    authors = paper.get("authors")
    if not authors:
        return None
    return json.dumps(list(authors))


def decode_topics(row: sqlite3.Row) -> dict:
    raw = row["topics_json"] if row is not None else None
    if not raw:
        return {"primary_topic": None, "topics": []}
    return json.loads(raw)
