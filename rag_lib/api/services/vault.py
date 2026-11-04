"""Vault service — orchestration for the Phase 6 vault routes.

Functions here are router-thin: they take ``(db, settings, user_id, ...)``
and return Pydantic-shaped dicts the router serializes verbatim.

The upload path is the only write surface:
  1. Refuse anything without a PDF header (``UploadRejected``), then
     hash the bytes.
  2. Dedup against this user's ``paper_uploads`` row for the hash —
     a re-upload of the same file returns the existing VaultDoc,
     marked ``duplicate``.
  3. Write bytes to ``RADAR_VAULT_DIR/{user_id}/{sha256}.pdf``
     (content-addressed; the original filename is preserved on
     ``papers.local_path``).
  4. ``vault.ingest_pdf`` extracts title / body_text / DOI / n_pages; a
     file it cannot open is deleted and refused.
  5. Best-effort OpenAlex enrichment: lookup_by_doi → title search, each
     kept only if it passes the match rules (``title_rule`` /
     ``author_rule``).
     On failure we still create a row with ``source='user_pdf'`` and a
     synthetic ``openalex_id = "local:<hash>"`` so dedup still works.
  6. Embed via the active model (``settings.RADAR_DEFAULT_EMBEDDING_MODEL``,
     defaults to ``placeholder-v1`` so the demo runs without phase1b).
  7. Optionally attach the paper to a profile (``profile_slug``)
     via ``profile_seeds`` so the doc shows up tagged.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any

import structlog


log = structlog.get_logger("rag_lib.api.services.vault")

from ...db.repos import (
    draft_owned as draft_owned_repo,
    embeddings as embeddings_repo,
    paper_rows as paper_rows_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    users as users_repo,
    vault as vault_repo,
    gather_runs as gather_runs_repo,
    versions as versions_repo,
)
from ... import content_topics, openalex_status, seed_duplicates
from ...embed import build_embedding_input
from ...embedders import get_embedder
from ...openalex_client import OpenAlexClient
from ...paper import Paper
from ...rag import indexer as rag_indexer
from ...vault import compute_file_hash, ingest_pdf


# How much text to hand the UMLS extractor. ``extract_umls_concepts``
# wants the abstract; body_text is only a fallback for papers that have
# none. A full PDF body runs 50k-200k chars, which takes minutes through
# en_core_sci_lg + the UMLS linker and can trip spaCy's 1,000,000-char
# ``nlp.max_length`` guard — that raises ValueError E088, which
# ``_try_extract_umls`` swallows, so long documents would silently end up
# with no UMLS data at all. It also hurts precision: references, methods
# boilerplate and figure captions contribute concepts that have nothing
# to do with the paper's subject.
_UMLS_TEXT_LIMIT = 20_000   # == rag_lib.umls.text.UMLS_TEXT_LIMIT


# Lives with the UMLS code (rag_lib/umls/text.py) so the papers repo can tell
# when a paper's extraction input changed; the names here are kept for callers.
from ...umls.text import umls_input_text as _umls_input_text  # noqa: E402


def _vault_dir_for_user(settings: Any, user_id: int) -> Path:
    p = Path(settings.RADAR_VAULT_DIR) / str(user_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _row_to_vault_doc(
    conn: sqlite3.Connection,
    user_id: int,
    row: sqlite3.Row,
    *,
    settings: Any | None = None,
    collection: Any | None = None,
    numbers: dict[str, tuple[int, str]] | None = None,
) -> dict:
    """Shape one papers row as the TS ``VaultDoc`` the frontend expects.

    ``numbers``: ``versions.version_numbers`` for the user, when the caller
    shapes many rows; read here otherwise.

    ``chunks`` is the per-paper chunk count from Chroma. We pass either
    a pre-resolved ``collection`` (when the caller is iterating over
    many papers and wants to amortize the Chroma open) or ``settings``
    so we can resolve a per-user collection lazily. Both default to
    ``None`` so the row → dict conversion stays usable in DB-only paths
    (CLI tooling, repo tests) where Chroma may not be available.
    """
    authors_json = row["authors_json"]
    authors = list(json.loads(authors_json)) if authors_json else []
    chunk_count = _chunk_count_for_paper(
        settings=settings,
        collection=collection,
        user_id=user_id,
        openalex_id=row["openalex_id"],
    )
    return {
        "id": row["openalex_id"],
        # The public work an uploaded version is a copy of: its OpenAlex
        # page, and what says it matched. None for a paper not uploaded.
        "canonicalId": versions_repo.canonical_id(conn, row["openalex_id"]),
        "title": row["title"] or "",
        "authors": authors,
        "venue": row["venue"] or "",
        "tags": vault_repo.tags_for_paper(conn, user_id, row["openalex_id"]),
        "pages": int(row["n_pages"] or 0),
        "chunks": chunk_count,
        # 'pdf' when it is the one printed in this user's uploaded file
        # (FUNCTIONAL_SPEC P.5 Q2): the Vault page says so.
        "abstract": _col(row, "abstract") or None,
        "abstractSource": _col(row, "abstract_source"),
        # Which of the user's files of this paper it is (1 = the first
        # uploaded), when they have more than one; None otherwise.
        "version": (numbers if numbers is not None else versions_repo.version_numbers(conn, user_id))
                   .get(row["openalex_id"], (None, None))[0],
        # This user's own time: first_seen_at is whoever uploaded it first.
        "added": (row["added_at"] if "added_at" in row.keys() else vault_repo.added_at(conn, user_id, row["openalex_id"], row["first_seen_at"])) or "",
    }


class UploadRejected(ValueError):
    """The upload is not something the vault can take; the message says why.

    The router's 4xx. A file that is not a PDF used to reach pdfplumber,
    fail as a 500, and leave the bytes on disk under the user's vault.
    """

    def __init__(self, message: str, *, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


_SEED_FROM = {
    "upload": "an uploaded PDF", "researcher": "a researcher profile", "vault": "the Vault",
    "feed": "the feed", "orcid": "ORCID", "prosopia": "Prosopia",
}


def _refuse_twin(conn: sqlite3.Connection, profile_id: int, *, file_hash: str | None = None,
                 work: str | None = None) -> None:
    """Refuse (409) a paper that is a seed of ``profile_id`` already
    (FUNCTIONAL_SPEC CR.10): the same OpenAlex ``work`` (its public record
    or a file of it, from wherever the seed came), or a file with the same
    bytes."""
    if file_hash is not None:
        row = conn.execute(
            """SELECT s.source FROM profile_seeds s JOIN paper_versions v ON v.version_id = s.openalex_id
               WHERE s.profile_id = ? AND v.file_hash = ? LIMIT 1""",
            (profile_id, file_hash)).fetchone()
    else:
        row = conn.execute(
            """SELECT s.source FROM profile_seeds s LEFT JOIN paper_versions v ON v.version_id = s.openalex_id
               WHERE s.profile_id = ? AND (s.openalex_id = ? OR v.canonical_openalex_id = ?) LIMIT 1""",
            (profile_id, work, work)).fetchone()
    if row is None:
        return
    source = _SEED_FROM.get(row["source"] or "")
    raise UploadRejected(
        "This paper is already a seed of this interest" + (f" (from {source})." if source else "."),
        status_code=409)


def _check_target(conn: sqlite3.Connection, user_id: int, profile_id: int, slug: str) -> None:
    """The interest an upload goes into is still the draft it was, and free.

    A draft discarded meanwhile takes the upload with it (CR.2: the user
    "如果草稿被丢弃，那么新上传的pdf也被丢弃"); one being saved -- its first
    scan running -- or importing takes no seed until that is done (CR.11 L3).
    """
    row = conn.execute("SELECT id, is_draft FROM profiles WHERE id = ? AND user_id = ?",
                       (int(profile_id), int(user_id))).fetchone()
    if row is None:
        raise UploadRejected("The draft was discarded; the PDF was not kept.", status_code=404)
    if not row["is_draft"]:
        # A saved interest changes only through Edit interest, which
        # uploads into its copy (FUNCTIONAL_SPEC E.2-8, E.6 R12).
        raise UploadRejected(f"'{slug}' is a saved interest: add seeds to it in Edit interest.", status_code=409)
    if gather_runs_repo.running_scan(conn, int(profile_id)) is not None:
        raise UploadRejected(f"'{slug}' is being saved; add seeds to it once that finishes.", status_code=409)
    if gather_runs_repo.has_unfinished_import(conn, int(profile_id)):
        raise UploadRejected(
            f"'{slug}' is still importing papers; add seeds to it once that finishes.", status_code=409)


def looks_like_pdf(data: bytes) -> bool:
    """A PDF header within the first 1024 bytes, as readers accept it."""
    return b"%PDF-" in data[:1024]


def upload(
    conn: sqlite3.Connection,
    settings: Any,
    *,
    user_id: int,
    filename: str,
    data: bytes,
    profile_slug: str | None = None,
) -> dict:
    """Upload a PDF into draft ``profile_slug`` (a new interest's, or an
    Edit interest copy) as a seed of it. Returns the VaultDoc shape.

    Every upload is a paper of its own (CR.10); one that is a seed of this
    interest already -- the same file, or the same OpenAlex work -- is
    refused (409). Nothing is written until everything is read, matched
    and embedded, and then in one transaction (CR.3)."""
    import time as _time
    t0 = _time.monotonic()

    if not looks_like_pdf(data):
        raise UploadRejected(
            "This file is not a PDF. Only PDF files can be added.",
            status_code=415,
        )

    if not profile_slug:
        # Papers are uploaded into an interest (the wizard's first step, an
        # interest's Seeds tab); the Vault page has no upload of its own.
        raise UploadRejected("Upload a PDF from an interest: say which one to add it to.", status_code=400)

    file_hash = compute_file_hash(data)

    target_id: int | None = None
    if profile_slug:
        target = profiles_repo.get_by_slug(conn, user_id, profile_slug)
        if target is None:
            # The Seeds tab of an interest deleted in another tab, a wizard
            # draft since discarded, or a slug that is not this user's: the
            # upload used to land in the vault with a 200 and no seed, and
            # nothing said so.
            raise UploadRejected(f"no interest '{profile_slug}' to add the paper to", status_code=404)
        target_id = int(target["id"])
        _check_target(conn, user_id, target_id, profile_slug)
    # OpenAlex paused (its quota used up, or an API key wanted): no upload
    # until then -- a 503 from the app's OpenAlexUnavailable handler.
    openalex_status.check()

    # The same file is a seed here already (CR.10): one seed per paper.
    _refuse_twin(conn, target_id, file_hash=file_hash)

    pdf_dir = _vault_dir_for_user(settings, user_id)
    # Each upload's own file, named once its version id is known (CR.10,
    # review R1): two uploads of one file shared "<hash>.pdf", and deleting
    # one took the file from under the other.
    pdf_path: Path | None = None
    # Read from a scratch name and move it in place only once it parsed: a
    # file pdfplumber cannot open is not kept.
    # One per request: the same file uploaded twice at once (a double-click)
    # shared the name, and the first to finish moved it from under the other.
    part_path = pdf_dir / f".{file_hash}.{os.getpid()}.{threading.get_ident()}.{int(_time.monotonic_ns())}.part"
    part_path.write_bytes(data)
    try:

        t1 = _time.monotonic()
        try:
            record = ingest_pdf(part_path)
        except ImportError:
            part_path.unlink(missing_ok=True)
            raise
        except Exception as exc:  # noqa: BLE001 -- pdfplumber/pdfminer raise many kinds
            part_path.unlink(missing_ok=True)
            log.info("vault.upload.unreadable", filename=filename[:120], reason=str(exc)[:160])
            raise UploadRejected(
                "This file could not be read as a PDF. It may be damaged or password-protected.",
                status_code=422,
            ) from exc
        title_from_file = record.title == part_path.stem
        if title_from_file:
            # No title on the page: ingest_pdf fell back to the file's name,
            # which here is the scratch name (".<hash>"), not the user's.
            record.title = Path(filename).stem.strip() or "Untitled PDF"
        log.info("vault.upload.ingest_pdf", elapsed_ms=round((_time.monotonic() - t1) * 1000, 1))

        t2 = _time.monotonic()
        mailto = users_repo.contact_mailto(conn, user_id, settings.RADAR_DEFAULT_MAILTO)
        try:
            # Only an answer from OpenAlex says whether it has the paper: when it
            # cannot answer (OpenAlexUnavailable, a 503) nothing is kept, not the
            # file either. Stored as "not on OpenAlex", it stayed that for good.
            openalex_id, paper_dict = _enrich_via_openalex(
                record, mailto=mailto, filename=filename, file_hash=file_hash,
            )
        except BaseException:
            part_path.unlink(missing_ok=True)
            raise
        log.info("vault.upload.openalex", elapsed_ms=round((_time.monotonic() - t2) * 1000, 1))

        # The work is public: what OpenAlex says about it goes on its shared
        # row. This file -- its text, vectors, concepts, topics -- is the
        # user's own version (rag_lib.db.repos.versions), one per upload
        # (CR.10): another upload of the paper, by anyone, never shares it.
        canonical = None if openalex_id.startswith("local:") else openalex_id
        public_doi = paper_dict.pop("openalex_doi", None)
        try:
            if canonical is not None:
                # The work is a seed here already, from wherever (CR.10).
                _refuse_twin(conn, target_id, work=canonical)
        except BaseException:
            part_path.unlink(missing_ok=True)
            raise
        public = None
        if canonical is not None:
            public = versions_repo.public_fields(paper_dict)
            # A DOI read from the PDF stays on the user's own version: it may be
            # a reference's, and the shared row holds only what OpenAlex says
            # (CR.11 L2, U10, M.1).
            public["doi"] = public_doi
        openalex_id = versions_repo.new_upload_id(user_id, file_hash)
        pdf_path = pdf_dir / f"{file_hash}-{openalex_id.rsplit(':', 1)[-1]}.pdf"
        paper_dict.update({
            "openalex_id": openalex_id,
            "source": "user_pdf_version",
            "file_hash": file_hash,
            "uploaded_by_user_id": user_id,
            "n_pages": record.n_pages,
            "local_path": str(pdf_path),
            "body_text": record.body_text,
        })
        # OpenAlex has no abstract for it: the one printed in the PDF stands
        # in, on this version only -- it was read from the user's own file
        # (FUNCTIONAL_SPEC P.3-3).
        from_pdf = False
        if not (paper_dict.get("abstract") or "").strip() and record.abstract:
            paper_dict["abstract"] = record.abstract
            from_pdf = True
        if canonical is None:
            # OpenAlex does not have it, so nothing says what it is about: read
            # that from the PDF itself (rag_lib.content_topics).
            found = content_topics.try_topics_json(
                paper_dict.get("title"), paper_dict.get("abstract"), record.body_text, settings=settings)
            if found is not None:
                paper_dict["topics_json"] = found

        # Embedded before anything is written, from what will be stored, so the
        # writes below are one transaction (CR.3): under the default model and
        # under the interest's, if that is another.
        t3 = _time.monotonic()
        text = build_embedding_input(Paper(
            doi=paper_dict.get("doi"), openalex_id=openalex_id, title=paper_dict.get("title") or record.title,
            abstract=paper_dict.get("abstract") or "", year=paper_dict.get("year"), venue=paper_dict.get("venue"),
            body_text=paper_dict.get("body_text"),
        ))
        models = {settings.RADAR_DEFAULT_EMBEDDING_MODEL, target["embedding_model"] or settings.RADAR_DEFAULT_EMBEDDING_MODEL}
        try:
            vectors = {m: get_embedder(m)(text) for m in sorted(models)}
        except BaseException:
            part_path.unlink(missing_ok=True)
            raise
        log.info("vault.upload.embed", models=sorted(models), elapsed_ms=round((_time.monotonic() - t3) * 1000, 1))

        os.replace(part_path, pdf_path)

        # From here the file is in place: any failure below, taking the lock
        # included (review R2), removes it again -- no row names it, and the
        # name is this upload's alone.
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
        try:
            # Looked at again under the write lock: the draft discarded, saved
            # or busy in another tab while the file was read (CR.2, CR.11 L3).
            _check_target(conn, user_id, target_id, profile_slug)
            _refuse_twin(conn, target_id, file_hash=file_hash)
            if canonical is not None:
                _refuse_twin(conn, target_id, work=canonical)
            # Rows this upload adds to the database: an upload into a draft is
            # the draft's until it is saved (draft_owned).
            new_rows = [openalex_id]
            if public is not None:
                if papers_repo.get_by_openalex_id(conn, canonical) is None:
                    new_rows.append(canonical)
                papers_repo.upsert(conn, public, commit=False)
            papers_repo.upsert(conn, paper_dict, commit=False)
            if from_pdf or (paper_dict.get("abstract") or "").strip():
                conn.execute("UPDATE papers SET abstract_source = ? WHERE openalex_id = ?",
                             ("pdf" if from_pdf else None, openalex_id))
            versions_repo.register(conn, user_id=user_id, paper_id=openalex_id,
                                   canonical_id=canonical, file_hash=file_hash,
                                   title_from_file=title_from_file)
            papers_repo.record_upload(
                conn, user_id=user_id, openalex_id=openalex_id, file_hash=file_hash,
                local_path=str(pdf_path), n_pages=record.n_pages, commit=False,
            )
            for model, vector in vectors.items():
                embeddings_repo.upsert(conn, openalex_id, model, vector, commit=False)
            profiles_repo.attach_seed(conn, target_id, openalex_id, source="upload", commit=False)
            for oa in new_rows:
                draft_owned_repo.own(conn, target_id, "paper", oa)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
    except BaseException:
        # Nothing of a failed upload stays on disk: its scratch file, or
        # the file moved into place (CR.3).
        part_path.unlink(missing_ok=True)
        if pdf_path is not None:
            pdf_path.unlink(missing_ok=True)
        raise

    # UMLS extraction is not started here. It used to run on a daemon
    # thread so the upload could return early, and the cost of that was
    # paid by whatever request came next: spaCy and the ollama embedding
    # calls hold the GIL, the event loop is where request bodies are
    # read, and a concurrent upload therefore stalled mid-stream. One
    # took 312 seconds and then failed as "uploaded file is empty",
    # which is a spectacularly unhelpful way to report contention. The
    # Chroma indexing (_defer_heavy_indexing) is off for the same reason:
    # its background thread held the DB write lock for seconds per chunk.
    #
    # Nothing is lost by dropping it: _try_merge_umls_topics computes
    # whatever a profile's seeds are missing when step 3 asks for the
    # topics, which is the only place the result is read.

    row = papers_repo.get_by_openalex_id(conn, openalex_id)
    log.info(
        "vault.upload.done",
        openalex_id=openalex_id,
        total_ms=round((_time.monotonic() - t0) * 1000, 1),
    )
    return _row_to_vault_doc(conn, user_id, row, settings=settings)


def _col(row: sqlite3.Row, key: str):
    """A column some callers' rows do not select."""
    return row[key] if key in row.keys() else None


def list_docs(
    conn: sqlite3.Connection,
    user_id: int,
    *,
    tag: str | None = None,
    settings: Any | None = None,
) -> list[dict]:
    rows = vault_repo.list_for_user(conn, user_id, tag=tag)
    # A work that is this user's upload is listed once, as the upload, also
    # where its public record is one of their researcher's papers.
    copies = set(versions_repo.canonical_ids(conn, [r["openalex_id"] for r in rows]).values())
    rows = [r for r in rows if r["openalex_id"] not in copies]
    # Resolve the user's Chroma collection once and pass it through so
    # we don't re-open the persistent client per row.
    collection = _try_open_user_collection(settings, user_id) if settings else None
    numbers = versions_repo.version_numbers(conn, user_id)
    docs = [
        _row_to_vault_doc(conn, user_id, r, settings=settings, collection=collection, numbers=numbers)
        for r in rows
    ]
    return _with_same_title(conn, user_id, tag, docs)


def _with_same_title(conn: sqlite3.Connection, user_id: int, tag: str | None, docs: list[dict]) -> list[dict]:
    """An interest's docs (``tag``) with its seeds sharing a title marked
    and listed together (FUNCTIONAL_SPEC M.4)."""
    if tag in (None, "", "all"):
        return docs
    row = conn.execute("SELECT id FROM profiles WHERE user_id = ? AND slug = ?", (user_id, tag)).fetchone()
    if row is None:
        return docs
    same = seed_duplicates.marks(conn, int(row[0]))
    # A mark only where its copy is listed too: the public record of a work
    # whose upload is listed is not (list_docs above).
    shown: dict[str, int] = {}
    for d in docs:
        if d["id"] in same:
            shown[same[d["id"]]["group"]] = shown.get(same[d["id"]]["group"], 0) + 1
    same = {k: m for k, m in same.items() if shown.get(m["group"], 0) >= 2}
    for d in docs:
        m = same.get(d["id"])
        if m:
            d["sameTitle"], d["sameTitleGroup"] = m["kind"], m["group"]
    return seed_duplicates.grouped(docs, same, lambda d: d["id"])


def stats(conn: sqlite3.Connection, user_id: int, settings: Any | None = None) -> dict:
    """Frontend-facing stats. ``chunks`` reflects the per-user Chroma
    collection size (0 if Chroma is unavailable)."""
    s = vault_repo.stats_for_user(conn, user_id)
    collection = _try_open_user_collection(settings, user_id) if settings else None
    chunks = rag_indexer.collection_size(collection) if collection is not None else 0
    return {
        "docs": s["docs"],
        "pages": s["pages"],
        "chunks": chunks,
        "lastIngest": s["last_ingest"] or "",
    }


def meta(conn: sqlite3.Connection, settings: Any, user_id: int) -> dict:
    s = vault_repo.stats_for_user(conn, user_id)
    user_chroma = Path(settings.RADAR_CHROMA_DIR) / str(user_id)
    return {
        "rootPath": str(_vault_dir_for_user(settings, user_id)),
        "indexPath": str(user_chroma),
        "chunkSize": "500 tok + 50 overlap",
        "lastIngest": s["last_ingest"] or "",
    }


def tag_counts(conn: sqlite3.Connection, user_id: int) -> dict[str, int]:
    counts = vault_repo.tag_counts_for_user(conn, user_id)
    # The UI shows "all" alongside per-profile counts; "all" must NOT
    # double-count multi-tagged docs, so we compute it from the docs
    # table directly rather than summing the per-tag counts.
    counts["all"] = vault_repo.stats_for_user(conn, user_id)["docs"]
    return counts


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _defer_heavy_indexing(
    *,
    settings: Any,
    user_id: int,
    openalex_id: str,
    body_text: str,
) -> None:
    """Run UMLS extraction and Chroma indexing in a background thread.

    These are best-effort enrichment steps that take 10-60s each.
    Deferring them keeps the upload response fast (~2-5s instead of
    30-300s).
    """
    def _work() -> None:
        from ...db import connect
        try:
            conn = connect(settings.RADAR_DB_PATH)
            try:
                row = papers_repo.get_by_openalex_id(conn, openalex_id)
                if row is None:
                    return
                profile_slugs = vault_repo.tags_for_paper(conn, user_id, openalex_id)

                try:
                    embedder = get_embedder(settings.RADAR_DEFAULT_EMBEDDING_MODEL)
                    collection = rag_indexer.index_user_collection(settings, user_id)
                    rag_indexer.index_paper(
                        collection, row, embedder, profile_slugs=profile_slugs,
                    )
                except Exception:
                    pass

                chat_model = (settings.RADAR_CHAT_EMBEDDING_MODEL or "").strip()
                if chat_model:
                    try:
                        chat_embedder = get_embedder(chat_model)
                        chat_collection = rag_indexer.index_user_chat_collection(
                            settings, user_id,
                        )
                        rag_indexer.index_paper(
                            chat_collection, row, chat_embedder,
                            profile_slugs=profile_slugs,
                            target_tokens=300, overlap=40,
                        )
                    except Exception as exc:
                        log.warning(
                            "vault.deferred.chat_index_skipped",
                            openalex_id=openalex_id,
                            reason=type(exc).__name__,
                        )
            finally:
                conn.close()
        except Exception as exc:
            log.warning(
                "vault.deferred.failed",
                openalex_id=openalex_id,
                reason=type(exc).__name__,
                detail=str(exc)[:200],
            )

    t = threading.Thread(target=_work, daemon=True)
    t.start()


def _try_open_user_collection(settings: Any, user_id: int) -> Any | None:
    """Best-effort handle to the per-user Chroma collection.

    Returns ``None`` when chromadb isn't installed or the persistent
    client raises (e.g., directory permissions). The caller treats
    ``None`` as "no chunk data available" rather than blowing up the
    request.
    """
    if settings is None:
        return None
    try:
        return rag_indexer.index_user_collection(settings, user_id)
    except ImportError:
        return None
    except Exception:
        return None


def _chunk_count_for_paper(
    *,
    settings: Any | None,
    collection: Any | None,
    user_id: int,
    openalex_id: str,
) -> int:
    if collection is None and settings is not None:
        collection = _try_open_user_collection(settings, user_id)
    if collection is None:
        return 0
    return rag_indexer.count_chunks_for_paper(collection, openalex_id)


def _refused_is_none(lookup, value):
    """``lookup(value)``, None when OpenAlex answered but refused the query
    (a 400 for a DOI or title it cannot parse): that is an answer -- it has
    nothing under it. Not answering (``OpenAlexUnavailable``) still raises."""
    import requests

    try:
        return lookup(value)
    except openalex_status.OpenAlexUnavailable:
        raise
    except requests.HTTPError as exc:
        code = getattr(exc.response, "status_code", None)
        if code is None or not 400 <= code < 500:
            raise
        log.info("vault.upload.openalex_refused", status=code, query=str(value)[:120])
        return None


def _enrich_via_openalex(
    record,
    *,
    mailto: str,
    filename: str,
    file_hash: str,
) -> tuple[str, dict]:
    """Try DOI then title. Returns ``(openalex_id, paper_dict)``.

    OpenAlex not having the paper is not an error: the paper_dict then has
    a synthetic ``local:<hash>`` id and ``source='user_pdf'``. OpenAlex not
    answering is: ``OpenAlexUnavailable`` propagates and nothing is stored.
    Authors land on ``paper_dict['authors']`` when OpenAlex resolves the work.
    """
    paper_dict: dict[str, Any] = {
        "title": record.title or filename,
        "doi": record.doi,
        "source": "user_pdf",
    }
    client = OpenAlexClient(mailto=mailto)
    work = None
    if record.doi:
        work = _refused_is_none(client.lookup_by_doi, record.doi)
        # A DOI on the page can be a cited work's; keep it only if the
        # PDF looks like that work.
        if work is not None and not _verified_doi_match(record, work):
            log.info("vault.upload.doi_rejected", doi=record.doi,
                     work_title=(work.get("title") or "")[:120])
            work = None
            # It is some other paper's DOI, so it must not stay on this
            # one: seeds are matched by DOI, and the local paper then
            # kept the cited paper (often the most relevant) out of the
            # feed as if it were a seed itself.
            paper_dict["doi"] = None
    if work is None and record.title:
        work = _verified_title_match(record, _refused_is_none(client.search_by_title, record.title) or [])
    if work is None and getattr(record, "doi_from_body", False):
        paper_dict["doi"] = None     # not vouched for by anything: likely a reference's
    if work is not None:
        paper = client.paper_from_work(work, source="user_pdf")
        paper_dict["openalex_id"] = paper.openalex_id
        # What OpenAlex says the DOI is, for the shared row; the version
        # keeps the PDF's when OpenAlex has none.
        paper_dict["openalex_doi"] = paper.doi
        paper_dict["doi"] = paper.doi or paper_dict.get("doi")
        paper_dict["title"] = paper.title or paper_dict["title"]
        paper_dict["abstract"] = paper.abstract
        paper_dict["year"] = paper.year
        paper_dict["venue"] = paper.venue
        paper_dict["primary_topic"] = (
            paper.primary_topic.to_dict() if paper.primary_topic else None
        )
        paper_dict["topics"] = [t.to_dict() for t in paper.topics]
        paper_dict["authors"] = _authors_from_work(work)

    if "openalex_id" not in paper_dict:
        paper_dict["openalex_id"] = f"local:{file_hash}"
    return paper_dict["openalex_id"], paper_dict


# Whether an OpenAlex work is the uploaded PDF is decided by rules, each a
# yes/no check of one fact, not by a similarity score: OpenAlex ranks a
# near miss first for a title it does not have, and a short title matches
# something else exactly -- a lab note titled "Deep learning" became
# Goodfellow et al.'s book, and its text was written into that shared paper.
#
#   title    the PDF's title is the work's (normalized; or minus a subtitle),
#            or the work's whole title is printed on the PDF's first page
#   authors  the work's first author is named in the PDF's first pages, and
#            a second of its authors too -- unless it has one author, or the
#            first is a group ("DeepSeek-AI") whose members are listed last
#
# A title-search hit must pass both. A DOI hit needs either: the DOI already
# names the work, and the check only catches one the extractor took from
# the references. A DOI that differs is not held against a title hit: a
# preprint carries its arXiv DOI while OpenAlex has the published one, and
# the title search only runs once the PDF's DOI has failed to identify it.
#
# Names are matched on text with the spaces taken out: pdfplumber often
# drops them ("AlexMallen1", "KathleenC.Frasera" -- the digits and the
# trailing letter are affiliation marks), so a whole-word match missed
# most real bylines.

# Authors and the title are looked for where they are printed: the start.
_HEAD_CHARS = 6000          # <= rag_lib.vault.FRONT_CHARS: read from front_text
_FIRST_PAGE_CHARS = 3000
# Letters allowed between a given name and the family name: a middle
# initial or a particle ("Kathleen C. Fraser", "Guillaume Le Moing").
_MIDDLE = 4


def _norm(text: str | None) -> str:
    """Lowercase ASCII words: accents stripped, punctuation dropped."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    ascii_ = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", ascii_.lower()).strip()


def _squash(text: str | None) -> str:
    """Letters only, no spaces: how a byline reads once the spaces are lost."""
    return re.sub(r"[^a-z]+", "", _norm(text))


def _work_title(work: dict) -> str:
    return work.get("title") or work.get("display_name") or ""


def _main_title(title: str | None) -> str:
    """The title without a subtitle (what follows the first colon)."""
    return _norm((title or "").split(":", 1)[0])


def title_rule(record: Any, work: dict) -> bool:
    pdf, oa = _norm(record.title), _norm(_work_title(work))
    if not pdf or not oa:
        return False
    if pdf == oa or _main_title(record.title) == oa or pdf == _main_title(_work_title(work)):
        return True
    return _squash(_work_title(work)) in _squash(_front_matter(_front_text(record)))


_REFERENCES_HEADING = re.compile(r"\n\s*(?:\d+\.?\s*)?(?:references|bibliography|literature cited)\s*\n", re.IGNORECASE)


def _front_text(record: Any) -> str:
    """The start of the PDF as pdfplumber reads it (``PdfIngestRecord.front_text``);
    a record without one (built by hand) is looked at through its body."""
    return getattr(record, "front_text", "") or record.body_text or ""


def _front_matter(body: str | None) -> str:
    """The first page's worth of text, stopping at a reference list: on a
    short PDF the references start inside it, and a cited title found
    there is not this paper's."""
    head = (body or "")[:_FIRST_PAGE_CHARS]
    m = _REFERENCES_HEADING.search(head)
    return head[: m.start()] if m else head


def _name_parts(name: str) -> tuple[str, str]:
    """``(given, family)``; OpenAlex gives "First Last" or "Last, First"."""
    if "," in name:
        family, given = name.split(",", 1)
    else:
        tokens = name.strip().split()
        family, given = (tokens[-1], " ".join(tokens[:-1])) if tokens else ("", "")
    return _squash(given), _squash(family)


def _is_group(name: str) -> bool:
    return len(name.split()) == 1 and "," not in name


def _named_in(name: str, head: str, spaced_head: str) -> bool:
    """Whether this author is printed in the head.

    The full name on the squashed text ("kathleen" up to 4 letters "fraser"),
    or, for a byline of initials ("K. C. Fraser"), the initial and the family
    name as words on the spaced text, an affiliation letter allowed after it.
    """
    if _is_group(name):
        token = _squash(name)
        return len(token) >= 4 and token in head
    given, family = _name_parts(name)
    if len(family) < 2 or not given:
        return False
    if re.search(re.escape(given) + f"[a-z]{{0,{_MIDDLE}}}" + re.escape(family), head):
        return True
    initials = r"(?<![a-z])" + re.escape(given[0]) + r"(?: [a-z]){0,2} " + re.escape(family) + r"[a-z]?(?![a-z])"
    return len(family) >= 3 and re.search(initials, spaced_head) is not None


def author_rule(record: Any, work: dict, *, first_only: bool = False) -> bool:
    text = _front_text(record)[:_HEAD_CHARS]
    head = _squash(text)
    spaced = re.sub(r"[0-9]+", " ", _norm(text))
    if not head:
        return False
    names = [n for n in _authors_from_work(work) if _squash(n)]
    if not names or not _named_in(names[0], head, spaced):
        return False
    if first_only or len(names) == 1 or _is_group(names[0]):
        return True
    return any(_named_in(n, head, spaced) for n in names[1:])


def _verified_title_match(record: Any, works: list[dict] | None) -> dict | None:
    """The search hit that passes every rule, or None (the paper stays local)."""
    for work in works or []:
        checks = {"title": title_rule(record, work), "authors": author_rule(record, work)}
        if all(checks.values()):
            return work
        log.info("vault.upload.title_match_rejected", title=(record.title or "")[:120],
                 candidate=_work_title(work)[:120], failed=[k for k, ok in checks.items() if not ok])
    return None


def _verified_doi_match(record: Any, work: dict) -> bool:
    """A DOI the PDF prints as its own needs one more fact; one found only
    further in the body may be a reference's, and a self-citation passes the
    first-author check -- so that one must pass both rules, like a title hit."""
    if getattr(record, "doi_from_body", False):
        return title_rule(record, work) and author_rule(record, work)
    return title_rule(record, work) or author_rule(record, work, first_only=True)


def _authors_from_work(work: dict) -> list[str]:
    out: list[str] = []
    for entry in (work.get("authorships") or []):
        author = (entry or {}).get("author") or {}
        name = author.get("display_name")
        if name:
            out.append(name)
    return out


# One extraction at a time across the process. The spaCy pipeline and its
# UMLS linker are one shared object, and extraction now runs from request
# threads (upload, wizard step 3), the import thread and the scheduler's
# seed backfill at once; nothing documents that pipeline as safe for
# concurrent calls, and the cost of serializing is only latency.
_UMLS_LOCK = threading.Lock()

# When each paper's extraction last failed in this process. The sweep takes
# the first ``limit`` NULL rows; a paper that keeps failing stays NULL, so
# without this the same 40 were retried every 15 minutes and nothing behind
# them was ever reached. Skipped for a while, not until a restart: the
# backend runs for weeks, and a failure is usually an outage (the topic
# embedder down) that ends long before then.
_UMLS_FAILED: dict[str, float] = {}
_UMLS_RETRY_AFTER_S = 6 * 3600


def _recently_failed(openalex_id: str) -> bool:
    at = _UMLS_FAILED.get(openalex_id)
    return at is not None and time.monotonic() - at < _UMLS_RETRY_AFTER_S


def _try_extract_umls(
    conn: sqlite3.Connection,
    settings: Any,
    openalex_id: str,
    text: str,
    *,
    force: bool = False,
) -> bool:
    """Best-effort UMLS extraction for one paper, serialized. True if stored.

    Callers choose what to extract (``umls_concepts_json IS NULL``) before
    they hold the lock, so the import thread, the periodic sweep and step 3
    could each pick the same paper and run the linker on it again. Checked
    once more under the lock; ``force`` (the backfill script's re-extract)
    skips that.
    """
    with _UMLS_LOCK:
        if not force:
            row = conn.execute(
                "SELECT umls_concepts_json FROM papers WHERE openalex_id = ?", (openalex_id,),
            ).fetchone()
            if row is not None and row[0] is not None:
                return True
        return _try_extract_umls_unlocked(conn, settings, openalex_id, text)


def backfill_seed_umls(
    conn: sqlite3.Connection,
    settings: Any,
    *,
    openalex_ids: list[str] | None = None,
    limit: int = 40,
) -> int:
    """Extract UMLS for seed papers that have never been extracted.

    Seeds used to be extracted only when wizard step 3 opened, within a
    30-second budget, the rest "picked up the next time step 3 is opened"
    -- which never happens once the interest is committed. On 2026-09-30
    an 82-seed import got 6 and kept 76 empty for good. This runs from
    the import thread (``openalex_ids`` = that import's seeds) and from a
    periodic scheduler job (all seeds), at most ``limit`` papers a call,
    through the same extraction as every other path. Returns how many
    were attempted.
    """
    if not settings.RADAR_UMLS_ENABLED:
        return 0
    if openalex_ids is not None:
        if not openalex_ids:
            return 0
        marks = ",".join("?" for _ in openalex_ids)
        rows = conn.execute(
            f"SELECT openalex_id, title, abstract, body_text FROM papers "
            f"WHERE umls_concepts_json IS NULL AND openalex_id IN ({marks}) LIMIT ?",
            (*openalex_ids, limit + len(_UMLS_FAILED)),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT p.openalex_id, p.title, p.abstract, p.body_text FROM papers p
            WHERE p.umls_concepts_json IS NULL
              AND EXISTS (SELECT 1 FROM profile_seeds s WHERE s.openalex_id = p.openalex_id)
            LIMIT ?
            """,
            (limit + len(_UMLS_FAILED),),
        ).fetchall()
    rows = [r for r in rows if not _recently_failed(r["openalex_id"])][:limit]
    n_failed = 0
    for r in rows:
        if _try_extract_umls(conn, settings, r["openalex_id"],
                             _umls_input_text(r["title"], r["abstract"], r["body_text"])):
            _UMLS_FAILED.pop(r["openalex_id"], None)
        else:
            _UMLS_FAILED[r["openalex_id"]] = time.monotonic()
            n_failed += 1
    if rows:
        log.info("vault.umls_backfill", n=len(rows), n_failed=n_failed,
                 scoped=openalex_ids is not None)
    return len(rows)


def seed_umls_status(conn: sqlite3.Connection, settings: Any, openalex_ids: list[str]) -> dict:
    """How far UMLS extraction is for a set of seeds.

    ``done``: extracted (concepts found or not). ``failed``: extraction
    failed recently and waits out ``_UMLS_RETRY_AFTER_S``. ``pending``:
    not extracted yet. With UMLS switched off everything counts as done.
    """
    total = len(openalex_ids)
    if not settings.RADAR_UMLS_ENABLED or not openalex_ids:
        return {"enabled": bool(settings.RADAR_UMLS_ENABLED), "total": total, "done": total,
                "pending": 0, "failed": 0}
    marks = ",".join("?" for _ in openalex_ids)
    null_ids = [r[0] for r in conn.execute(
        f"SELECT openalex_id FROM papers WHERE umls_concepts_json IS NULL AND openalex_id IN ({marks})",
        tuple(openalex_ids),
    ).fetchall()]
    failed = sum(1 for oa in null_ids if _recently_failed(oa))
    return {"enabled": True, "total": total, "done": total - len(null_ids),
            "pending": len(null_ids) - failed, "failed": failed}


def _try_extract_umls_unlocked(
    conn: sqlite3.Connection,
    settings: Any,
    openalex_id: str,
    text: str,
) -> bool:
    """Best-effort UMLS extraction + topic mapping for a paper.

    Extracts UMLS concepts from the paper text, maps them to OpenAlex
    topics, and stores both as JSON on the paper row. Failures are logged
    and silently swallowed — UMLS enrichment is optional.
    """
    try:
        if not settings.RADAR_UMLS_ENABLED:
            return False
        if not text or not text.strip():
            # Nothing to read is a result, not a failure: without the
            # marker the backfill picked the same empty papers every run.
            conn.execute(
                "UPDATE papers SET umls_concepts_json = '[]', "
                "umls_mapped_topics_json = NULL WHERE openalex_id = ?",
                (openalex_id,),
            )
            conn.commit()
            return True

        from ...umls.extractor import extract_umls_concepts
        from ...umls.topic_mapper import map_concepts_to_topics

        # cache_dir has to be passed here too, not just in the app.py
        # warmup: _get_nlp() only honours it on the *first* load, so if
        # the warmup was skipped or failed this call becomes the first
        # one and would otherwise download the ~1GB UMLS KB into
        # ~/.scispacy instead of RADAR_UMLS_CACHE_DIR.
        concepts = extract_umls_concepts(
            text,
            min_confidence=settings.RADAR_UMLS_MIN_CONFIDENCE,
            spacy_model=settings.RADAR_UMLS_SPACY_MODEL,
            cache_dir=str(settings.RADAR_UMLS_CACHE_DIR),
            max_concepts=settings.RADAR_UMLS_MAX_CONCEPTS,
        )

        if not concepts:
            # "Looked, found nothing" is stored as [] rather than left NULL.
            # NULL is what every caller reads as "not extracted yet", so a
            # paper with no biomedical concept was re-run each time step 3
            # opened -- and would be re-run by the backfill on every pass.
            conn.execute(
                "UPDATE papers SET umls_concepts_json = '[]', "
                "umls_mapped_topics_json = NULL WHERE openalex_id = ?",
                (openalex_id,),
            )
            conn.commit()
            return True

        concepts_json = json.dumps([c.to_dict() for c in concepts])

        # Map concepts to OpenAlex topics via the topic embedding index
        cache_dir = str(settings.RADAR_UMLS_CACHE_DIR)
        mapped = map_concepts_to_topics(
            concepts,
            embedding_model=settings.RADAR_UMLS_EMBEDDING_MODEL,
            cache_dir=cache_dir,
        )
        # "[]" too: NULL beside concepts is a mapping by the earlier rules,
        # which found no topic near enough (wizard.forget_stale_umls_mappings).
        mapped_json = json.dumps([m.to_dict() for m in mapped])

        conn.execute(
            "UPDATE papers SET umls_concepts_json = ?, umls_mapped_topics_json = ? "
            "WHERE openalex_id = ?",
            (concepts_json, mapped_json, openalex_id),
        )
        conn.commit()

        log.info(
            "vault.umls_extracted",
            openalex_id=openalex_id,
            n_concepts=len(concepts),
            n_mapped=len(mapped),
        )
        return True
    except Exception as exc:
        log.warning(
            "vault.umls_extraction_skipped",
            openalex_id=openalex_id,
            reason=type(exc).__name__,
            detail=str(exc)[:200],
        )
        return False


class DeleteRefused(ValueError):
    """A Vault delete that cannot go ahead now; ``status_code`` for the router."""

    def __init__(self, message: str, status_code: int = 409):
        super().__init__(message)
        self.status_code = status_code


def delete_upload(conn: sqlite3.Connection, settings: Any, *, user_id: int, doc_id: str) -> dict:
    """Delete the user's uploaded PDF from the Vault (FUNCTIONAL_SPEC F1).

    The file, its text and its vectors go, and the Vault no longer lists
    it. Wherever it is a seed it stops being one: on a saved interest it
    stays on the seed list, last and marked deleted (``deleted_seeds``);
    its seeds no longer match its fit, so its next scan refits without it
    (``jobs._refit_if_fit_stale``);
    once that scan has scored without it the mark goes
    (``gather_runs.finish``). Only the user's own upload; another user's
    copy of the same file is theirs and stays (M.1).
    """
    # Checked under the write lock (CR.11 L5): a seed removed in another tab,
    # or a scan started, between the check and the delete left an interest
    # with no seed at all.
    if not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")
    try:
        upload = conn.execute(
            "SELECT local_path FROM paper_uploads WHERE user_id = ? AND openalex_id = ?",
            (int(user_id), doc_id),
        ).fetchone()
        if upload is None:
            raise DeleteRefused("This PDF is not one of your uploads.", status_code=404)
        seeded = conn.execute(
            "SELECT p.id, p.name, p.is_draft, p.edit_of FROM profiles p"
            " JOIN profile_seeds s ON s.profile_id = p.id WHERE s.openalex_id = ? AND p.user_id = ?",
            (doc_id, int(user_id)),
        ).fetchall()
        for p in seeded:
            if (gather_runs_repo.running_scan(conn, int(p["id"])) is not None
                    or gather_runs_repo.has_unfinished_import(conn, int(p["id"]))):
                raise DeleteRefused(f"A scan or an import of '{p['name']}' is running. Delete the PDF once it has finished.")
            n = conn.execute("SELECT COUNT(*) FROM profile_seeds WHERE profile_id = ?", (int(p["id"]),)).fetchone()[0]
            if n <= 1 and not p["is_draft"]:
                raise DeleteRefused(f"It is the only seed of '{p['name']}'. Add another seed there first.")
        title = conn.execute("SELECT title FROM papers WHERE openalex_id = ?", (doc_id,)).fetchone()
        # Shown on the deleted mark beside the delete time (CR.4).
        uploaded = conn.execute(
            "SELECT uploaded_at FROM paper_uploads WHERE user_id = ? AND openalex_id = ?", (int(user_id), doc_id),
        ).fetchone()
        paths = {r[0] for r in conn.execute(
            "SELECT local_path FROM papers WHERE openalex_id = ? AND local_path IS NOT NULL "
            "UNION SELECT local_path FROM paper_uploads WHERE openalex_id = ? AND local_path IS NOT NULL",
            (doc_id, doc_id))}
        for p in seeded:
            if not p["is_draft"]:
                conn.execute(
                    "INSERT OR REPLACE INTO deleted_seeds (profile_id, openalex_id, title, uploaded_at) VALUES (?, ?, ?, ?)",
                    (int(p["id"]), doc_id, title["title"] if title else None, uploaded[0] if uploaded else None),
                )
        conn.execute(
            "DELETE FROM profile_seeds WHERE openalex_id = ? AND profile_id IN (SELECT id FROM profiles WHERE user_id = ?)",
            (doc_id, int(user_id)),
        )
        for p in seeded:
            conn.execute("UPDATE profiles SET n_seed = (SELECT COUNT(*) FROM profile_seeds WHERE profile_id = ?) WHERE id = ?",
                         (int(p["id"]), int(p["id"])))
        # This user's hold on it goes; then the row, if nothing else names it.
        conn.execute("DELETE FROM draft_owned WHERE kind = 'paper' AND item = ?", (doc_id,))
        conn.execute("DELETE FROM paper_uploads WHERE user_id = ? AND openalex_id = ?", (int(user_id), doc_id))
        conn.execute("DELETE FROM vault_kept WHERE user_id = ? AND openalex_id = ?", (int(user_id), doc_id))
        if not paper_rows_repo.delete_if_unused(conn, doc_id):
            # Something else still names the row: its content goes, the row stays.
            conn.execute("DELETE FROM paper_embeddings WHERE openalex_id = ?", (doc_id,))
            conn.execute("UPDATE papers SET body_text = NULL, local_path = NULL WHERE openalex_id = ?", (doc_id,))
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    for path in paths:
        still = conn.execute(
            "SELECT 1 FROM papers WHERE local_path = ? UNION SELECT 1 FROM paper_uploads WHERE local_path = ?",
            (path, path)).fetchone()
        if still is None:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError as exc:
                log.warning("vault.delete.file_left", path=path, reason=str(exc)[:160])
    # Its chunks, in retrieval's collection and in chat's (CR.11 L6); chat's
    # exists only once cli/reindex_chat has filled it under a chat model.
    openers = [rag_indexer.index_user_collection]
    if (getattr(settings, "RADAR_CHAT_EMBEDDING_MODEL", "") or "").strip():
        openers.append(rag_indexer.index_user_chat_collection)
    for open_collection in openers:
        try:
            collection = open_collection(settings, user_id) if settings is not None else None
        except Exception:  # noqa: BLE001 -- chromadb missing or unusable: no chunks to drop
            collection = None
        if collection is None:
            continue
        try:
            collection.delete(where={"openalex_id": str(doc_id)})
        except Exception as exc:  # noqa: BLE001 -- the chunks are only retrieval's
            log.warning("vault.delete.chunks_left", openalex_id=doc_id, reason=str(exc)[:160])
    log.info("vault.deleted", user_id=user_id, openalex_id=doc_id, n_interests=len(seeded))
    return {"ok": True, "interests": [p["name"] for p in seeded if not p["is_draft"]]}
