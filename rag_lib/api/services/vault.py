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
    embeddings as embeddings_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    users as users_repo,
    vault as vault_repo,
    gather_runs as gather_runs_repo,
)
from ... import content_topics
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
) -> dict:
    """Shape one papers row as the TS ``VaultDoc`` the frontend expects.

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
        "title": row["title"] or "",
        "authors": authors,
        "venue": row["venue"] or "",
        "tags": vault_repo.tags_for_paper(conn, user_id, row["openalex_id"]),
        "pages": int(row["n_pages"] or 0),
        "chunks": chunk_count,
        "added": row["first_seen_at"] or "",
    }


class UploadRejected(ValueError):
    """The upload is not something the vault can take; the message says why.

    The router's 4xx. A file that is not a PDF used to reach pdfplumber,
    fail as a 500, and leave the bytes on disk under the user's vault.
    """

    def __init__(self, message: str, *, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


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
    """Idempotent upload. Returns the VaultDoc shape for the resulting paper."""
    import time as _time
    t0 = _time.monotonic()

    if not looks_like_pdf(data):
        raise UploadRejected(
            "This file is not a PDF. Only PDF files can be added.",
            status_code=415,
        )

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
        if gather_runs_repo.has_unfinished_import(conn, int(target["id"])):
            # The Vault's "upload to <draft>" while that draft's import runs:
            # the wizard holds its own buttons, this path had no guard.
            raise UploadRejected(
                f"'{profile_slug}' is still importing papers; add seeds to it once that finishes.",
                status_code=409,
            )

    existing = papers_repo.get_by_user_hash(conn, user_id, file_hash)
    if existing is not None:
        # The rows are written before the paper is embedded, so an upload
        # that failed there (embedder down: a 500) left a paper with no
        # vector, and every retry was a "duplicate" that never embedded it
        # -- a seed silently missing from the fit. A retry finishes the job.
        _embed_if_missing(conn, existing, settings)
        if profile_slug:
            _attach_to_profile(conn, user_id, profile_slug, existing["openalex_id"], target_id)
        log.info(
            "vault.upload.dedup_hit",
            openalex_id=existing["openalex_id"],
            elapsed_ms=round((_time.monotonic() - t0) * 1000, 1),
        )
        doc = _row_to_vault_doc(conn, user_id, existing, settings=settings)
        doc["duplicate"] = True
        return doc

    pdf_dir = _vault_dir_for_user(settings, user_id)
    pdf_path = pdf_dir / f"{file_hash}.pdf"
    # Read from a scratch name and move it in place only once it parsed: a
    # file pdfplumber cannot open is not kept.
    # One per request: the same file uploaded twice at once (a double-click)
    # shared the name, and the first to finish moved it from under the other.
    part_path = pdf_dir / f".{file_hash}.{os.getpid()}.{threading.get_ident()}.{int(_time.monotonic_ns())}.part"
    part_path.write_bytes(data)

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
    try:
        os.replace(part_path, pdf_path)
    except FileNotFoundError:
        if not pdf_path.exists():
            raise
    if record.title == part_path.stem:
        # No title on the page: ingest_pdf fell back to the file's name,
        # which here is the scratch name (".<hash>"), not the user's.
        record.title = Path(filename).stem.strip() or "Untitled PDF"
    log.info("vault.upload.ingest_pdf", elapsed_ms=round((_time.monotonic() - t1) * 1000, 1))

    t2 = _time.monotonic()
    mailto = users_repo.contact_mailto(conn, user_id, settings.RADAR_DEFAULT_MAILTO)
    openalex_id, paper_dict = _enrich_via_openalex(
        record, mailto=mailto, filename=filename, file_hash=file_hash,
    )
    log.info("vault.upload.openalex", elapsed_ms=round((_time.monotonic() - t2) * 1000, 1))

    # Another user may have uploaded this paper already. Its stored text and
    # embedding are what their seeds are scored with, so a second copy of
    # the PDF does not rewrite them; this user gets their own upload row.
    prior = papers_repo.get_by_openalex_id(conn, openalex_id)
    keep_text = prior is not None and bool((prior["body_text"] or "").strip())
    paper_dict.update({
        "file_hash": file_hash,
        "uploaded_by_user_id": user_id,
        "n_pages": record.n_pages,
        "local_path": str(pdf_path),
        "body_text": None if keep_text else record.body_text,
    })
    if content_topics.is_local(openalex_id) and not (prior is not None and prior["topics_json"]):
        # OpenAlex does not have it, so nothing says what it is about: read
        # that from the PDF itself (rag_lib.content_topics).
        found = content_topics.try_topics_json(paper_dict.get("title"), record.body_text, settings=settings)
        if found is not None:
            paper_dict["topics_json"] = found
    papers_repo.upsert(conn, paper_dict)
    papers_repo.record_upload(
        conn, user_id=user_id, openalex_id=openalex_id, file_hash=file_hash,
        local_path=str(pdf_path), n_pages=record.n_pages,
    )

    t3 = _time.monotonic()
    embedding_model = settings.RADAR_DEFAULT_EMBEDDING_MODEL
    stored = papers_repo.get_by_openalex_id(conn, openalex_id)
    paper = Paper(
        doi=paper_dict.get("doi"),
        openalex_id=openalex_id,
        title=paper_dict.get("title") or record.title,
        abstract=paper_dict.get("abstract") or "",
        year=paper_dict.get("year"),
        venue=paper_dict.get("venue"),
        body_text=stored["body_text"] if keep_text else record.body_text,
    )
    if not (keep_text and embeddings_repo.get(conn, openalex_id, embedding_model) is not None):
        text = build_embedding_input(paper)
        embedder = get_embedder(embedding_model)
        embeddings_repo.upsert(conn, openalex_id, embedding_model, embedder(text))
    log.info("vault.upload.embed", model=embedding_model, elapsed_ms=round((_time.monotonic() - t3) * 1000, 1))

    if profile_slug:
        _attach_to_profile(conn, user_id, profile_slug, openalex_id, target_id)

    # UMLS extraction is not started here. It used to run on a daemon
    # thread so the upload could return early, and the cost of that was
    # paid by whatever request came next: spaCy and the ollama embedding
    # calls hold the GIL, the event loop is where request bodies are
    # read, and a concurrent upload therefore stalled mid-stream. One
    # took 312 seconds and then failed as "uploaded file is empty",
    # which is a spectacularly unhelpful way to report contention. The
    # commented-out Chroma block below was disabled for the same reason.
    #
    # Nothing is lost by dropping it: _try_merge_umls_topics computes
    # whatever a profile's seeds are missing when step 3 asks for the
    # topics, which is the only place the result is read. That moves a
    # few seconds onto a deliberate click and takes them off an upload
    # that has no idea why it is slow.

    row = papers_repo.get_by_openalex_id(conn, openalex_id)

    # Chroma indexing deferred — runs outside the request so the upload
    # returns fast.  Disabled for now: the background thread holds a
    # separate SQLite connection and the per-chunk Ollama calls take
    # seconds each, which starves every other request of the DB write
    # lock.  Re-enable once Chroma uses its own DB or the indexer is
    # batched.
    # _defer_heavy_indexing(
    #     settings=settings,
    #     user_id=user_id,
    #     openalex_id=openalex_id,
    #     body_text=record.body_text or paper_dict.get("abstract") or "",
    # )

    log.info(
        "vault.upload.done",
        openalex_id=openalex_id,
        total_ms=round((_time.monotonic() - t0) * 1000, 1),
    )
    return _row_to_vault_doc(conn, user_id, row, settings=settings)


def list_docs(
    conn: sqlite3.Connection,
    user_id: int,
    *,
    tag: str | None = None,
    settings: Any | None = None,
) -> list[dict]:
    rows = vault_repo.list_for_user(conn, user_id, tag=tag)
    # Resolve the user's Chroma collection once and pass it through so
    # we don't re-open the persistent client per row.
    collection = _try_open_user_collection(settings, user_id) if settings else None
    return [
        _row_to_vault_doc(conn, user_id, r, settings=settings, collection=collection)
        for r in rows
    ]


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


def _attach_to_profile(
    conn: sqlite3.Connection, user_id: int, profile_slug: str, openalex_id: str,
    profile_id: int | None = None,
) -> None:
    """Idempotent profile_seeds upsert onto the interest ``upload`` checked
    at its start (``profile_id``).

    Raises ``UploadRejected`` (404) when that interest was deleted while the
    paper was read and embedded: the upload answered 200 and no seed existed
    anywhere -- or, the name re-used meanwhile, the seed went to the new one.
    The paper itself is stored by then and stays in the vault.

    A live interest is re-fitted when this adds a seed, same as
    ``POST /{key}/seeds``: the Seeds tab's PDF drop zone lands here, and
    without it an uploaded seed was listed but never scored against.
    """
    gone = UploadRejected(
        f"'{profile_slug}' was deleted while the paper uploaded. The paper is in your vault, not a seed.",
        status_code=404,
    )
    profile = profiles_repo.get_by_slug(conn, user_id, profile_slug)
    if profile is None or (profile_id is not None and int(profile["id"]) != profile_id):
        raise gone
    try:
        profiles_repo.attach_seed(conn, int(profile["id"]), openalex_id)
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        raise gone from exc
    from . import wizard as wizard_service

    # The upload embedded under the default model; a draft under another
    # model (the default changed since it was created) had a seed with no
    # vector, which coherence then counted as no seed at all.
    wizard_service.embed_missing(conn, profile["embedding_model"], [openalex_id])
    # Even when the seed was already there: re-uploading the PDF is then how
    # a refit that failed after an earlier attach gets another go.
    if not profile["is_draft"]:
        try:
            wizard_service.refit_profile(conn, user_id=user_id, slug=profile_slug)
        except (wizard_service.ScanRunning, ValueError) as exc:
            # The upload is stored and the seed attached; what did not
            # happen is the refit (a scan is running, or nothing embeds).
            # That was a 500 for an upload that had succeeded.
            log.warning("vault.upload.refit_skipped", profile=profile_slug, reason=str(exc)[:160])


def _embed_if_missing(conn: sqlite3.Connection, row: sqlite3.Row, settings: Any) -> None:
    """Embed a stored paper that has no vector under the default model."""
    model = settings.RADAR_DEFAULT_EMBEDDING_MODEL
    oa = row["openalex_id"]
    if embeddings_repo.get(conn, oa, model) is not None:
        return
    paper = Paper(
        doi=row["doi"], openalex_id=oa, title=row["title"] or "",
        abstract=row["abstract"] or "", year=row["year"], venue=row["venue"],
        body_text=row["body_text"],
    )
    embeddings_repo.upsert(conn, oa, model, get_embedder(model)(build_embedding_input(paper)))
    log.info("vault.upload.embedded_on_retry", openalex_id=oa, model=model)


def _enrich_via_openalex(
    record,
    *,
    mailto: str,
    filename: str,
    file_hash: str,
) -> tuple[str, dict]:
    """Try DOI then title. Returns ``(openalex_id, paper_dict)``.

    Failure is non-fatal: we still produce a paper_dict suitable for
    ``papers.upsert`` with a synthetic ``local:<hash>`` id and
    ``source='user_pdf'``. Authors land on ``paper_dict['authors']``
    when OpenAlex resolves the work.
    """
    paper_dict: dict[str, Any] = {
        "title": record.title or filename,
        "doi": record.doi,
        "source": "user_pdf",
    }
    try:
        client = OpenAlexClient(mailto=mailto)
        work = None
        if record.doi:
            work = client.lookup_by_doi(record.doi)
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
            work = _verified_title_match(record, client.search_by_title(record.title))
        if work is None and getattr(record, "doi_from_body", False):
            paper_dict["doi"] = None     # not vouched for by anything: likely a reference's
        if work is not None:
            paper = client.paper_from_work(work, source="user_pdf")
            paper_dict["openalex_id"] = paper.openalex_id
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
    except Exception:
        # Network / OpenAlex outage shouldn't kill the upload — fall
        # through to the synthetic-id branch below.
        pass

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
_HEAD_CHARS = 6000
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
    return _squash(_work_title(work)) in _squash(_front_matter(record.body_text))


_REFERENCES_HEADING = re.compile(r"\n\s*(?:\d+\.?\s*)?(?:references|bibliography|literature cited)\s*\n", re.IGNORECASE)


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
    text = (record.body_text or "")[:_HEAD_CHARS]
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
        author = entry.get("author") or {}
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
            min_similarity=settings.RADAR_UMLS_MIN_TOPIC_SIMILARITY,
            embedding_model=settings.RADAR_UMLS_EMBEDDING_MODEL,
            cache_dir=cache_dir,
        )
        mapped_json = json.dumps([m.to_dict() for m in mapped]) if mapped else None

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
