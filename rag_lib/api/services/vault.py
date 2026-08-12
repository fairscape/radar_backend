"""Vault service — orchestration for the Phase 6 vault routes.

Functions here are router-thin: they take ``(db, settings, user_id, ...)``
and return Pydantic-shaped dicts the router serializes verbatim.

The upload path is the only write surface:
  1. Hash the bytes.
  2. Dedup against ``papers.uploaded_by_user_id + file_hash`` —
     a re-upload of the same file returns the existing VaultDoc.
  3. Write bytes to ``RADAR_VAULT_DIR/{user_id}/{sha256}.pdf``
     (content-addressed; the original filename is preserved on
     ``papers.local_path``).
  4. ``vault.ingest_pdf`` extracts title / body_text / DOI / n_pages.
  5. Best-effort OpenAlex enrichment: lookup_by_doi → lookup_by_title.
     On failure we still create a row with ``source='user_pdf'`` and a
     synthetic ``openalex_id = "local:<hash>"`` so dedup still works.
  6. Embed via the active model (``settings.RADAR_DEFAULT_EMBEDDING_MODEL``,
     defaults to ``placeholder-v1`` so the demo runs without phase1b).
  7. Optionally attach the paper to a profile (``profile_slug``)
     via ``profile_seeds`` so the doc shows up tagged.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import structlog


log = structlog.get_logger("rag_lib.api.services.vault")

from ...db.repos import (
    embeddings as embeddings_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    vault as vault_repo,
)
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
_UMLS_TEXT_LIMIT = 20_000


def _umls_input_text(abstract: str | None, body_text: str | None) -> str:
    """Pick the text to run UMLS extraction over: abstract first."""
    if abstract and abstract.strip():
        return abstract
    return (body_text or "")[:_UMLS_TEXT_LIMIT]


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

    file_hash = compute_file_hash(data)

    existing = papers_repo.get_by_user_hash(conn, user_id, file_hash)
    if existing is not None:
        if profile_slug:
            _attach_to_profile(conn, user_id, profile_slug, existing["openalex_id"])
        if not existing["umls_concepts_json"]:
            _oa_id = existing["openalex_id"]
            _text = _umls_input_text(existing["abstract"], existing["body_text"])
            def _bg():
                from ...db import connect as _connect
                try:
                    c = _connect(settings.RADAR_DB_PATH)
                    try:
                        _try_extract_umls(c, settings, _oa_id, _text)
                    finally:
                        c.close()
                except Exception:
                    pass
            threading.Thread(target=_bg, daemon=True).start()
        log.info(
            "vault.upload.dedup_hit",
            openalex_id=existing["openalex_id"],
            elapsed_ms=round((_time.monotonic() - t0) * 1000, 1),
        )
        return _row_to_vault_doc(conn, user_id, existing, settings=settings)

    pdf_dir = _vault_dir_for_user(settings, user_id)
    pdf_path = pdf_dir / f"{file_hash}.pdf"
    pdf_path.write_bytes(data)

    t1 = _time.monotonic()
    record = ingest_pdf(pdf_path)
    log.info("vault.upload.ingest_pdf", elapsed_ms=round((_time.monotonic() - t1) * 1000, 1))

    t2 = _time.monotonic()
    mailto = settings.RADAR_DEFAULT_MAILTO
    openalex_id, paper_dict = _enrich_via_openalex(
        record, mailto=mailto, filename=filename, file_hash=file_hash,
    )
    log.info("vault.upload.openalex", elapsed_ms=round((_time.monotonic() - t2) * 1000, 1))

    paper_dict.update({
        "file_hash": file_hash,
        "uploaded_by_user_id": user_id,
        "n_pages": record.n_pages,
        "local_path": str(pdf_path),
        "body_text": record.body_text,
    })
    papers_repo.upsert(conn, paper_dict)

    t3 = _time.monotonic()
    embedding_model = settings.RADAR_DEFAULT_EMBEDDING_MODEL
    paper = Paper(
        doi=paper_dict.get("doi"),
        openalex_id=openalex_id,
        title=paper_dict.get("title") or record.title,
        abstract=paper_dict.get("abstract") or "",
        year=paper_dict.get("year"),
        venue=paper_dict.get("venue"),
        body_text=record.body_text,
    )
    text = build_embedding_input(paper)
    embedder = get_embedder(embedding_model)
    embeddings_repo.upsert(conn, openalex_id, embedding_model, embedder(text))
    log.info("vault.upload.embed", model=embedding_model, elapsed_ms=round((_time.monotonic() - t3) * 1000, 1))

    if profile_slug:
        _attach_to_profile(conn, user_id, profile_slug, openalex_id)

    def _bg_umls():
        from ...db import connect as _connect
        try:
            c = _connect(settings.RADAR_DB_PATH)
            try:
                _try_extract_umls(
                    c, settings, openalex_id,
                    _umls_input_text(
                        paper_dict.get("abstract"), record.body_text
                    ),
                )
            finally:
                c.close()
        except Exception as exc:
            log.warning("vault.upload.bg_umls_failed", error=str(exc)[:200])
    threading.Thread(target=_bg_umls, daemon=True).start()

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
    conn: sqlite3.Connection, user_id: int, profile_slug: str, openalex_id: str
) -> None:
    """Idempotent profile_seeds upsert; silently no-ops if the slug is unknown."""
    profile = profiles_repo.get_by_slug(conn, user_id, profile_slug)
    if profile is None:
        return
    profiles_repo.attach_seed(conn, int(profile["id"]), openalex_id)


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
        if work is None and record.title:
            work = client.lookup_by_title(record.title)
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


def _authors_from_work(work: dict) -> list[str]:
    out: list[str] = []
    for entry in (work.get("authorships") or []):
        author = entry.get("author") or {}
        name = author.get("display_name")
        if name:
            out.append(name)
    return out


def _try_extract_umls(
    conn: sqlite3.Connection,
    settings: Any,
    openalex_id: str,
    text: str,
) -> None:
    """Best-effort UMLS extraction + topic mapping for a paper.

    Extracts UMLS concepts from the paper text, maps them to OpenAlex
    topics, and stores both as JSON on the paper row. Failures are logged
    and silently swallowed — UMLS enrichment is optional.
    """
    try:
        if not settings.RADAR_UMLS_ENABLED:
            return
        if not text or not text.strip():
            return

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
            return

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
    except Exception as exc:
        log.warning(
            "vault.umls_extraction_skipped",
            openalex_id=openalex_id,
            reason=type(exc).__name__,
            detail=str(exc)[:200],
        )
