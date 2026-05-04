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
    file_hash = compute_file_hash(data)

    existing = papers_repo.get_by_user_hash(conn, user_id, file_hash)
    if existing is not None:
        if profile_slug:
            _attach_to_profile(conn, user_id, profile_slug, existing["openalex_id"])
        return _row_to_vault_doc(conn, user_id, existing, settings=settings)

    # Persist the bytes content-addressed; keep the original filename
    # on papers.local_path for display.
    pdf_dir = _vault_dir_for_user(settings, user_id)
    pdf_path = pdf_dir / f"{file_hash}.pdf"
    pdf_path.write_bytes(data)

    record = ingest_pdf(pdf_path)

    # Best-effort OpenAlex enrichment. On any failure we keep going with a
    # synthetic openalex_id so the upload still lands as a valid paper row.
    mailto = settings.RADAR_DEFAULT_MAILTO
    openalex_id, paper_dict = _enrich_via_openalex(
        record, mailto=mailto, filename=filename, file_hash=file_hash,
    )
    paper_dict.update({
        "file_hash": file_hash,
        "uploaded_by_user_id": user_id,
        "n_pages": record.n_pages,
        "local_path": str(pdf_path),
        "body_text": record.body_text,
    })
    papers_repo.upsert(conn, paper_dict)

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

    if profile_slug:
        _attach_to_profile(conn, user_id, profile_slug, openalex_id)

    # Push chunks into the user's Chroma collection so /api/chat can
    # retrieve them. Best-effort: a missing chromadb extra (or a Chroma
    # outage) shouldn't fail the upload — chat will surface its own 503
    # later if Chroma stays unavailable.
    row = papers_repo.get_by_openalex_id(conn, openalex_id)
    profile_slugs = vault_repo.tags_for_paper(conn, user_id, openalex_id)
    try:
        collection = rag_indexer.index_user_collection(settings, user_id)
        rag_indexer.index_paper(
            collection,
            row,
            embedder,
            profile_slugs=profile_slugs,
        )
    except ImportError:
        collection = None
    except Exception:
        # Defensive: a Chroma write error shouldn't unwind the upload
        # transaction. The paper is in SQLite; the operator can re-run
        # an indexing pass later.
        collection = None

    # Parallel chat-retrieval index. Best-effort: ollama may be down or
    # the embedder model not pulled — log and skip. SPECTER2 indexing
    # above is unaffected, so the upload still succeeds.
    chat_model = (settings.RADAR_CHAT_EMBEDDING_MODEL or "").strip()
    if chat_model:
        try:
            chat_embedder = get_embedder(chat_model)
            chat_collection = rag_indexer.index_user_chat_collection(
                settings, user_id
            )
            # Smaller windows: mxbai-embed-large / nomic-embed-text /
            # bge-large all cap at 512 tokens. PDF-extracted text often
            # tokenizes to 1.5–3 tokens per word (URLs, formulas, fused
            # words), so a 500-word window can blow past the limit and
            # ollama returns 500. ~300 words keeps us safely under 512
            # BPE tokens for typical English text.
            rag_indexer.index_paper(
                chat_collection,
                row,
                chat_embedder,
                profile_slugs=profile_slugs,
                target_tokens=300,
                overlap=40,
            )
        except Exception as exc:
            log.warning(
                "vault.upload.chat_index_skipped",
                openalex_id=openalex_id,
                chat_embedder=chat_model,
                reason=type(exc).__name__,
                detail=str(exc)[:200],
            )

    return _row_to_vault_doc(
        conn, user_id, row, settings=settings, collection=collection,
    )


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
