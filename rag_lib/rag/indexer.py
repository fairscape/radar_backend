"""Indexer: chunk vault docs and push them into a Chroma collection.

Two layers:

  - ``chunk_text`` is a pure-Python sliding window on whitespace tokens
    (target 500 / overlap 50 by default). No tokenizer dependency so
    the demo runs without phase1b extras; the chunk size is approximate
    in real BPE tokens and good enough for SPECTER2 retrieval at this
    scale.

  - ``index_paper`` calls ``collection.add`` once per chunk with the
    chunk text, its embedding, and metadata (``openalex_id``, ``title``,
    ``profile_slugs``). Chroma uses the ``where`` clause on
    ``profile_slugs`` at query time for scope filtering.

  - ``index_user_collection`` resolves (creating on first call) the
    per-user PersistentClient collection at ``RADAR_CHROMA_DIR/<user_id>``.
    chromadb is a phase1b extra; the import is deferred so this module
    stays importable without it.

The indexer accepts an embedder callable so tests can pass a stub and
the upload path can stay aligned with ``settings.RADAR_DEFAULT_EMBEDDING_MODEL``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Callable, Iterable


Embedder = Callable[[str], list[float]]


def chunk_text(
    text: str,
    target_tokens: int = 500,
    overlap: int = 50,
    token_count_fn: Callable[[str], int] | None = None,
) -> list[str]:
    """Split ``text`` into overlapping windows of approximately
    ``target_tokens`` tokens with ``overlap`` shared on each boundary.

    Tokens here are whitespace-separated words — close enough to BPE
    counts for the windowing math at the demo scale. ``token_count_fn``
    is accepted so a SPECTER2-aware caller can pass a real tokenizer
    if needed; the default counts whitespace splits.

    Empty / whitespace-only input returns ``[]``.
    Inputs shorter than ``target_tokens`` return one chunk verbatim.
    """
    if not text or not text.strip():
        return []
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if overlap < 0 or overlap >= target_tokens:
        raise ValueError("overlap must be in [0, target_tokens)")

    words = text.split()
    if not words:
        return []
    if len(words) <= target_tokens:
        return [" ".join(words)]

    step = target_tokens - overlap
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + target_tokens, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    # token_count_fn is currently advisory; reserved for a Phase-13
    # tokenizer-aware refinement.
    _ = token_count_fn
    return chunks


def index_paper(
    collection: Any,
    paper_row: sqlite3.Row | dict,
    embedder: Embedder,
    *,
    profile_slugs: Iterable[str] | None = None,
    target_tokens: int = 500,
    overlap: int = 50,
) -> int:
    """Chunk ``paper_row.body_text`` and write each chunk to ``collection``.

    Chroma's ``where`` clause does not support list overlap natively, so
    profile slugs are joined into a delimiter-padded string
    (``"|slug1|slug2|"``) and queries match with ``$contains "|slug|"``.
    See ``retriever.retrieve`` for the matching read side.

    Returns the number of chunks indexed (0 on missing body / empty text).
    Re-indexing the same paper is idempotent: chunk ids are deterministic
    on ``openalex_id + chunk_index`` and ``collection.add`` upserts.
    """
    body = _row_field(paper_row, "body_text") or ""
    chunks = chunk_text(body, target_tokens=target_tokens, overlap=overlap)
    if not chunks:
        return 0

    openalex_id = str(_row_field(paper_row, "openalex_id"))
    title = _row_field(paper_row, "title") or ""
    slugs = list(profile_slugs or [])

    ids = [f"{openalex_id}::chunk::{i:04d}" for i in range(len(chunks))]
    embeddings = [embedder(c) for c in chunks]
    metadatas = [
        {
            "openalex_id": openalex_id,
            "title": title,
            # Chroma 0.5 metadata values must be scalars; serialize the
            # slug list as a delimiter-padded string. Empty slug list is
            # encoded as the empty string so retrieval-by-substring still
            # matches "anything" via a missing $contains filter.
            "profile_slugs": _encode_slugs(slugs),
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=chunks,
    )
    return len(chunks)


def index_user_collection(settings: Any, user_id: int) -> Any:
    """Return the per-user Chroma collection (creating on first call).

    Each user gets their own PersistentClient at
    ``RADAR_CHROMA_DIR/<user_id>``; collection name is fixed at
    ``vault``. chromadb is a phase1b extra — ``ImportError`` here is
    surfaced to the caller so the upload / chat path can return a
    targeted 503.
    """
    try:
        import chromadb  # type: ignore
    except ImportError as e:  # pragma: no cover — exercised only when extras absent
        raise ImportError(
            "Chroma vector index unavailable. Install with: "
            "pip install -e '.[dev,phase1b]'"
        ) from e

    base = Path(settings.RADAR_CHROMA_DIR) / str(user_id)
    base.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(base))
    return client.get_or_create_collection(name="vault")


def collection_size(collection: Any) -> int:
    """Best-effort chunk count for a Chroma collection.

    Used to populate ``VaultStats.chunks``. Chroma's ``count`` returns an
    int; we wrap defensively because tests can pass a stub.
    """
    try:
        return int(collection.count())
    except Exception:
        return 0


def count_chunks_for_paper(collection: Any, openalex_id: str) -> int:
    """Best-effort chunk count for a single paper inside a collection.

    Used to populate ``VaultDoc.chunks``. We use ``collection.get`` with
    a metadata filter rather than ``count`` because Chroma's ``count``
    is unfiltered. Returns 0 on any error so a Chroma outage doesn't
    break the vault listing.
    """
    if not openalex_id:
        return 0
    try:
        result = collection.get(where={"openalex_id": str(openalex_id)})
    except Exception:
        return 0
    if not isinstance(result, dict):
        return 0
    ids = result.get("ids") or []
    return len(ids)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _row_field(row: sqlite3.Row | dict, name: str) -> Any:
    if row is None:
        return None
    if isinstance(row, dict):
        return row.get(name)
    try:
        return row[name]
    except (IndexError, KeyError):
        return None


def _encode_slugs(slugs: Iterable[str]) -> str:
    """Encode a list of profile slugs as ``"|s1|s2|"``.

    Paired with ``"|slug|"`` substring matching on the read side, this
    gives us an "any-of" filter without storing a separate row per slug
    or relying on Chroma list-metadata support.
    """
    cleaned = [s for s in (str(x) for x in slugs) if s]
    if not cleaned:
        return ""
    return "|" + "|".join(cleaned) + "|"
