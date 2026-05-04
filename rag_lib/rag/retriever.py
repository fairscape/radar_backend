"""Retriever: query a Chroma collection scoped to a list of profiles.

Scope filtering uses the encoding from ``indexer._encode_slugs``: each
chunk's ``profile_slugs`` metadata is a delimiter-padded string like
``"|neonatal-vitals|sepsis|"`` and queries match with a substring
``"|<slug>|"``. Chroma's ``$or`` over ``$contains`` gives us union
("any-of") semantics with one query.

Chroma scores are distances (smaller = closer); we convert to a
``score`` in ``[0, 1]`` (1 = perfect match) using ``1 - distance``
clamped to non-negative. SPECTER2 + cosine distance lives in [0, 2];
the clamp protects callers from negative similarity surprises while
keeping the relative ordering intact.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import structlog


log = structlog.get_logger("rag_lib.rag.retriever")

Embedder = Callable[[str], list[float]]


def retrieve(
    collection: Any,
    query: str,
    *,
    scope: Iterable[str] | None = None,
    k: int = 10,
    embedder: Embedder | None = None,
) -> list[dict]:
    """Return top-``k`` chunks matching ``query`` for the given scope.

    ``scope`` is a list of profile slugs the chunk must overlap; an
    empty / ``None`` list searches across the whole user collection.
    ``embedder`` is required when the collection isn't configured with
    its own embedding function — the chat path passes the same callable
    the indexer used so the vector spaces line up.

    Returns a list of dicts with ``text``, ``title``, ``score`` (higher =
    more relevant), and ``openalex_id``. The list is empty (not an
    exception) when the collection holds no matching chunks.
    """
    if not query or not query.strip():
        log.info("retrieve.empty_query")
        return []

    scope_list = [s for s in (scope or []) if s]
    where = _scope_where_clause(scope_list)

    target_k = max(1, int(k))
    max_per_paper = 4
    # Pull a wider candidate pool than the caller asked for so the
    # per-paper cap below has room to drop near-duplicates without
    # starving the final result. Embedders cluster sibling chunks from
    # the same paper, so without diversification the top-k is often
    # dominated by 5–9 slices of one paper, crowding out the actual
    # answer that lives in a different paper.
    n_results = target_k * 4

    kwargs: dict[str, Any] = {
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        kwargs["where"] = where

    query_vec = None
    if embedder is not None:
        query_vec = embedder(query)
        kwargs["query_embeddings"] = [query_vec]
    else:
        kwargs["query_texts"] = [query]

    log.info(
        "retrieve.start",
        query_preview=query[:120],
        scope=scope_list,
        k=kwargs["n_results"],
        where=where,
        embedder_used=embedder is not None,
        query_vec_dim=(len(query_vec) if query_vec is not None else None),
    )

    try:
        raw = collection.query(**kwargs)
    except Exception as exc:
        log.error(
            "retrieve.chroma_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
            where=where,
        )
        raise

    documents = _first_row(raw, "documents")
    metadatas = _first_row(raw, "metadatas")
    distances = _first_row(raw, "distances")

    log.info(
        "retrieve.raw",
        n_documents=len(documents),
        n_metadatas=len(metadatas),
        n_distances=len(distances),
        raw_keys=sorted(list(raw.keys())) if isinstance(raw, dict) else None,
    )

    # Diversify: walk candidates in best-score order; keep up to
    # ``max_per_paper`` chunks per ``openalex_id`` until we hit
    # ``target_k`` total. Skipped near-duplicates are recorded so the
    # log makes the trim visible.
    out: list[dict] = []
    per_paper: dict[str, int] = {}
    skipped_dupes = 0
    for text, meta, dist in zip(documents, metadatas, distances):
        meta = meta or {}
        oa = str(meta.get("openalex_id") or "")
        if per_paper.get(oa, 0) >= max_per_paper:
            skipped_dupes += 1
            continue
        out.append({
            "text": text or "",
            "title": str(meta.get("title") or ""),
            "openalex_id": oa,
            "score": _distance_to_score(dist),
        })
        per_paper[oa] = per_paper.get(oa, 0) + 1
        if len(out) >= target_k:
            break

    log.info(
        "retrieve.done",
        n_results=len(out),
        target_k=target_k,
        candidates_pulled=len(documents),
        skipped_same_paper=skipped_dupes,
        max_per_paper=max_per_paper,
        papers_represented=len(per_paper),
        top_hits=[
            {
                "title": r["title"][:80],
                "openalex_id": r["openalex_id"],
                "score": round(r["score"], 4),
                "chunk_chars": len(r["text"]),
            }
            for r in out[:5]
        ],
    )
    return out


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _scope_where_clause(slugs: list[str]) -> dict | None:
    """Build a Chroma ``where`` clause matching any of ``slugs``.

    Chroma 0.5+ does not support ``$contains`` on metadata ``where`` —
    that operator is only valid on ``where_document``. We match the
    encoded ``"|slug|"`` value with ``$in`` instead, which gives the
    same any-of semantics for the common case where each chunk's
    ``profile_slugs`` is a single ``"|slug|"`` literal. Multi-tagged
    chunks (``"|a|b|"``) are not supported by this scheme — switch to
    per-slug boolean metadata if that becomes a real case.
    """
    if not slugs:
        return None
    return {"profile_slugs": {"$in": [f"|{s}|" for s in slugs]}}


def _first_row(result: dict, key: str) -> list:
    """Chroma's ``query`` returns each field as a list-of-lists keyed
    by query batch. We always send a single query, so unwrap the outer
    list and tolerate a missing key by returning an empty list."""
    val = result.get(key) if isinstance(result, dict) else None
    if not val:
        return []
    if isinstance(val, list) and val and isinstance(val[0], list):
        return val[0]
    return list(val)


def _distance_to_score(distance: Any) -> float:
    """Map a Chroma distance to ``[0, 1]`` (1 = perfect)."""
    if distance is None:
        return 0.0
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return 0.0
    score = 1.0 - d
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score
