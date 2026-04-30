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
        return []

    scope_list = [s for s in (scope or []) if s]
    where = _scope_where_clause(scope_list)

    kwargs: dict[str, Any] = {
        "n_results": max(1, int(k)),
        "include": ["documents", "metadatas", "distances"],
    }
    if where is not None:
        kwargs["where"] = where

    if embedder is not None:
        kwargs["query_embeddings"] = [embedder(query)]
    else:
        kwargs["query_texts"] = [query]

    raw = collection.query(**kwargs)

    documents = _first_row(raw, "documents")
    metadatas = _first_row(raw, "metadatas")
    distances = _first_row(raw, "distances")

    out: list[dict] = []
    for text, meta, dist in zip(documents, metadatas, distances):
        meta = meta or {}
        out.append({
            "text": text or "",
            "title": str(meta.get("title") or ""),
            "openalex_id": str(meta.get("openalex_id") or ""),
            "score": _distance_to_score(dist),
        })
    return out


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _scope_where_clause(slugs: list[str]) -> dict | None:
    """Build a Chroma ``where`` clause matching any of ``slugs``.

    Each slug is searched as the substring ``"|slug|"`` against the
    delimiter-padded ``profile_slugs`` metadata. Multiple slugs are
    joined with ``$or``. Empty list returns ``None`` (= search all).
    """
    if not slugs:
        return None
    if len(slugs) == 1:
        return {"profile_slugs": {"$contains": f"|{slugs[0]}|"}}
    return {
        "$or": [
            {"profile_slugs": {"$contains": f"|{s}|"}} for s in slugs
        ]
    }


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
