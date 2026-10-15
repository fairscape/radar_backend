"""paper_embeddings repo.

Many embeddings per paper, one per model. Vectors are float32 BLOBs
(see ``rag_lib.db.codec``). ``has`` is the cheap check the gather path
uses to skip re-embedding papers it already has.
"""

from __future__ import annotations

import sqlite3
from typing import Iterable

import numpy as np

from ..codec import decode_vector, encode_vector


def upsert(
    conn: sqlite3.Connection,
    openalex_id: str,
    model: str,
    vector: Iterable[float] | np.ndarray,
) -> None:
    blob = encode_vector(vector)
    conn.execute(
        """
        INSERT INTO paper_embeddings (openalex_id, embedding_model, vector)
        VALUES (?, ?, ?)
        ON CONFLICT(openalex_id, embedding_model) DO UPDATE SET
          vector      = excluded.vector,
          computed_at = datetime('now')
        """,
        (openalex_id, model, blob),
    )
    conn.commit()


def get(
    conn: sqlite3.Connection, openalex_id: str, model: str
) -> np.ndarray | None:
    row = conn.execute(
        "SELECT vector FROM paper_embeddings WHERE openalex_id = ? AND embedding_model = ?",
        (openalex_id, model),
    ).fetchone()
    if row is None:
        return None
    return decode_vector(row["vector"])


def has(conn: sqlite3.Connection, openalex_id: str, model: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM paper_embeddings WHERE openalex_id = ? AND embedding_model = ? LIMIT 1",
        (openalex_id, model),
    ).fetchone()
    return row is not None


def models_for(conn: sqlite3.Connection, openalex_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT embedding_model FROM paper_embeddings WHERE openalex_id = ?",
        (openalex_id,),
    ).fetchall()
    return [r["embedding_model"] for r in rows]


def attach_stored(conn: sqlite3.Connection, papers: list, model: str) -> int:
    """Give each paper the vector already stored for it, if any. Returns how many.

    A gather re-embedded every paper it fetched, and ``store_papers`` then
    kept the vector stored the first time (``skip_re_embed``). Embedding is
    not bit-stable, so the score written for a resurfaced paper came from a
    vector that was not stored, and a refit rescoring from stored vectors
    moved it (by up to 0.017). Scoring with the stored vector keeps the two
    the same, and skips the embedding.
    """
    ids = [p.openalex_id for p in papers if p.openalex_id and model not in (p.embeddings or {})]
    found = 0
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        marks = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"SELECT openalex_id, vector FROM paper_embeddings "
            f"WHERE embedding_model = ? AND openalex_id IN ({marks})",
            (model, *chunk),
        ).fetchall()
        vecs = {r[0]: r[1] for r in rows}
        for p in papers:
            blob = vecs.get(p.openalex_id)
            if blob is not None and model not in (p.embeddings or {}):
                p.embeddings[model] = [float(x) for x in decode_vector(blob)]
                found += 1
    return found
