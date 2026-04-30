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
