"""Disk cache for the OpenAlex topic embedding index.

Loads the precomputed topic metadata + embedding matrix created by
``scripts/build_topic_index.py``. The index is loaded lazily on first
access and cached in memory for the lifetime of the process.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton index
# ---------------------------------------------------------------------------

_index: TopicIndex | None = None
_index_lock = Lock()


class TopicIndex:
    """In-memory OpenAlex topic index for cosine similarity search."""

    def __init__(self, topics: list[dict], embeddings: np.ndarray):
        self.topics = topics  # [{id, display_name, subfield, field, domain}, ...]
        # Row-normalize for fast cosine via dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        self.embeddings = (embeddings / norms).astype(np.float32)
        self.dim = self.embeddings.shape[1]
        self._row_by_id = {
            t["id"]: i for i, t in enumerate(topics) if t.get("id")
        }

    def vector_for(self, topic_id: str) -> np.ndarray | None:
        """Row-normalized embedding for an OpenAlex topic id.

        Returns ``None`` when the id isn't in the index. Rows are already
        unit-length, so the dot product of two of them *is* the cosine —
        callers can compare topics to each other without re-embedding.
        """
        row = self._row_by_id.get(topic_id)
        return None if row is None else self.embeddings[row]

    def search(
        self, query_vec: np.ndarray, *, top_k: int = 3, min_similarity: float = 0.40
    ) -> list[tuple[dict, float]]:
        """Return top-k topics by cosine similarity to query_vec.

        Returns list of (topic_dict, similarity) sorted descending.
        """
        qn = np.linalg.norm(query_vec) + 1e-12
        q = (query_vec / qn).astype(np.float32)
        scores = self.embeddings @ q  # (N,)
        # Get top-k indices
        if top_k < len(scores):
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
        else:
            top_indices = np.arange(len(scores))
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            sim = float(scores[idx])
            if sim < min_similarity:
                break
            results.append((self.topics[idx], sim))
        return results


def get_topic_index(cache_dir: str | Path) -> TopicIndex:
    """Load (or return cached) topic index from disk."""
    global _index
    if _index is not None:
        return _index

    with _index_lock:
        if _index is not None:
            return _index

        cache_dir = Path(cache_dir)
        topics_path = cache_dir / "openalex_topics.json"
        embeddings_path = cache_dir / "openalex_topic_embeddings.npz"

        if not topics_path.exists() or not embeddings_path.exists():
            raise FileNotFoundError(
                f"Topic index not found in {cache_dir}. "
                f"Run: python scripts/build_topic_index.py"
            )

        log.info("Loading OpenAlex topic index from %s", cache_dir)
        with open(topics_path, encoding="utf-8") as fh:
            topics = json.load(fh)
        embeddings = np.load(embeddings_path)["embeddings"]

        if embeddings.shape[0] != len(topics):
            raise ValueError(
                f"Topic count mismatch: {len(topics)} topics vs "
                f"{embeddings.shape[0]} embeddings"
            )

        _index = TopicIndex(topics, embeddings)
        log.info("Topic index loaded: %d topics, %dd vectors", len(topics), _index.dim)
        return _index
