"""Disk cache for the OpenAlex topic embedding index.

Loads the precomputed topic metadata + embedding matrix created by
``scripts/build_topic_index.py``. The index is loaded lazily on first
access and cached in memory for the lifetime of the process.

The vectors are only comparable with concept vectors from the *same*
embedder. The index files used to record nothing about it, and the build
script defaulted to specter2 (768-d) while the mapper embeds with
``RADAR_UMLS_EMBEDDING_MODEL`` (mxbai-embed-large, 1024-d): a rebuild with
the defaults made every search raise inside the extractor, which swallowed
it, so no paper got UMLS topics and every seed was retried every sweep. A
same-dimension wrong model would have matched nonsense without an error.
The build now writes ``META_FILE`` naming its model, and both a named
mismatch and a dimension mismatch raise ``TopicIndexMismatch``.

Two indexes, the same topics:
- by name (``get_topic_index``): the topics compared with each other, to
  leave out one an interest already has (topic_mapper._is_near_duplicate);
- by name, description and keywords (``get_rich_topic_index``, CTX-2):
  what a concept's sentence is matched against. A name is a few words; the
  description OpenAlex writes says what the papers under it are about
  (records/35). Built with ``build_topic_index.py --rich``.
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
_rich_index: TopicIndex | None = None
_index_lock = Lock()

TOPICS_FILE = "openalex_topics.json"
EMBEDDINGS_FILE = "openalex_topic_embeddings.npz"
META_FILE = "openalex_topic_index.meta.json"
RICH_TOPICS_FILE = "openalex_topics_rich.json"
RICH_EMBEDDINGS_FILE = "openalex_topic_rich_embeddings.npz"
RICH_META_FILE = "openalex_topic_rich_index.meta.json"


def rich_topic_text(topic: dict) -> str:
    """What a topic is embedded as in the rich index: its name, OpenAlex's
    description of it and its keywords, the higher-scored first
    ("Zoonotic diseases and public health. This cluster of papers focuses
    on ... Keywords: disease transmission from animals to humans; ...")."""
    keywords = sorted(topic.get("keywords") or [], key=lambda k: -(k.get("score") or 0))
    text = f'{topic.get("display_name") or ""}. {topic.get("description") or ""}'.strip()
    words = "; ".join(k.get("display_name") or "" for k in keywords if k.get("display_name"))
    return f"{text} Keywords: {words}" if words else text


class TopicIndexMismatch(ValueError):
    """The topic index was built with a different embedder than the one asked for."""


class TopicIndex:
    """In-memory OpenAlex topic index for cosine similarity search."""

    def __init__(self, topics: list[dict], embeddings: np.ndarray, *, embedding_model: str | None = None,
                 rebuild: str = "python scripts/build_topic_index.py"):
        # None for an index built before META_FILE existed.
        self.embedding_model = embedding_model
        self.rebuild = rebuild
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

    def check_compatible(self, embedding_model: str | None, dim: int | None = None) -> None:
        """Raise ``TopicIndexMismatch`` unless vectors from ``embedding_model``
        (of length ``dim``, when known) can be searched against this index."""
        if embedding_model and self.embedding_model and embedding_model != self.embedding_model:
            raise TopicIndexMismatch(
                f"topic index was built with {self.embedding_model!r} but concepts are embedded "
                f"with {embedding_model!r}; rebuild it: "
                f"{self.rebuild} --embedding-model {embedding_model}"
            )
        if dim is not None and dim != self.dim:
            raise TopicIndexMismatch(
                f"topic index has {self.dim}-d vectors but {embedding_model or 'the embedder'} "
                f"gives {dim}-d; rebuild it: "
                f"{self.rebuild} --embedding-model {embedding_model}"
            )

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


def _load(cache_dir: str | Path, topics_file: str, embeddings_file: str, meta_file: str,
          rebuild: str) -> TopicIndex:
    cache_dir = Path(cache_dir)
    topics_path = cache_dir / topics_file
    embeddings_path = cache_dir / embeddings_file

    if not topics_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(f"Topic index not found in {cache_dir}. Run: {rebuild}")

    log.info("Loading OpenAlex topic index %s from %s", topics_file, cache_dir)
    with open(topics_path, encoding="utf-8") as fh:
        topics = json.load(fh)
    embeddings = np.load(embeddings_path)["embeddings"]
    meta_path = cache_dir / meta_file
    built_with = None
    if meta_path.exists():
        built_with = json.loads(meta_path.read_text(encoding="utf-8")).get("embedding_model")
    else:
        log.warning("Topic index in %s has no %s; cannot check which model built it", cache_dir, meta_file)

    if embeddings.shape[0] != len(topics):
        raise ValueError(
            f"Topic count mismatch: {len(topics)} topics vs "
            f"{embeddings.shape[0]} embeddings"
        )

    index = TopicIndex(topics, embeddings, embedding_model=built_with, rebuild=rebuild)
    log.info("Topic index loaded: %d topics, %dd vectors", len(topics), index.dim)
    return index


def get_topic_index(cache_dir: str | Path) -> TopicIndex:
    """Load (or return cached) the by-name topic index from disk."""
    global _index
    if _index is not None:
        return _index

    with _index_lock:
        if _index is None:
            _index = _load(cache_dir, TOPICS_FILE, EMBEDDINGS_FILE, META_FILE,
                           "python scripts/build_topic_index.py")
        return _index


def rich_index_problem(cache_dir: str | Path, embedding_model: str | None) -> str | None:
    """Why the UMLS mapper cannot search the rich index (not built, built
    with another embedder, unreadable), or None when it can. Asked before
    old UMLS data is sent back to extraction, at start-up and by
    /api/health (review CTX-R1: without the index every mapping fails)."""
    try:
        get_rich_topic_index(cache_dir).check_compatible(embedding_model)
    except Exception as exc:  # noqa: BLE001 -- missing file, mismatch, bad JSON, bad npz
        return f"{type(exc).__name__}: {exc}"[:300]
    return None


def get_rich_topic_index(cache_dir: str | Path) -> TopicIndex:
    """Load (or return cached) the by-description topic index (CTX-2)."""
    global _rich_index
    if _rich_index is not None:
        return _rich_index

    with _index_lock:
        if _rich_index is None:
            _rich_index = _load(cache_dir, RICH_TOPICS_FILE, RICH_EMBEDDINGS_FILE, RICH_META_FILE,
                                "python scripts/build_topic_index.py --rich")
        return _rich_index
