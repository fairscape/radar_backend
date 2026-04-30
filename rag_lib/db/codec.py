"""Vector ↔ BLOB codec for embeddings.

Embeddings live in ``paper_embeddings.vector`` as raw little-endian
float32 bytes. numpy is the only consumer that needs to read them; we
keep the codec in one place so Phase 5 (API mappers) and Phase 9 (RAG
indexer) speak the same wire format.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def encode_vector(vec: Iterable[float] | np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def decode_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
