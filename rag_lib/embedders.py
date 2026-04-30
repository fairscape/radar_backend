"""Embedders — callables of shape ``(text: str) -> list[float]``.

Phase 1A shipped a deterministic hash-based placeholder. Phase 1B adds
``specter2_embed`` (SPECTER2 + proximity adapter via sentence-transformers
+ adapters). Both slot into ``Profile.from_csv`` / ``Profile.from_pdfs``
via the same callable interface, selected by the ``embedder=`` keyword
with the chosen key recorded in ``profile.embedding_model``.

A Paper can carry multiple embeddings side-by-side (``Paper.embeddings``
is keyed by model name), so switching from placeholder to SPECTER2 does
not invalidate prior vectors — the Selector just reads a different key.

Registry. ``EMBEDDERS`` maps a model key to its callable; ``get_embedder
(name)`` is the selector-side lookup. At select time, Selectors resolve
``profile.embedding_model`` through this registry rather than carrying a
bound embedder, which keeps selector config JSON-safe (the key is a
string; the callable is rehydrated on the consumer side).

SPECTER2 imports are lazy and cached. Importing ``rag_lib.embedders`` at
module level does NOT pull in sentence-transformers or adapters; the
first call to ``specter2_embed`` loads the model. This keeps the 1A
compliance environment slim.
"""

from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np


Embedder = Callable[[str], list[float]]


# ----------------------------------------------------------------------
# Placeholder (1A)
# ----------------------------------------------------------------------


def placeholder_embed(text: str, *, dim: int = 128) -> list[float]:
    """Deterministic hash-seeded unit-norm vector.

    Not a real semantic embedding — same input yields the same vector,
    but semantically similar texts will produce unrelated vectors. Used
    for fast/offline tests and as the default for ``Profile.from_csv``.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big", signed=False)
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    v = v / (np.linalg.norm(v) + 1e-9)
    return v.tolist()


# ----------------------------------------------------------------------
# SPECTER2 (1B)
# ----------------------------------------------------------------------


SPECTER2_MODEL_ID = "allenai/specter2_base"
SPECTER2_ADAPTER = "allenai/specter2"


_specter2_model = None  # lazy-loaded; reused across calls


def _load_specter2():
    """Load SPECTER2 with the proximity adapter. Cached.

    The adapter is the correct configuration for nearest-neighbor
    retrieval between papers (cited pairs close, uncited pushed apart).
    Heavy import — deferred until first call. Requires the ``phase1b``
    extras: ``pip install -e '.[dev,phase1b]'``.
    """
    global _specter2_model
    if _specter2_model is not None:
        return _specter2_model
    try:
        from adapters import AutoAdapterModel  # type: ignore
        from transformers import AutoTokenizer  # type: ignore
    except ImportError as e:
        raise ImportError(
            "specter2_embed requires the phase1b extras. "
            "Install with: pip install -e '.[dev,phase1b]'"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(SPECTER2_MODEL_ID)
    model = AutoAdapterModel.from_pretrained(SPECTER2_MODEL_ID)
    model.load_adapter(SPECTER2_ADAPTER, source="hf", load_as="proximity",
                       set_active=True)
    model.eval()
    _specter2_model = (tokenizer, model)
    return _specter2_model


def specter2_embed(text: str) -> list[float]:
    """Encode text with SPECTER2 + proximity adapter. Returns a
    768-dim unit vector (the adapter head pools on [CLS])."""
    import torch  # type: ignore — lazy

    tokenizer, model = _load_specter2()
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512,
    )
    with torch.no_grad():
        out = model(**inputs)
    # SPECTER2's proximity adapter emits [CLS]-pooled embeddings.
    vec = out.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
    norm = np.linalg.norm(vec) + 1e-9
    return (vec / norm).tolist()


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


EMBEDDERS: dict[str, Embedder] = {
    "placeholder-v1": placeholder_embed,
    "specter2": specter2_embed,
}


def get_embedder(name: str) -> Embedder:
    """Resolve an embedder by its profile.embedding_model key. Selectors
    call this at select time with ``profile.embedding_model``."""
    if name not in EMBEDDERS:
        raise ValueError(
            f"Unknown embedder '{name}'. Known: {sorted(EMBEDDERS)}"
        )
    return EMBEDDERS[name]


def register_embedder(name: str, fn: Embedder) -> None:
    """Plug in a custom embedder — useful for tests and Phase 2+
    experiments. Overrides an existing key without warning."""
    EMBEDDERS[name] = fn
