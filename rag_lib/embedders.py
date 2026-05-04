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
import threading
from contextlib import contextmanager
from typing import Callable, Iterator

import numpy as np


Embedder = Callable[[str], list[float]]


# ----------------------------------------------------------------------
# Per-call progress hook
#
# Long gather runs are dominated by per-paper SPECTER2 inference inside
# the selector. The Selector Protocol is frozen (see selector.py), so
# instead of plumbing a callback through every selector signature, we
# expose a thread-local hook here that any embedder calls after
# producing a vector. Job code wraps ``selector.select(...)`` in
# ``embed_progress(callback)`` and gets a tick per embedded candidate
# without touching the selector layer.
# ----------------------------------------------------------------------


_progress_state = threading.local()


def _fire_embed_progress() -> None:
    cb = getattr(_progress_state, "callback", None)
    if cb is not None:
        try:
            cb()
        except Exception:  # noqa: BLE001 — progress is best-effort
            pass


@contextmanager
def embed_progress(callback: Callable[[], None] | None) -> Iterator[None]:
    """Install a per-thread callback fired once per real embedding call.

    ``callback`` is invoked from inside the embedder after each vector
    is produced. Restored to whatever was previously installed on exit
    so nested context managers compose. Pass ``None`` to disable
    progress reporting within the block.
    """
    prev = getattr(_progress_state, "callback", None)
    _progress_state.callback = callback
    try:
        yield
    finally:
        _progress_state.callback = prev


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
    result = (vec / norm).tolist()
    _fire_embed_progress()
    return result


# ----------------------------------------------------------------------
# Ollama-hosted embedders (chat retrieval)
#
# Used by the chat path's ``vault_chat`` collection. SPECTER2 above is
# kept as the selector/centroid embedder; these are query→passage
# embedders that perform much better for free-form questions over
# chunked text. They go through the same ollama service the chat LLM
# uses, so no extra dependencies — just an ``ollama pull <name>`` once.
# ----------------------------------------------------------------------


def _ollama_embed_factory(model_name: str) -> Embedder:
    """Build an embedder that calls ``POST {ollama_url}/api/embeddings``.

    The base URL is read from ``RADAR_OLLAMA_URL`` at call time (not at
    factory time) so test injection of settings still works. Raises the
    underlying httpx error on failure — callers in the indexer wrap
    this so a missing model / unreachable ollama doesn't break uploads.
    """
    import httpx  # local import — keeps module import cheap for tests
    import os

    # mxbai / nomic / bge-large all cap input at 512 BPE tokens.
    # Char-to-token ratio is ~4 for typical English prose but can be
    # ~2 for dense scientific PDFs (formulas, URLs, fused words from
    # poor extraction). Rather than guess, we try and halve on 500.
    # Worst-case 600 chars × 2 chars/token = 300 tokens — well under
    # the 512 cap, so the bottom rung should never legitimately 500.
    _SHRINK_LADDER = (1500, 1000, 600, 300)

    import structlog
    _emb_log = structlog.get_logger("rag_lib.embedders.ollama")

    def _embed(text: str) -> list[float]:
        url = os.environ.get("RADAR_OLLAMA_URL", "http://localhost:11434").rstrip("/")
        endpoint = f"{url}/api/embeddings"

        last_error: Exception | None = None
        for attempt, limit in enumerate(_SHRINK_LADDER, start=1):
            payload_text = text if len(text) <= limit else text[:limit]
            try:
                with httpx.Client(timeout=60.0) as client:
                    response = client.post(
                        endpoint,
                        json={"model": model_name, "prompt": payload_text},
                    )
                    response.raise_for_status()
                    data = response.json()
            except httpx.HTTPStatusError as exc:
                # 500 = ollama choked (most often: input too long).
                # Shrink and retry. Any other status (404, 503) means
                # the model isn't available or ollama is down — bail.
                if exc.response.status_code != 500:
                    raise
                _emb_log.warning(
                    "embed.shrink",
                    model=model_name,
                    attempt=attempt,
                    sent_chars=len(payload_text),
                    full_chars=len(text),
                    status=500,
                )
                last_error = exc
                continue
            vec = data.get("embedding") if isinstance(data, dict) else None
            if not isinstance(vec, list) or not vec:
                raise RuntimeError(
                    f"ollama {endpoint} returned no embedding for model={model_name!r}"
                )
            if attempt > 1:
                _emb_log.info(
                    "embed.recovered",
                    model=model_name,
                    attempt=attempt,
                    sent_chars=len(payload_text),
                )
            _fire_embed_progress()
            return [float(x) for x in vec]

        raise RuntimeError(
            f"ollama {endpoint} 500'd at all input sizes "
            f"({_SHRINK_LADDER}) for model={model_name!r}; last={last_error!r}"
        )

    return _embed


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------


EMBEDDERS: dict[str, Embedder] = {
    "placeholder-v1": placeholder_embed,
    "specter2": specter2_embed,
    # Ollama-hosted, query→passage retrievers. Pulled by the
    # ollama-init compose job; first call lazily resolves to ollama.
    "mxbai-embed-large": _ollama_embed_factory("mxbai-embed-large"),
    "nomic-embed-text": _ollama_embed_factory("nomic-embed-text"),
    "bge-large": _ollama_embed_factory("bge-large"),
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
