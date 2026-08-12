"""Reranker protocol — the third pluggable stage, parallel to Selector.

A Reranker takes the Selector's ranked output and reorders it with a
more expensive, more accurate model. The split exists because the two
stages have opposite cost profiles: a bi-encoder Selector compares
pre-computed vectors and can score thousands of candidates cheaply but
never sees the two texts together, while a cross-encoder Reranker
encodes each (query, candidate) pair jointly — far better at fine
distinctions, far too slow to run over the raw candidate pool. Coarse
filter first, precise reorder second.

Position in the pipeline::

    Gatherer  -> candidate Papers
    Selector  -> [(cosine, Paper, breakdown), ...]   filtered by threshold
    Reranker  -> [(blended, Paper, breakdown), ...]  reordered, same length
    persistence -> profile_candidates

Invariant — reranking reorders, it does not filter.
    ``rerank()`` returns exactly the entries it was given. Dropping
    candidates is the Selector's job (via its threshold) and the read
    layer's job (via LIMIT). A Reranker that returned fewer entries
    would make ``n_fetched`` / ``n_new`` accounting silently wrong and
    would hide papers from the audit trail, so callers are entitled to
    assume ``len(out) == len(ranked)``.

Invariant — the breakdown dict is append-only.
    Rerankers add their own keys (``score_reranker_raw``,
    ``score_reranker_norm``, ``score_blended``, ...) to the breakdown the
    Selector produced. They must not remove or overwrite the Selector's
    keys: ``score_raw`` in particular is the only remaining record of the
    pre-rerank score once the blended value takes over the primary slot,
    and the comparison endpoint depends on it.

Scale caution for implementers.
    The first tuple element is what everything downstream sorts on, so
    whatever mixing an implementation does has to leave the two inputs
    genuinely comparable. Normalising one side batch-relative (min-max)
    and the other on an absolute scale silently hands the ranking to the
    batch-relative side, no matter what the nominal blend weights say.

Adding or removing methods on this Protocol is a breaking change for
the scheduler, the wizard dry-run, and the comparison API. Keep it in
step with ``selector.py`` and ``gatherer.py``.
"""

from __future__ import annotations

import sqlite3
from typing import Protocol, runtime_checkable

from .paper import Paper
from .profile import Profile

# (score, paper, breakdown) — the tuple every ranking stage passes on.
Ranked = list[tuple[float, Paper, dict]]


@runtime_checkable
class Reranker(Protocol):
    name: str  # "noop" | "medcpt" | ...

    def rerank(
        self,
        ranked: Ranked,
        profile: Profile,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> Ranked:
        """Reorder the Selector's output.

        ``ranked`` arrives sorted by the Selector's score, descending.
        The return value carries the same entries — see the reorder-only
        invariant above — re-sorted by whatever score the Reranker puts
        in the first tuple slot, descending.

        ``profile`` supplies the query side: topic display names, seed
        papers, or whatever else the implementation builds queries from.
        ``conn`` is an optional read-only handle for implementations that
        need extra per-paper state (e.g. stored UMLS concepts); it must
        not be written through.
        """

    def diagnostics(self) -> dict:
        """Per-run health metrics: at minimum ``status``, plus whatever
        explains the last call (candidates seen, queries used, pairs
        scored, aggregation mode)."""

    def config(self) -> dict:
        """Serializable config for ``profiles.reranker_config_json``.
        Must round-trip through ``from_config`` and include ``type`` so
        the registry can resolve the class."""

    @classmethod
    def from_config(cls, config: dict) -> "Reranker":
        """Reconstruct from persisted config."""

    def cost(self) -> dict:
        """``{'wall_seconds': float}`` for the last ``rerank()`` call.
        Cross-encoders dominate gather wall time once enabled, so this is
        the number that explains a slow run."""
