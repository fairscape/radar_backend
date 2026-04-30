"""MaxSeedSelector — score by max cosine to any single seed.

Same interface as CentroidSelector, but instead of averaging the seed
embeddings into one vector and comparing each candidate to the mean, we
keep the full seed matrix and score each candidate as

    s = max_i cos(candidate, seed_i)

Rationale: when a profile contains a few methodologically distinctive
seeds alongside broader review-style seeds, the centroid mean dilutes
the distinctive ones. A candidate that's a near-twin of one specific
seed should score high even if it doesn't resemble the corpus
"average". Max-of-seeds preserves that signal.

Phase 3 dropped the legacy ``(c + 1) / 2`` mapping. The selector emits
``(score_raw, paper, breakdown)`` triples — the primary score is the
raw max cosine in [-1, 1] so wizard calibration thresholds match what
the selector compares. ``score_pct`` (per-batch rank percentile) is
still attached to the breakdown for the daily-feed UI.

Diagnostics include the same coherence summary as CentroidSelector
(it's a property of the seed matrix, not of the scoring rule).
"""

from __future__ import annotations

import time

import numpy as np

from ..coherence import coherence
from ..embed import build_embedding_input
from ..embedders import get_embedder
from ..paper import Paper
from ..profile import Profile
from ..scoring import attach_percentile


class MaxSeedSelector:
    name = "max_seed"

    def __init__(
        self,
        *,
        embedding_model: str | None = None,
        threshold: float | None = None,
        seed_matrix: list[list[float]] | None = None,
        diagnostics_snapshot: dict | None = None,
    ):
        self.embedding_model = embedding_model
        self.threshold = threshold
        self._seeds: np.ndarray | None = (
            _normalize_rows(np.asarray(seed_matrix, dtype=float))
            if seed_matrix is not None else None
        )
        self._diagnostics: dict = diagnostics_snapshot or {"status": "unfit"}
        self._last_cost: dict = {"wall_seconds": 0.0}

    # ------------------------------------------------------------------

    def fit(self, profile: Profile) -> None:
        t0 = time.time()
        key = self.embedding_model or profile.embedding_model
        self.embedding_model = key
        vecs = profile.seed_embeddings(key)
        arr = np.asarray(vecs, dtype=float)
        self._seeds = _normalize_rows(arr)
        self._diagnostics = {
            "status": "fit",
            "embedding_model": key,
            "n_seed": int(arr.shape[0]),
            **_prefixed(coherence(arr), "coherence_"),
        }
        self._last_cost = {"wall_seconds": time.time() - t0}

    def select(
        self,
        candidates: list[Paper],
        profile: Profile,
        threshold: float | None = None,
    ) -> list[tuple[float, Paper, dict]]:
        if self._seeds is None:
            raise RuntimeError("MaxSeedSelector.select() called before fit().")
        t0 = time.time()
        thr = threshold if threshold is not None else self.threshold
        key = self.embedding_model or profile.embedding_model

        embedder = None
        rows: list[tuple[float, Paper]] = []
        for p in candidates:
            v = p.embedding_for(key)
            if v is None:
                if embedder is None:
                    embedder = get_embedder(key)
                v = embedder(build_embedding_input(p))
            vec = np.asarray(v, dtype=float)
            n = np.linalg.norm(vec) + 1e-12
            score_raw = float((self._seeds @ (vec / n)).max())
            rows.append((score_raw, p))

        rows.sort(key=lambda r: r[0], reverse=True)
        ranked: list[tuple[float, Paper, dict]] = [
            (raw, p, {"score_raw": raw, "score_max_seed": raw})
            for raw, p in rows
        ]
        attach_percentile(ranked)

        if thr is not None:
            ranked = [r for r in ranked if r[0] >= thr]

        self._last_cost = {"wall_seconds": time.time() - t0}
        return ranked

    # ------------------------------------------------------------------

    def diagnostics(self) -> dict:
        return dict(self._diagnostics)

    def config(self) -> dict:
        return {
            "type": "max_seed",
            "embedding_model": self.embedding_model,
            "threshold": self.threshold,
            "seed_matrix": self._seeds.tolist() if self._seeds is not None else None,
            "diagnostics_snapshot": dict(self._diagnostics),
        }

    @classmethod
    def from_config(cls, config: dict) -> "MaxSeedSelector":
        return cls(
            embedding_model=config.get("embedding_model"),
            threshold=config.get("threshold"),
            seed_matrix=config.get("seed_matrix"),
            diagnostics_snapshot=config.get("diagnostics_snapshot"),
        )

    def cost(self) -> dict:
        return self._last_cost


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


def _prefixed(d: dict, prefix: str) -> dict:
    return {f"{prefix}{k}": v for k, v in d.items()}
