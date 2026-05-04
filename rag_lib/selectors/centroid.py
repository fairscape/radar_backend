"""CentroidSelector — Phase 1B body, Phase 3 score-tuning.

fit() computes the mean embedding across the profile's seed papers AND
keeps the unit-normalized seed matrix so select() can also report
max-cosine-to-any-seed alongside the cosine-to-centroid signal.

select() returns ``(score_raw, paper, breakdown)`` triples sorted
descending by ``score_raw``. ``breakdown`` carries:

  score_raw          raw cosine to centroid, in [-1, 1] (no remapping).
  score_max_seed     max cosine to any single seed paper, in [-1, 1].
  score_median_seed  median cosine across all seeds, in [-1, 1].
  score_pct          per-batch rank percentile in [0, 1] (still computed
                     and persisted for the daily-feed UI; not the primary
                     tuple score because rank-percentile makes absolute
                     thresholds meaningless during calibration).

The primary score is the raw cosine so the wizard's calibration sweep
threshold semantics (``0.50``…``0.95``) match what the selector is
actually comparing.

Persisted state (``config()`` / ``from_config()``): centroid + seed
matrix as plain lists, embedding_model key, coherence diagnostics, and
threshold. Reconstructing does not require re-running fit, and
``score_max_seed`` survives a round-trip.
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


class CentroidSelector:
    name = "centroid"

    def __init__(
        self,
        *,
        embedding_model: str | None = None,
        threshold: float | None = None,
        centroid: list[float] | None = None,
        seed_matrix: list[list[float]] | None = None,
        diagnostics_snapshot: dict | None = None,
    ):
        self.embedding_model = embedding_model
        self.threshold = threshold
        self._centroid: np.ndarray | None = (
            np.asarray(centroid, dtype=float) if centroid is not None else None
        )
        self._seeds: np.ndarray | None = (
            _normalize_rows(np.asarray(seed_matrix, dtype=float))
            if seed_matrix is not None else None
        )
        self._diagnostics: dict = diagnostics_snapshot or {"status": "unfit"}
        self._last_cost: dict = {"wall_seconds": 0.0}

    # ------------------------------------------------------------------

    def fit(self, profile: Profile) -> None:
        """Find centroid based on text embeddings"""
        t0 = time.time()
        key = self.embedding_model or profile.embedding_model
        self.embedding_model = key
        vecs = profile.seed_embeddings(key)
        arr = np.asarray(vecs, dtype=float)
        self._centroid = arr.mean(axis=0)
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
        if self._centroid is None:
            raise RuntimeError("CentroidSelector.select() called before fit().")
        if self._seeds is None:
            key = self.embedding_model or profile.embedding_model
            self._seeds = _normalize_rows(
                np.asarray(profile.seed_embeddings(key), dtype=float)
            )
        t0 = time.time()
        thr = threshold if threshold is not None else self.threshold
        key = self.embedding_model or profile.embedding_model

        embedder = None
        # Stage 1: compute raw cosines without thresholding (need full batch
        # to compute percentile).
        rows: list[tuple[float, float, float, Paper]] = []
        for p in candidates:
            v = p.embedding_for(key)
            if v is None:
                if embedder is None:
                    embedder = get_embedder(key)
                v = embedder(build_embedding_input(p))
            vec = np.asarray(v, dtype=float)
            score_raw = _cosine(self._centroid, vec)
            n = np.linalg.norm(vec) + 1e-12
            seed_cos = self._seeds @ (vec / n)
            score_max_seed = float(seed_cos.max())
            score_median_seed = float(np.median(seed_cos))
            rows.append(
                (float(score_raw), score_max_seed, score_median_seed, p)
            )

        # Stage 2: sort by raw cosine to centroid (selector's primary signal),
        # attach percentile to the breakdown for persistence, then apply the
        # optional threshold against the raw cosine.
        rows.sort(key=lambda r: r[0], reverse=True)
        ranked: list[tuple[float, Paper, dict]] = []
        for raw, max_seed, median_seed, paper in rows:
            breakdown = {
                "score_raw": raw,
                "score_max_seed": max_seed,
                "score_median_seed": median_seed,
            }
            ranked.append((raw, paper, breakdown))
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
            "type": "centroid",
            "embedding_model": self.embedding_model,
            "threshold": self.threshold,
            "centroid": self._centroid.tolist() if self._centroid is not None else None,
            "seed_matrix": self._seeds.tolist() if self._seeds is not None else None,
            "diagnostics_snapshot": dict(self._diagnostics),
        }

    @classmethod
    def from_config(cls, config: dict) -> "CentroidSelector":
        return cls(
            embedding_model=config.get("embedding_model"),
            threshold=config.get("threshold"),
            centroid=config.get("centroid"),
            seed_matrix=config.get("seed_matrix"),
            diagnostics_snapshot=config.get("diagnostics_snapshot"),
        )

    def cost(self) -> dict:
        return self._last_cost


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


def _prefixed(d: dict, prefix: str) -> dict:
    return {f"{prefix}{k}": v for k, v in d.items()}
