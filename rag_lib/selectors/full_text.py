"""FullTextSelector — corpus-aware selector that uses full PDF text.

Where CentroidSelector scores against one title+abstract embedding per
paper, FullTextSelector chunks each paper's body and embeds each chunk.
At fit time it builds a corpus-aware weighted centroid: chunks that
appear in many seeds (boilerplate methods, generic intros) are
down-weighted; distinctive chunks are up-weighted. At select time each
candidate is scored two ways and blended:

  score_pooled  = cosine(mean(candidate_chunks), weighted_centroid)
  score_maxsim  = mean over the top-k candidate chunks of
                  (max cosine to any seed chunk)
  score = alpha * score_pooled + (1 - alpha) * score_maxsim

PDF fetching is on by default — that's the point of this selector.
Most candidates from the gatherer arrive with ``Paper.pdf_url``
populated but ``Paper.body_text=""``, and a full-text selector that
silently fell back to title+abstract on those would be CentroidSelector
with extra steps. Pass ``fetch_missing_pdfs=False`` to opt out (e.g.,
when running against a corpus you've already pre-loaded). When fetch is
disabled or a candidate has no ``pdf_url``, that candidate falls back
to an abstract-only embedding with ``breakdown["fallback"]="abstract_only"``
so callers see the coverage gap.

Persisted state (``config()`` / ``from_config()``): every constructor
param plus the seed chunk matrix, per-chunk corpus weights, per-chunk
paper index, the precomputed weighted centroid, and the diagnostics
snapshot. Reconstructing does not require re-running fit.
"""

from __future__ import annotations

import time

import numpy as np
import structlog

from ..coherence import coherence
from ..embed import build_embedding_input
from ..embedders import get_embedder
from ..paper import Paper
from ..profile import Profile
from ..rag.indexer import chunk_text
from ..scoring import attach_percentile
from ..vault import fetch_pdf_text


log = structlog.get_logger("rag_lib.selectors.full_text")

PROGRESS_EVERY = 25  # emit a progress log every N candidates during select


class FullTextSelector:
    name = "full_text"

    def __init__(
        self,
        *,
        embedding_model: str | None = None,
        threshold: float | None = None,
        fetch_missing_pdfs: bool = True,
        chunk_target_tokens: int = 450,
        chunk_overlap: int = 50,
        top_k_maxsim: int = 3,
        alpha: float = 0.5,
        seed_chunk_matrix: list[list[float]] | None = None,
        seed_chunk_weights: list[float] | None = None,
        seed_chunk_paper_idx: list[int] | None = None,
        weighted_centroid: list[float] | None = None,
        diagnostics_snapshot: dict | None = None,
    ):
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.fetch_missing_pdfs = fetch_missing_pdfs
        self.chunk_target_tokens = chunk_target_tokens
        self.chunk_overlap = chunk_overlap
        self.top_k_maxsim = top_k_maxsim
        self.alpha = alpha
        self._seed_chunks: np.ndarray | None = (
            _normalize_rows(np.asarray(seed_chunk_matrix, dtype=float))
            if seed_chunk_matrix is not None else None
        )
        self._seed_weights: np.ndarray | None = (
            np.asarray(seed_chunk_weights, dtype=float)
            if seed_chunk_weights is not None else None
        )
        self._seed_paper_idx: np.ndarray | None = (
            np.asarray(seed_chunk_paper_idx, dtype=int)
            if seed_chunk_paper_idx is not None else None
        )
        self._weighted_centroid: np.ndarray | None = (
            np.asarray(weighted_centroid, dtype=float)
            if weighted_centroid is not None else None
        )
        self._diagnostics: dict = diagnostics_snapshot or {"status": "unfit"}
        self._last_cost: dict = {
            "wall_seconds": 0.0,
            "pdf_fetches": 0,
            "chunks_embedded": 0,
        }

    # ------------------------------------------------------------------

    def fit(self, profile: Profile) -> None:
        t0 = time.time()
        key = self.embedding_model or profile.embedding_model
        self.embedding_model = key
        embedder = get_embedder(key)
        log.info(
            "full_text.fit.start",
            n_seed_papers=len(profile.papers),
            embedding_model=key,
            chunk_target_tokens=self.chunk_target_tokens,
            chunk_overlap=self.chunk_overlap,
        )

        all_chunk_vecs: list[list[float]] = []
        paper_idx: list[int] = []
        n_without_body = 0
        for i, paper in enumerate(profile.papers):
            chunks = chunk_text(
                paper.body_text or "",
                target_tokens=self.chunk_target_tokens,
                overlap=self.chunk_overlap,
            )
            if chunks:
                log.debug(
                    "full_text.fit.seed",
                    i=i, openalex_id=paper.openalex_id,
                    n_chunks=len(chunks),
                )
                vecs = [embedder(c) for c in chunks]
            else:
                # Fallback: one abstract-only chunk so the seed isn't dropped.
                n_without_body += 1
                log.debug(
                    "full_text.fit.seed.no_body",
                    i=i, openalex_id=paper.openalex_id,
                )
                stored = paper.embedding_for(key)
                vecs = (
                    [list(stored)] if stored is not None
                    else [embedder(build_embedding_input(paper))]
                )
            for v in vecs:
                all_chunk_vecs.append(v)
                paper_idx.append(i)

        chunks_arr = _normalize_rows(np.asarray(all_chunk_vecs, dtype=float))
        idx_arr = np.asarray(paper_idx, dtype=int)
        weights = _corpus_aware_weights(chunks_arr, idx_arr)
        weighted_centroid = _weighted_centroid(chunks_arr, weights)

        self._seed_chunks = chunks_arr
        self._seed_weights = weights
        self._seed_paper_idx = idx_arr
        self._weighted_centroid = weighted_centroid

        self._diagnostics = {
            "status": "fit",
            "embedding_model": key,
            "n_seed_papers": int(len(profile.papers)),
            "n_seed_chunks": int(chunks_arr.shape[0]),
            "n_seed_papers_no_body": int(n_without_body),
            **_prefixed(coherence(chunks_arr), "coherence_"),
        }
        wall = time.time() - t0
        self._last_cost = {
            "wall_seconds": wall,
            "pdf_fetches": 0,
            "chunks_embedded": int(chunks_arr.shape[0]),
        }
        log.info(
            "full_text.fit.done",
            n_seed_papers=len(profile.papers),
            n_seed_chunks=int(chunks_arr.shape[0]),
            n_seed_papers_no_body=n_without_body,
            wall_seconds=round(wall, 3),
        )

    def select(
        self,
        candidates: list[Paper],
        profile: Profile,
        threshold: float | None = None,
    ) -> list[tuple[float, Paper, dict]]:
        if self._seed_chunks is None or self._weighted_centroid is None:
            raise RuntimeError("FullTextSelector.select() called before fit().")
        t0 = time.time()
        thr = threshold if threshold is not None else self.threshold
        key = self.embedding_model or profile.embedding_model
        embedder = get_embedder(key)

        seeds = self._seed_chunks
        wc = self._weighted_centroid
        wc_norm = wc / (np.linalg.norm(wc) + 1e-12)
        log.info(
            "full_text.select.start",
            n_candidates=len(candidates),
            fetch_missing_pdfs=self.fetch_missing_pdfs,
            alpha=self.alpha,
        )

        n_fetches = 0
        n_chunks_embedded = 0
        rows: list[
            tuple[float, float, float, int, bool, str | None, Paper]
        ] = []
        for idx, paper in enumerate(candidates):
            chunk_vecs, fallback, fetched = self._candidate_chunk_vecs(
                paper, embedder, key
            )
            if fetched:
                n_fetches += 1
            n_chunks_embedded += len(chunk_vecs)
            if (idx + 1) % PROGRESS_EVERY == 0:
                log.info(
                    "full_text.select.progress",
                    done=idx + 1, total=len(candidates),
                    pdf_fetches=n_fetches,
                    chunks_embedded=n_chunks_embedded,
                )

            cand_arr = _normalize_rows(np.asarray(chunk_vecs, dtype=float))
            pooled = cand_arr.mean(axis=0)
            pooled = pooled / (np.linalg.norm(pooled) + 1e-12)
            score_pooled = float(np.dot(pooled, wc_norm))

            sim = cand_arr @ seeds.T
            per_chunk_max = sim.max(axis=1)
            k = min(self.top_k_maxsim, per_chunk_max.shape[0])
            top_k = np.sort(per_chunk_max)[-k:]
            score_maxsim = float(top_k.mean())

            score = self.alpha * score_pooled + (1.0 - self.alpha) * score_maxsim
            rows.append(
                (score, score_pooled, score_maxsim, cand_arr.shape[0],
                 fetched, fallback, paper)
            )

        rows.sort(key=lambda r: r[0], reverse=True)
        ranked: list[tuple[float, Paper, dict]] = []
        for score, sp, sm, nc, fetched, fallback, paper in rows:
            breakdown = {
                "score_raw": score,
                "score_pooled": sp,
                "score_maxsim": sm,
                "n_chunks": nc,
                "fetched_pdf": bool(fetched),
                "fallback": fallback,
            }
            ranked.append((score, paper, breakdown))
        attach_percentile(ranked)

        if thr is not None:
            ranked = [r for r in ranked if r[0] >= thr]

        wall = time.time() - t0
        self._last_cost = {
            "wall_seconds": wall,
            "pdf_fetches": n_fetches,
            "chunks_embedded": n_chunks_embedded,
        }
        log.info(
            "full_text.select.done",
            n_results=len(ranked),
            pdf_fetches=n_fetches,
            chunks_embedded=n_chunks_embedded,
            wall_seconds=round(wall, 3),
        )
        return ranked

    # ------------------------------------------------------------------

    def _candidate_chunk_vecs(
        self, paper: Paper, embedder, key: str
    ) -> tuple[list[list[float]], str | None, bool]:
        """Resolve the chunk vectors for a single candidate.

        Returns ``(chunk_vecs, fallback_tag, fetched_pdf)``.
        ``fallback_tag`` is ``None`` when full body was used,
        ``"abstract_only"`` when we degraded to title+abstract.
        """
        if paper.body_text:
            chunks = chunk_text(
                paper.body_text,
                target_tokens=self.chunk_target_tokens,
                overlap=self.chunk_overlap,
            )
            if chunks:
                return [embedder(c) for c in chunks], None, False

        if self.fetch_missing_pdfs and paper.pdf_url:
            t_fetch = time.time()
            log.info(
                "full_text.select.fetch.start",
                openalex_id=paper.openalex_id, url=paper.pdf_url,
            )
            try:
                body = fetch_pdf_text(paper.pdf_url)
            except Exception as e:
                log.warning(
                    "full_text.select.fetch.failed",
                    openalex_id=paper.openalex_id, url=paper.pdf_url,
                    err=str(e),
                )
                body = ""
            else:
                log.info(
                    "full_text.select.fetch.done",
                    openalex_id=paper.openalex_id,
                    n_chars=len(body),
                    wall_seconds=round(time.time() - t_fetch, 3),
                )
            if body:
                chunks = chunk_text(
                    body,
                    target_tokens=self.chunk_target_tokens,
                    overlap=self.chunk_overlap,
                )
                if chunks:
                    return [embedder(c) for c in chunks], None, True

        stored = paper.embedding_for(key)
        if stored is not None:
            return [list(stored)], "abstract_only", False
        return (
            [embedder(build_embedding_input(paper))],
            "abstract_only",
            False,
        )

    # ------------------------------------------------------------------

    def diagnostics(self) -> dict:
        return dict(self._diagnostics)

    def config(self) -> dict:
        return {
            "type": "full_text",
            "embedding_model": self.embedding_model,
            "threshold": self.threshold,
            "fetch_missing_pdfs": self.fetch_missing_pdfs,
            "chunk_target_tokens": self.chunk_target_tokens,
            "chunk_overlap": self.chunk_overlap,
            "top_k_maxsim": self.top_k_maxsim,
            "alpha": self.alpha,
            "seed_chunk_matrix": (
                self._seed_chunks.tolist()
                if self._seed_chunks is not None else None
            ),
            "seed_chunk_weights": (
                self._seed_weights.tolist()
                if self._seed_weights is not None else None
            ),
            "seed_chunk_paper_idx": (
                self._seed_paper_idx.tolist()
                if self._seed_paper_idx is not None else None
            ),
            "weighted_centroid": (
                self._weighted_centroid.tolist()
                if self._weighted_centroid is not None else None
            ),
            "diagnostics_snapshot": dict(self._diagnostics),
        }

    @classmethod
    def from_config(cls, config: dict) -> "FullTextSelector":
        return cls(
            embedding_model=config.get("embedding_model"),
            threshold=config.get("threshold"),
            fetch_missing_pdfs=bool(config.get("fetch_missing_pdfs", True)),
            chunk_target_tokens=int(config.get("chunk_target_tokens", 450)),
            chunk_overlap=int(config.get("chunk_overlap", 50)),
            top_k_maxsim=int(config.get("top_k_maxsim", 3)),
            alpha=float(config.get("alpha", 0.5)),
            seed_chunk_matrix=config.get("seed_chunk_matrix"),
            seed_chunk_weights=config.get("seed_chunk_weights"),
            seed_chunk_paper_idx=config.get("seed_chunk_paper_idx"),
            weighted_centroid=config.get("weighted_centroid"),
            diagnostics_snapshot=config.get("diagnostics_snapshot"),
        )

    def cost(self) -> dict:
        return dict(self._last_cost)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


def _corpus_aware_weights(
    chunks: np.ndarray, paper_idx: np.ndarray
) -> np.ndarray:
    """Per-chunk weight = clip(1 - max-cos to any chunk in another paper).

    When all weights collapse near zero (single seed paper, or every
    chunk is a near-duplicate of one in another paper), fall back to
    uniform weights so the weighted centroid is well-defined.
    """
    n = chunks.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)
    sim = chunks @ chunks.T
    same_paper = paper_idx[:, None] == paper_idx[None, :]
    sim_masked = np.where(same_paper, -np.inf, sim)
    if not np.isfinite(sim_masked).any():
        return np.ones(n, dtype=float)
    max_other = sim_masked.max(axis=1)
    max_other = np.where(np.isfinite(max_other), max_other, 0.0)
    weights = np.clip(1.0 - max_other, 0.0, 1.0)
    if float(weights.sum()) < 1e-9:
        return np.ones(n, dtype=float)
    return weights


def _weighted_centroid(
    chunks: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    centroid = (weights[:, None] * chunks).sum(axis=0)
    norm = np.linalg.norm(centroid) + 1e-12
    return centroid / norm


def _prefixed(d: dict, prefix: str) -> dict:
    return {f"{prefix}{k}": v for k, v in d.items()}
