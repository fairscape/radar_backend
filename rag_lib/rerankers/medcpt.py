"""MedCPTReranker — cross-encoder rescoring with configurable queries.

Uses ``ncbi/MedCPT-Cross-Encoder`` (a BERT cross-encoder trained on
PubMed search logs) to rescore candidates.  Two query modes are
supported, controlled by the ``query_mode`` parameter:

* ``"topic"`` — user-selected topic display names serve as queries;
  candidate ``title. abstract`` serves as articles.
* ``"article"`` — seed paper content (built via
  ``build_embedding_input``, same as the selector's SPECTER2 input)
  serves as queries; candidate content built the same way serves as
  articles.  This keeps information consistent between selector and
  reranker.

Per-query logits are aggregated (mean or max), min-max normalized to
[0, 1], then blended with the selector's cosine score to produce the
final ranking.

The model is loaded lazily on first use and cached for the process
lifetime (same pattern as ``rag_lib.embedders``).
"""

from __future__ import annotations

import math
import time
from threading import Lock

from ..embed import build_embedding_input
from ..paper import Paper
from ..profile import Profile
from ..scoring import attach_percentile


# ---------------------------------------------------------------------------
# Lazy model singleton
# ---------------------------------------------------------------------------

_model_cache: tuple | None = None
_model_lock = Lock()


def _load_model(model_id: str, device: str):
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    with _model_lock:
        if _model_cache is not None:
            return _model_cache
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch  # noqa: F811

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.eval()
        if device != "cpu":
            model = model.to(device)
        _model_cache = (tokenizer, model, device)
        return _model_cache


# ---------------------------------------------------------------------------
# MedCPTReranker
# ---------------------------------------------------------------------------


class MedCPTReranker:
    name = "medcpt"

    def __init__(
        self,
        *,
        model_id: str = "ncbi/MedCPT-Cross-Encoder",
        alpha: float = 0.4,
        beta: float = 0.6,
        device: str = "cpu",
        batch_size: int = 64,
        max_queries: int = 30,
        min_umls_confidence: float = 0.7,
        aggregation: str = "mean",
        query_mode: str = "topic",
    ):
        self.model_id = model_id
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.batch_size = batch_size
        self.max_queries = max_queries
        self.min_umls_confidence = min_umls_confidence
        self.aggregation = aggregation
        self.query_mode = query_mode
        self._last_cost: dict = {"wall_seconds": 0.0}
        self._last_diagnostics: dict = {"status": "idle"}

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    def rerank(
        self,
        ranked: list[tuple[float, Paper, dict]],
        profile: Profile,
        *,
        conn: "sqlite3.Connection | None" = None,
    ) -> list[tuple[float, Paper, dict]]:
        t0 = time.time()

        if not ranked:
            self._last_cost = {"wall_seconds": 0.0}
            return ranked

        queries = self._load_queries(profile)
        if not queries:
            self._last_diagnostics = {
                "status": "skipped",
                "reason": "no_queries",
            }
            self._last_cost = {"wall_seconds": time.time() - t0}
            return ranked

        # Build article texts
        articles: list[str] = []
        if self.query_mode == "article":
            for _, paper, _ in ranked:
                articles.append(build_embedding_input(paper))
        else:
            for _, paper, _ in ranked:
                text = paper.title or ""
                if paper.abstract:
                    text = f"{text}. {paper.abstract}" if text else paper.abstract
                articles.append(text)

        # Score all (query, article) pairs
        all_logits = self._batch_score(queries, articles)

        # Aggregate per-candidate raw scores first, then normalize
        n_q = len(queries)
        raw_scores: list[tuple[float, int]] = []  # (raw, best_idx)
        for i in range(len(ranked)):
            cand_logits = all_logits[i * n_q: (i + 1) * n_q]
            raw, best_idx = self._aggregate(cand_logits)
            raw_scores.append((raw, best_idx))

        # Min-max normalization across the batch for better spread
        raw_vals = [r[0] for r in raw_scores]
        rr_min = min(raw_vals)
        rr_max = max(raw_vals)
        rr_range = rr_max - rr_min if rr_max > rr_min else 1.0

        result: list[tuple[float, Paper, dict]] = []
        for i, (sel_score, paper, bd) in enumerate(ranked):
            raw, best_idx = raw_scores[i]
            norm = (raw - rr_min) / rr_range
            sel_raw = bd.get("score_raw", sel_score)
            sel_norm = (sel_raw + 1.0) / 2.0  # cosine [-1,1] -> [0,1]
            blended = self.alpha * sel_norm + self.beta * norm

            bd["score_pct_selector"] = bd.get("score_pct")
            bd["score_reranker_raw"] = raw
            bd["score_reranker_norm"] = norm
            bd["score_selector_norm"] = sel_norm
            bd["score_blended"] = blended
            bd["reranker_queries_used"] = n_q
            best_q = queries[best_idx]
            bd["reranker_best_query"] = (
                best_q[:120] + "\u2026" if len(best_q) > 120 else best_q
            )
            result.append((blended, paper, bd))

        result.sort(key=lambda r: r[0], reverse=True)
        attach_percentile(result)

        self._last_diagnostics = {
            "status": "ok",
            "n_candidates": len(ranked),
            "n_queries": n_q,
            "n_pairs": len(ranked) * n_q,
            "aggregation": self.aggregation,
            "query_mode": self.query_mode,
        }
        self._last_cost = {"wall_seconds": time.time() - t0}
        return result

    def config(self) -> dict:
        return {
            "type": "medcpt",
            "model_id": self.model_id,
            "alpha": self.alpha,
            "beta": self.beta,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_queries": self.max_queries,
            "min_umls_confidence": self.min_umls_confidence,
            "aggregation": self.aggregation,
            "query_mode": self.query_mode,
        }

    @classmethod
    def from_config(cls, config: dict) -> "MedCPTReranker":
        return cls(
            model_id=config.get("model_id", "ncbi/MedCPT-Cross-Encoder"),
            alpha=config.get("alpha", 0.4),
            beta=config.get("beta", 0.6),
            device=config.get("device", "cpu"),
            batch_size=config.get("batch_size", 64),
            max_queries=config.get("max_queries", 30),
            min_umls_confidence=config.get("min_umls_confidence", 0.7),
            aggregation=config.get("aggregation", "mean"),
            query_mode=config.get("query_mode", "topic"),
        )

    def cost(self) -> dict:
        return dict(self._last_cost)

    def diagnostics(self) -> dict:
        return dict(self._last_diagnostics)

    # ------------------------------------------------------------------
    # Query construction
    # ------------------------------------------------------------------

    def _load_queries(
        self,
        profile: Profile,
    ) -> list[str]:
        """Build query strings for the cross-encoder.

        When ``query_mode="topic"`` (default):
          1. User-selected topic display names
          2. Seed paper titles (fallback)

        When ``query_mode="article"``:
          1. Seed paper content via ``build_embedding_input``
             (title + abstract + body, same as selector's SPECTER2 input)
          2. Seed paper titles (fallback)
        """
        if self.query_mode == "article":
            queries = self._seed_content_queries(profile)
        else:
            queries = self._topic_queries(profile)
        if queries:
            return queries
        return self._title_queries(profile)

    def _topic_queries(self, profile: Profile) -> list[str]:
        """Display names of the topics the user has left switched on.

        The ``on`` check is load-bearing. Deselecting a topic used to
        delete it from ``topic_filters``, so this list was implicitly
        already filtered; it now marks ``on: False`` instead (so that
        re-aggregation can't resurrect pruned topics). Without the check
        we would rerank against topics the user explicitly switched off.
        """
        from ..openalex_tiers import is_enabled

        names = [
            t["display_name"]
            for t in profile.topic_filters.get("topics", [])
            if t.get("display_name") and is_enabled(t)
        ]
        return names[: self.max_queries] if names else []

    def _seed_content_queries(self, profile: Profile) -> list[str]:
        contents = []
        for paper in profile.papers:
            text = build_embedding_input(paper)
            if text.strip():
                contents.append(text)
        return contents[: self.max_queries] if contents else []

    def _title_queries(self, profile: Profile) -> list[str]:
        titles = [p.title for p in profile.papers if p.title]
        return titles[: self.max_queries] if titles else []

    # ------------------------------------------------------------------
    # Cross-encoder scoring
    # ------------------------------------------------------------------

    def _batch_score(
        self, queries: list[str], articles: list[str]
    ) -> list[float]:
        """Score all (query, article) pairs via MedCPT cross-encoder.

        Pairs are ordered candidate-major: for candidate i and query j
        the pair index is ``i * len(queries) + j``.
        """
        import torch

        tokenizer, model, device = _load_model(self.model_id, self.device)

        pairs: list[list[str]] = []
        for article in articles:
            for query in queries:
                pairs.append([query, article])

        all_logits: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start: start + self.batch_size]
            with torch.no_grad():
                encoded = tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=512,
                )
                if device != "cpu":
                    encoded = {k: v.to(device) for k, v in encoded.items()}
                logits = model(**encoded).logits.squeeze(dim=1)
                all_logits.extend(logits.cpu().tolist())

        return all_logits

    def _aggregate(self, logits: list[float]) -> tuple[float, int]:
        """Aggregate per-query logits for one candidate.

        Returns ``(aggregated_score, best_query_index)``.
        """
        if not logits:
            return 0.0, 0
        if self.aggregation == "mean":
            best_idx = max(range(len(logits)), key=lambda i: logits[i])
            return sum(logits) / len(logits), best_idx
        # default: max
        best_idx = max(range(len(logits)), key=lambda i: logits[i])
        return logits[best_idx], best_idx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)
