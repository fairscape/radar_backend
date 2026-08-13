"""MedCPTReranker — cross-encoder rescoring with configurable queries.

Uses ``ncbi/MedCPT-Cross-Encoder`` (a BERT cross-encoder trained on
PubMed search logs) to rescore candidates.  Two query modes are
supported, controlled by the ``query_mode`` parameter:

* ``"topic"`` — user-selected topic display names serve as queries;
  candidate ``title. abstract`` serves as articles. Short phrases are
  the shape this model was trained on, and it is the default.
* ``"title"`` — seed paper titles serve as queries. Short like a topic
  name, but a title names disease, measurement and method together
  where topic names split them across separate queries that then get
  averaged. Under evaluation, not yet the default.
* ``"article"`` — seed paper content (built via
  ``build_embedding_input``, same as the selector's SPECTER2 input)
  serves as queries; candidate content built the same way serves as
  articles.  Intended to keep information consistent between selector
  and reranker, but it puts a ~300-word document where the model
  expects a search string. Measured on a type-1 diabetes profile the
  two stages came out *uncorrelated* (rank correlation -0.04, against
  +0.62 for topic mode on the same candidates), and the top of the
  ranking filled with off-domain papers that happen to be long and
  technical. Kept for comparison; not recommended.

Per-query logits are aggregated (mean or max) and min-max normalized to
[0, 1], then blended with the selector's score — which is min-max
normalized over the same batch, so that ``alpha`` and ``beta`` weight
two comparable quantities.

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


def _batch_span(values: list[float]) -> tuple[float, float]:
    """Offset and span for a batch min-max: ``(x - lo) / span`` -> [0, 1].

    A batch where every value is identical has no spread to normalise;
    returning 1.0 leaves the whole batch at 0.0 rather than dividing by
    zero, which is the right answer — that stage has expressed no
    preference and should not move the ranking.
    """
    lo, hi = min(values), max(values)
    span = hi - lo
    return lo, span if span > 0 else 1.0


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

        # Build article texts. Only "article" mode gives the candidate
        # the full treatment; "topic" and "title" pair a short query with
        # the candidate's title and abstract, which is the shape MedCPT
        # was trained on.
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

        # Both stages are min-max normalised over the same batch, and it
        # matters that they are normalised the *same way*.
        #
        # The selector previously used a fixed (cos + 1) / 2. Because the
        # reranker's min-max fills [0, 1] by construction while real
        # SPECTER2 cosines within one batch sit in a band about a
        # twentieth as wide, the two terms entered the sum with spreads
        # differing by more than an order of magnitude — and what moves a
        # ranking is weight times spread, not weight. Measured on a
        # 60-candidate batch: selector spread 0.057 x alpha 0.4 = 0.023
        # against reranker spread 1.0 x beta 0.6 = 0.600, an effective
        # 4:96 split from a nominal 40:60, with the blend correlating
        # 0.997 with the reranker alone and 0.014 with the selector.
        #
        # Normalising both the same way makes alpha and beta mean what
        # they say, and — since the selector's spread varies from batch
        # to batch — makes the split stable rather than a property of
        # whatever was fetched that morning.
        rr_lo, rr_span = _batch_span([r[0] for r in raw_scores])
        sel_lo, sel_span = _batch_span(
            [bd.get("score_raw", s) for s, _, bd in ranked]
        )

        result: list[tuple[float, Paper, dict]] = []
        for i, (sel_score, paper, bd) in enumerate(ranked):
            raw, best_idx = raw_scores[i]
            norm = (raw - rr_lo) / rr_span
            sel_raw = bd.get("score_raw", sel_score)
            sel_norm = (sel_raw - sel_lo) / sel_span
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

        When ``query_mode="title"``:
          Seed paper titles, and nothing else.

        Titles are worth having as a mode of their own rather than only
        as the fallback. A seed titled "Basal Glucose Control in Type 1
        Diabetes Using Deep Reinforcement Learning" states the disease,
        the measurement and the method in one string, so a candidate is
        scored against the whole of what the user asked for. Topic names
        cannot express that: they arrive as separate queries — "Diabetes
        Management and Research", "Artificial Intelligence in Healthcare"
        — and averaging over them scores a paper that satisfies one the
        same as a paper that satisfies both. A full abstract can express
        it but is several times longer than anything on the query side of
        this model's training data, and it crowds the candidate out of
        the 512-token budget the pair shares.

        Not the default: on the one profile measured so far it moved a
        single paper, which is not evidence of much. Selectable so that
        it can be evaluated properly.
        """
        if self.query_mode == "article":
            queries = self._seed_content_queries(profile)
        elif self.query_mode == "title":
            return self._title_queries(profile)
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
