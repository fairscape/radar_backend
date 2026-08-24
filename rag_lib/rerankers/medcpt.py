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

# (model_id, device) -> (tokenizer, model, device). Keyed, because a
# single slot ignoring the arguments hands the caller whatever was loaded
# first: the sweep scripts vary device between runs in one process and
# would have been told they were on cuda while running on the cpu model.
_model_cache: dict[tuple[str, str], tuple] = {}
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
    key = (model_id, device)
    cached = _model_cache.get(key)
    if cached is not None:
        return cached
    with _model_lock:
        cached = _model_cache.get(key)
        if cached is not None:
            return cached
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch  # noqa: F811

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.eval()
        if device != "cpu":
            model = model.to(device)
        _model_cache[key] = (tokenizer, model, device)
        return _model_cache[key]


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

        # [rerank-probe] 临时诊断 — 确认后整块删除（本文件共四处：三处输入 + 一处输出）
        if ranked:
            _raws = [r[2].get("score_raw", r[0]) for r in ranked]
            print(f"\n[rerank] 收到 {len(ranked)} 个候选（selector 已按 θ 过滤过）, "
                  f"score_raw {min(_raws):.4f} - {max(_raws):.4f}", flush=True)

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

        # [rerank-probe] 临时诊断 — 确认输入后整块删除
        print(f"[rerank] --- query 侧: {len(queries)} 条, mode={self.query_mode} ---",
              flush=True)
        for _i, _q in enumerate(queries):
            print(f"[rerank]   Q{_i}: {len(_q):6d} chars | {_q[:220]!r}", flush=True)
        print(f"[rerank] --- article 侧: {len(articles)} 条（前 3 条）---", flush=True)
        for _i, _a in enumerate(articles[:3]):
            print(f"[rerank]   A{_i}: {len(_a):6d} chars | {_a[:220]!r}", flush=True)
        _no_abs = sum(1 for _, _p, _ in ranked if not (_p.abstract or "").strip())
        print(f"[rerank]   其中 {_no_abs}/{len(ranked)} 篇候选没有 abstract"
              f"（这些只能靠标题打分）", flush=True)

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

        # [rerank-probe] 临时诊断 — 确认后整块删除。输出侧：两个分量各自的
        # 跨度、混合权重的实际效果、以及相对 selector 原序的位移。
        _before = {id(p): i + 1 for i, (_, p, _) in enumerate(ranked)}
        _after = sorted(result, key=lambda r: -r[0])
        _rr = [r[0] for r in raw_scores]
        _sn = [bd.get("score_selector_norm", 0.0) for _, _, bd in result]
        _rn = [bd.get("score_reranker_norm", 0.0) for _, _, bd in result]
        print("[rerank] " + "=" * 70, flush=True)
        print(f"[rerank] logit 原始范围 {min(_rr):.3f} - {max(_rr):.3f}"
              f"  (aggregation={self.aggregation}, 每候选 {n_q} 个 query)", flush=True)
        print(f"[rerank] 归一化前跨度: selector {sel_span:.4f} / reranker {rr_span:.4f}"
              f"   -> 归一化后两边都是 [0,1]", flush=True)
        print(f"[rerank] 权重 alpha={self.alpha} (selector) beta={self.beta} (reranker)"
              f"   -> 有效影响 = 权重 x 跨度 = "
              f"{self.alpha:.2f} vs {self.beta:.2f}", flush=True)
        print(f"[rerank] 归一化后实际分布: sel_norm "
              f"{min(_sn):.3f}-{max(_sn):.3f}  rr_norm {min(_rn):.3f}-{max(_rn):.3f}",
              flush=True)
        _moved = 0
        _shift = []
        for _i, (_s, _p, _bd) in enumerate(_after, 1):
            _b = _before.get(id(_p))
            if _b is not None and _b != _i:
                _moved += 1
                _shift.append(abs(_b - _i))
        if _shift:
            print(f"[rerank] 名次变动 {_moved}/{len(result)}, "
                  f"平均位移 {sum(_shift) / len(_shift):.1f}, 最大 {max(_shift)}",
                  flush=True)
        print(f"[rerank] 前 10 名 (rank  原名次  blended = "
              f"{self.alpha}*sel + {self.beta}*rr):", flush=True)
        for _i, (_s, _p, _bd) in enumerate(_after[:10], 1):
            print(f"[rerank]   {_i:3d}  <-#{_before.get(id(_p), 0):<4d} "
                  f"{_s:.4f} = {self.alpha}*{_bd.get('score_selector_norm', 0):.3f}"
                  f" + {self.beta}*{_bd.get('score_reranker_norm', 0):.3f}"
                  f"  (logit {_bd.get('score_reranker_raw', 0):+.2f}, "
                  f"cos {_bd.get('score_raw', 0):.4f})  {(_p.title or '')[:44]}",
                  flush=True)
        print(f"[rerank] selector 原前 5 名现在排第几:", flush=True)
        _now = {id(p): i + 1 for i, (_, p, _) in enumerate(_after)}
        for _, _p, _bd in ranked[:5]:
            print(f"[rerank]   #{_before[id(_p)]:<4d} -> #{_now.get(id(_p), 0):<4d} "
                  f"rr_norm={_bd.get('score_reranker_norm', 0):.3f}  "
                  f"{(_p.title or '')[:52]}", flush=True)
        print("[rerank] " + "=" * 70, flush=True)

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
                # [rerank-probe] 临时诊断 — 确认输入后整块删除。必须放在
                # 下面那行搬到 GPU 之前，encoded 这时还是 BatchEncoding。
                if start == 0:
                    _q, _a = batch[0]
                    _nq = len(tokenizer(_q)["input_ids"])
                    _na = len(tokenizer(_a)["input_ids"])
                    _kept = len(encoded["input_ids"][0])
                    print("[rerank] " + "=" * 70, flush=True)
                    print(f"[rerank] {len(queries)} queries x {len(articles)} articles "
                          f"= {len(pairs)} pairs, batch_size={self.batch_size}", flush=True)
                    print(f"[rerank] pair0 截断前: query {_nq} tok + article {_na} tok "
                          f"= {_nq + _na}；配对预算 512，实际保留 {_kept}", flush=True)
                    print(f"[rerank] pair0 模型实际看到的（[SEP] 分隔 query / article）:",
                          flush=True)
                    print(tokenizer.decode(encoded["input_ids"][0]), flush=True)
                    print("[rerank] " + "=" * 70, flush=True)
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


# ---------------------------------------------------------------------------
# Ad-hoc check: rerun the ranking stack over a profile's stored candidates
# ---------------------------------------------------------------------------
#
#     python -m rag_lib.rerankers.medcpt [profile-slug]
#
# Answers "would these candidates come out in the same order again". The
# centroid is refit from the seeds, every candidate is rescored against it,
# the threshold is applied, and the cross-encoder runs — the whole ranking
# stack, not just this class.
#
# Read from the database: the candidate list, the papers' text, and the
# stored SPECTER2 vectors. Vectors are an input to ranking, not a ranking
# decision; recomputing them would be measuring the embedder instead, and
# it costs an hour. Nothing that encodes an ordering — score_raw,
# score_blended, score_pct, the stored centroid — is fed into the rerun.
# Those are only ever compared against.
#
# Two things are expected NOT to reproduce bit-for-bit, and the tolerances
# say so: the cross-encoder runs on the GPU, where float accumulation order
# is not guaranteed, and the blend is a min-max over the batch, so it is
# only meaningful when the rerun batch equals the batch that was persisted.
# Ranks are the honest check; scores are reported to show the size of any
# drift.
#
# Imports live inside the function: this package sits below the scheduler
# and the API, and nothing about reranking should depend on either.

_SCORE_TOL = 1e-4


def _main() -> int:
    import json
    import sys
    import time

    import numpy as np

    from ..api.settings import get_settings
    from ..db import connect, decode_vector
    from ..db.repos import embeddings as embeddings_repo, papers as papers_repo
    from ..paper import Paper
    from ..scheduler.jobs import _build_reranker, _load_profile
    from ..selectors import get_selector

    slug = sys.argv[1] if len(sys.argv) > 1 else "type-1-diabetes"
    settings = get_settings()
    conn = connect(settings.RADAR_DB_PATH)

    prow = conn.execute(
        "SELECT * FROM profiles WHERE slug = ?", (slug,),
    ).fetchone()
    if prow is None:
        print(f"no profile {slug!r}")
        return 1
    profile_id = int(prow["id"])

    stored_rows = conn.execute(
        """
        SELECT pc.openalex_id, pc.score, pc.score_raw, pc.score_pct,
               pc.score_blended, pc.score_reranker_norm,
               p.title, p.abstract, p.year, p.venue, p.doi,
               p.topics_json, p.source
          FROM profile_candidates pc
          JOIN papers p USING (openalex_id)
         WHERE pc.profile_id = ?
           AND pc.score_blended IS NOT NULL
         ORDER BY pc.score_blended DESC
        """,
        (profile_id,),
    ).fetchall()
    if not stored_rows:
        print(f"no reranked candidates for {slug!r} — run a gather first")
        return 1

    profile = _load_profile(conn, prow)
    model = profile.embedding_model

    print(f"profile      : {slug}  ({len(profile.papers)} seeds, "
          f"{len(stored_rows)} stored candidates)")
    print(f"embed model  : {model}")
    print(f"threshold    : {profile.threshold}")

    # --- selector: refit the centroid rather than loading the stored one ---
    sel_cfg = json.loads(prow["selector_config_json"] or "{}")
    sel_type = sel_cfg.get("type") or "centroid"
    selector = get_selector(sel_type)(embedding_model=model)
    selector.fit(profile)
    print(f"selector     : {sel_type} (refit from {len(profile.papers)} seeds)")

    stored_centroid = (
        decode_vector(prow["centroid"]) if prow["centroid"] else None
    )
    fresh_centroid = getattr(selector, "_centroid", None)
    if stored_centroid is not None and fresh_centroid is not None:
        a = np.asarray(fresh_centroid, dtype=float)
        b = np.asarray(stored_centroid, dtype=float)
        cos = float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))
        print(f"centroid     : refit vs stored cosine = {cos:.6f}"
              f"{'  MATCH' if cos > 1 - 1e-6 else '  DIFFERS'}")

    # --- rebuild the candidate Papers, carrying their stored vectors ---
    candidates: list[Paper] = []
    missing_vec = 0
    for row in stored_rows:
        topics = papers_repo.decode_topics(row)
        paper = Paper.from_dict({
            "openalex_id": row["openalex_id"],
            "doi": row["doi"],
            "title": row["title"],
            "abstract": row["abstract"],
            "year": row["year"],
            "venue": row["venue"],
            "primary_topic": topics.get("primary_topic"),
            "topics": topics.get("topics") or [],
            "source": row["source"] or "unknown",
        })
        vec = embeddings_repo.get(conn, row["openalex_id"], model)
        if vec is None:
            missing_vec += 1
        else:
            paper.embeddings[model] = vec.tolist()
        candidates.append(paper)
    if missing_vec:
        print(f"  warning: {missing_vec} candidates have no stored vector "
              f"and will be re-embedded")

    # --- rerun the two stages ---
    t0 = time.monotonic()
    ranked = selector.select(candidates, profile, threshold=profile.threshold)
    sel_secs = time.monotonic() - t0
    print(f"\nselect       : {len(ranked)}/{len(candidates)} passed threshold"
          f"   ({sel_secs:.1f}s)")
    if len(ranked) != len(candidates):
        print("  note: everything stored had already passed, so a drop here "
              "means the refit centroid moved")
    if not ranked:
        conn.close()
        return 1

    reranker = _build_reranker(prow, settings)
    print(f"reranker     : {reranker.name}"
          f"  alpha={getattr(reranker, 'alpha', '?')}"
          f"  beta={getattr(reranker, 'beta', '?')}"
          f"  query_mode={getattr(reranker, 'query_mode', '?')}")
    if reranker.name == "noop":
        print("  RADAR_RERANKER_ENABLED is off — nothing to verify")
        conn.close()
        return 1

    t0 = time.monotonic()
    ranked = reranker.rerank(ranked, profile, conn=conn)
    rr_secs = time.monotonic() - t0
    print(f"rerank       : {len(ranked)} candidates   ({rr_secs:.1f}s)")
    print(f"  diagnostics: {reranker.diagnostics()}")

    # --- compare ---
    # Two different "before"s, and conflating them makes the output
    # meaningless. ``stored_rank`` is the blended order already in the
    # database — the thing the rerun has to reproduce, so when it does,
    # it equals the new rank for every row and carries no information.
    # ``selector_rank`` is the order this run produced *before* the
    # cross-encoder touched it, which is what shows the reranker working.
    stored_rank = {r["openalex_id"]: i + 1 for i, r in enumerate(stored_rows)}
    stored_by_id = {r["openalex_id"]: r for r in stored_rows}
    selector_rank = {
        p.openalex_id: i + 1
        for i, (_, p, _) in enumerate(
            sorted(ranked, key=lambda e: e[2].get("score_raw", e[0]), reverse=True)
        )
    }

    same_rank = 0
    raw_drift: list[float] = []
    blend_drift: list[float] = []
    moved: list[tuple[int, int, float, float, str]] = []

    for i, (score, paper, bd) in enumerate(ranked):
        oid = paper.openalex_id
        srow = stored_by_id.get(oid)
        if srow is None:
            continue
        rank_now, rank_was = i + 1, stored_rank[oid]
        if rank_now == rank_was:
            same_rank += 1
        else:
            moved.append((rank_was, rank_now, float(srow["score_blended"]),
                          float(score), paper.title or ""))
        if srow["score_raw"] is not None:
            raw_drift.append(abs(bd.get("score_raw", score) - srow["score_raw"]))
        blend_drift.append(abs(score - float(srow["score_blended"])))

    n = len(ranked)
    print("\n" + "=" * 78)
    print(f"{same_rank}/{n} candidates landed on their stored rank")
    if raw_drift:
        print(f"selector score drift : max {max(raw_drift):.2e}  "
              f"mean {sum(raw_drift)/len(raw_drift):.2e}"
              f"   ({sum(d > _SCORE_TOL for d in raw_drift)} over {_SCORE_TOL:g})")
    if blend_drift:
        print(f"blended score drift  : max {max(blend_drift):.2e}  "
              f"mean {sum(blend_drift)/len(blend_drift):.2e}"
              f"   ({sum(d > _SCORE_TOL for d in blend_drift)} over {_SCORE_TOL:g})")

    if moved:
        print(f"\n{len(moved)} moved (worst 15 by distance):")
        moved.sort(key=lambda m: abs(m[0] - m[1]), reverse=True)
        for was, now, sb, nb, title in moved[:15]:
            print(f"  {was:>4} -> {now:<4}  {sb:.4f} -> {nb:.4f}   {title[:44]}")

    # sel# is this run's own selector order; rank is after the blend.
    # A row where the two differ is the reranker overriding the selector.
    print("\ntop 10 after reranking       (sel# = where the selector had it)")
    print(f"  {'rank':>4} {'sel#':>5}  {'blended':>7}  {'rr':>6}  {'sel':>6}  title")
    for i, (score, paper, bd) in enumerate(ranked[:10]):
        srank = selector_rank.get(paper.openalex_id, 0)
        move = srank - (i + 1)
        arrow = f"{move:+d}" if move else "  ="
        print(f"  {i+1:>4} {srank:>5}  {score:>7.4f}  "
              f"{bd.get('score_reranker_norm', 0):>6.4f}  "
              f"{bd.get('score_raw', 0):>6.4f}  {arrow:>4}  "
              f"{(paper.title or '')[:38]}")

    # And what the selector would have shown on its own, for contrast.
    by_sel = sorted(ranked, key=lambda e: e[2].get("score_raw", e[0]), reverse=True)
    print("\ntop 10 the selector alone would have given:")
    for i, (score, paper, bd) in enumerate(by_sel[:10]):
        now = next(
            (j + 1 for j, (_, q, _) in enumerate(ranked)
             if q.openalex_id == paper.openalex_id), 0,
        )
        print(f"  {i+1:>4} -> {now:<4}  sel={bd.get('score_raw', 0):.4f}  "
              f"rr={bd.get('score_reranker_norm', 0):.4f}  "
              f"{(paper.title or '')[:38]}")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
