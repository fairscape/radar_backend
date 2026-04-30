"""Gather + embed + persist OpenAlex candidates for a built Profile.

Topic-filter strategy (three tiers, executed in order, stopping at the first
that yields >= --min-results):

  Tier 1 -- "core AND". Topics whose distinct-paper prevalence in the seed
            is >= --must-have-prevalence (default 0.5) are required: every
            candidate must contain ALL of them. AND across the same
            ``topics.id`` key is expressed by repeating it in the filter
            string (OpenAlex convention: comma between filters is AND, even
            within the same key).

  Tier 2 -- "top-N OR / subfields AND". The top-N most-frequent topics are
            OR'd (any one matches). The top-M most-frequent subfields are
            AND'd (each one required). Defaults: N=5, M=2. Picks up adjacent
            work that shares a core topic OR is in the same subfields as our
            seed.

  Tier 3 -- "subfields OR". Broadest fallback: just OR the top subfields.

Then each returned candidate is embedded with the profile's embedder and
ranked by cosine similarity to the seed centroid. The full ranked list (with
embeddings) is persisted to --out.

NO automatic retries -- on a 429 we surface immediately and stop. The user
can re-run after the rate-limit window clears.

Usage::

    python -m cli.gather --profile profiles/my_profile.json \\
        --email you@example.com --days 365 \\
        --out gathered/my_candidates.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import gather_runs as gather_runs_repo
from rag_lib.db.repos import profiles as profiles_repo
from rag_lib.embed import build_embedding_input
from rag_lib.embedders import get_embedder
from rag_lib.openalex_client import OpenAlexClient
from rag_lib.persistence import (
    dedup_and_insert_candidates,
    resolve_user_id,
    store_profile_from_object,
)
from rag_lib.profile import Profile
from rag_lib.radar import _days_ago_iso
from rag_lib.selectors.centroid import CentroidSelector
from rag_lib.selectors.max_seed import MaxSeedSelector


_SELECTORS = {"centroid": CentroidSelector, "max_seed": MaxSeedSelector}


# ---------------------------------------------------------------------------
# Topic prevalence + tier construction
# ---------------------------------------------------------------------------


def _bare_id(oa_id: str) -> str:
    """Strip the OpenAlex URL prefix; the filter API accepts both forms but
    bare IDs keep the URL short."""
    return oa_id.rsplit("/", 1)[-1] if "/" in oa_id else oa_id


def _distinct_paper_prevalence(profile: Profile) -> dict[str, float]:
    """For each topic-id, the fraction of seed papers it appears in
    (counting a paper once even if the topic shows up as both primary and
    in topics[]). Exact, unlike profile.topic_filters which uses an
    inflated tally."""
    n = len(profile.papers)
    if n == 0:
        return {}
    topic_to_papers: dict[str, set[int]] = defaultdict(set)
    for i, p in enumerate(profile.papers):
        seen: set[str] = set()
        if p.primary_topic and p.primary_topic.id:
            seen.add(p.primary_topic.id)
        for t in p.topics:
            if t and t.id:
                seen.add(t.id)
        for tid in seen:
            topic_to_papers[tid].add(i)
    return {tid: len(papers) / n for tid, papers in topic_to_papers.items()}


def _top_ids(profile_filters: dict, level: str, n: int) -> list[str]:
    items = profile_filters.get(level) or []
    return [_bare_id(it["id"]) for it in items[:n] if it.get("id")]


def _build_filter_string(parts: list[tuple[str, list[str], str]], since: str) -> str:
    """Build an OpenAlex filter string.

    parts: list of (oa_filter_key, ids, mode) where mode is "or" or "and".
      - "or"  -> joined with '|' (single key=val1|val2)
      - "and" -> emitted as repeated key=val1, key=val2 entries

    Standard non-topic filters (date, type, language) are added by the
    caller as additional ('key', [val], 'or') entries.
    """
    out = [f"from_publication_date:{since}", "type:article", "language:en"]
    for oa_key, ids, mode in parts:
        if not ids:
            continue
        if mode == "or":
            out.append(f"{oa_key}:{'|'.join(ids)}")
        else:  # and
            for v in ids:
                out.append(f"{oa_key}:{v}")
    return ",".join(out)


def _tier_specs(
    profile: Profile,
    *,
    must_have_prevalence: float,
    top_topics_n: int,
    top_subfields_n: int,
) -> list[tuple[str, list[tuple[str, list[str], str]]]]:
    """Return the three tiers as a list of (tier_name, parts) pairs, where
    parts is the input to _build_filter_string. Empty tiers are dropped."""
    prev = _distinct_paper_prevalence(profile)
    must_have = sorted(
        [tid for tid, p in prev.items() if p >= must_have_prevalence],
        key=lambda t: -prev[t],
    )
    must_have_bare = [_bare_id(t) for t in must_have]
    top_topics = _top_ids(profile.topic_filters, "topics", top_topics_n)
    top_subfields = _top_ids(profile.topic_filters, "subfields", top_subfields_n)

    tiers: list[tuple[str, list[tuple[str, list[str], str]]]] = []
    if must_have_bare:
        tiers.append(("must-have-AND", [("topics.id", must_have_bare, "and")]))
    if top_topics or top_subfields:
        tiers.append(("top-topics-OR-subfields-AND", [
            ("topics.id", top_topics, "or"),
            ("topics.subfield.id", top_subfields, "and"),
        ]))
    if top_subfields:
        tiers.append(("subfields-OR", [("topics.subfield.id", top_subfields, "or")]))
    return tiers


# ---------------------------------------------------------------------------
# Cursor-paginated walk -- one tier at a time, no retries
# ---------------------------------------------------------------------------


def _paged_search(
    client: OpenAlexClient,
    filter_str: str,
    *,
    limit: int,
    per_page: int,
) -> list[dict]:
    """Cursor-paginated /works query. Single attempt -- on any error,
    propagate. Stops when limit is reached or cursor is null."""
    out: list[dict] = []
    cursor: str | None = "*"
    while cursor:
        params = {"filter": filter_str, "per-page": per_page, "cursor": cursor}
        j = client._get("/works", params=params)
        results = j.get("results") or []
        remaining = limit - len(out)
        if len(results) > remaining:
            results = results[:remaining]
        out.extend(results)
        if len(out) >= limit:
            break
        cursor = (j.get("meta") or {}).get("next_cursor")
    return out


def _cascade_fetch(
    client: OpenAlexClient,
    profile: Profile,
    *,
    since: str,
    must_have_prevalence: float,
    top_topics_n: int,
    top_subfields_n: int,
    min_results: int,
    limit: int,
    per_page: int,
):
    tiers = _tier_specs(
        profile,
        must_have_prevalence=must_have_prevalence,
        top_topics_n=top_topics_n,
        top_subfields_n=top_subfields_n,
    )
    if not tiers:
        return [], None, "", []
    last = None
    for name, parts in tiers:
        filter_str = _build_filter_string(parts, since)
        print(f"\n-- tier: {name}\n   filter: {filter_str}")
        works = _paged_search(client, filter_str, limit=limit, per_page=per_page)
        papers = [client.paper_from_work(w, source="openalex_gatherer") for w in works]
        print(f"   fetched: {len(papers)}")
        last = (papers, name, filter_str, parts)
        if len(papers) >= min_results:
            break
    return last


# ---------------------------------------------------------------------------
# Percentile annotation
# ---------------------------------------------------------------------------


def _ranked_to_json(ranked):
    """Format a ranked list of (score, paper, breakdown) for JSON output.

    The selector already places ``score_pct`` in each breakdown (Phase 3);
    we surface it under both ``score`` and ``percentile`` for backward
    compatibility with consumers of the prior JSON shape, and pass
    through the diagnostic ``score_raw`` / ``score_max_seed`` fields.
    """
    out = []
    for entry in ranked:
        score = float(entry[0])
        paper = entry[1]
        breakdown = entry[2] if len(entry) >= 3 else {}
        out.append({
            "score": score,
            "percentile": float(breakdown.get("score_pct", score)),
            "score_raw": (
                float(breakdown["score_raw"])
                if breakdown.get("score_raw") is not None else None
            ),
            "score_max_seed": (
                float(breakdown["score_max_seed"])
                if breakdown.get("score_max_seed") is not None else None
            ),
            "paper": paper.to_dict(),
        })
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="gather")
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--email", type=str, required=True,
                    help="OpenAlex polite-pool mailto.")
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--limit", type=int, default=500,
                    help="Cap on candidates per tier.")
    ap.add_argument("--per-page", type=int, default=100,
                    help="OpenAlex per-page (max 200). Smaller = gentler on burst budget.")
    ap.add_argument("--rate-sleep", type=float, default=1.0,
                    help="Seconds to sleep between successful API calls.")
    ap.add_argument("--must-have-prevalence", type=float, default=0.5,
                    help="Tier 1: topics in >= this fraction of seed papers become AND-required.")
    ap.add_argument("--top-topics-n", type=int, default=5,
                    help="Tier 2: how many top topics to OR.")
    ap.add_argument("--top-subfields-n", type=int, default=2,
                    help="Tier 2: how many top subfields to AND. Tier 3: how many to OR.")
    ap.add_argument("--min-results", type=int, default=50,
                    help="Stop cascading when a tier returns >= this many candidates.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Drop candidates scoring below this. Default: keep all.")
    ap.add_argument("--selector", choices=sorted(_SELECTORS), default="max_seed",
                    help="Scoring rule. 'max_seed' = max cosine to any single seed "
                         "(default; preserves distinctive seeds). 'centroid' = cosine "
                         "to mean of seeds (legacy).")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--db", type=Path, default=None,
                    help="If set, persist candidates to this SQLite file with dedup against prior gather runs.")
    ap.add_argument("--user", type=str, default="demo@example.com",
                    help="Email identifying the owning user (auto-created on first use).")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()

    profile = Profile.from_json(args.profile)
    client = OpenAlexClient(mailto=args.email, rate_limit_sleep=args.rate_sleep)
    selector_cls = _SELECTORS[args.selector]
    selector = selector_cls(embedding_model=profile.embedding_model)
    selector.fit(profile)

    # When --db is set, resolve user + profile_id (creating both if they don't
    # exist yet) and open a gather_runs row up-front so any failure between
    # here and persistence still leaves an audit trail.
    db_state: dict | None = None
    if args.db is not None:
        conn = connect(args.db)
        apply_migrations(conn)
        user_id = resolve_user_id(conn, args.user)
        profile_row = profiles_repo.get_by_name(conn, user_id, profile.name)
        if profile_row is None:
            # First time we've seen this profile; persist it now so the
            # gather_runs FK has somewhere to point.
            profile_id = store_profile_from_object(
                conn, user_id=user_id, profile=profile,
            )
        else:
            profile_id = int(profile_row["id"])
        run_id = gather_runs_repo.start(
            conn,
            profile_id=profile_id,
            user_id=user_id,
            since_date=_days_ago_iso(args.days),
        )
        db_state = {"conn": conn, "profile_id": profile_id, "run_id": run_id}

    since = _days_ago_iso(args.days)
    try:
        result = _cascade_fetch(
            client, profile,
            since=since,
            must_have_prevalence=args.must_have_prevalence,
            top_topics_n=args.top_topics_n,
            top_subfields_n=args.top_subfields_n,
            min_results=args.min_results,
            limit=args.limit,
            per_page=args.per_page,
        )
    except Exception as exc:
        if db_state is not None:
            gather_runs_repo.finish(
                db_state["conn"], db_state["run_id"],
                n_fetched=0, n_new=0, n_redup=0,
                api_calls=client.api_calls,
                error=f"{type(exc).__name__}: {exc}",
            )
        raise

    if not result:
        if db_state is not None:
            gather_runs_repo.finish(
                db_state["conn"], db_state["run_id"],
                n_fetched=0, n_new=0, n_redup=0,
                api_calls=client.api_calls,
                error="no tiers; topic_filters empty",
            )
        print("No tiers had any IDs; profile.topic_filters is empty?")
        return 1
    candidates, tier_used, filter_str, _parts = result
    print(f"\n== chose tier: {tier_used} ({len(candidates)} candidates) ==")

    # Skip re-embedding when we already have a vector for this paper+model
    # in the DB. Without --db this is a no-op (DB skip falls back to the
    # in-memory absence check).
    embedder = get_embedder(profile.embedding_model)
    for p in candidates:
        if profile.embedding_model in p.embeddings:
            continue
        if db_state is not None and p.openalex_id:
            from rag_lib.db.repos import embeddings as embeddings_repo
            existing = embeddings_repo.get(
                db_state["conn"], p.openalex_id, profile.embedding_model,
            )
            if existing is not None:
                p.embeddings[profile.embedding_model] = existing.tolist()
                continue
        p.embeddings[profile.embedding_model] = embedder(build_embedding_input(p))

    ranked = selector.select(candidates, profile, threshold=args.threshold)

    out = {
        "source_profile": profile.name,
        "embedding_model": profile.embedding_model,
        "selector": selector.name,
        "since": since,
        "tier_used": tier_used,
        "filter": filter_str,
        "fetched": len(candidates),
        "kept": len(ranked),
        "ranked": _ranked_to_json(ranked),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {len(ranked)} ranked candidates -> {args.out}")

    if db_state is not None:
        n_new, n_redup = dedup_and_insert_candidates(
            db_state["conn"],
            profile_id=db_state["profile_id"],
            gather_run_id=db_state["run_id"],
            ranked=ranked,
            tier_used=tier_used,
        )
        gather_runs_repo.finish(
            db_state["conn"], db_state["run_id"],
            n_fetched=len(candidates),
            n_new=n_new,
            n_redup=n_redup,
            api_calls=client.api_calls,
            tier_used=tier_used,
        )
        print(f"persisted to {args.db}: n_new={n_new}, n_redup={n_redup}, run_id={db_state['run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
