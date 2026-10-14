#!/usr/bin/env python3
"""Rerank one interest's stored papers two ways and compare (read-only demo).

The question: do the research topics generated the Prosopia way
(rag_lib/interest_terms.py) make better MedCPT queries than what the
reranker uses today? This takes the interest's stored candidates that pass
its threshold, reranks them with the configured query mode and with
``terms``, and writes both orders side by side. Nothing is written to the
database. With ``--pairs`` it also writes a judgement set for the union of
both top-K lists, in the format probe_step6_judge.py reads, built with
probe_step6_pairs.py's own prompt builder so the labels are comparable
with the earlier evaluation.

There is deliberately no default database; run it on a copy.

Usage:
    python scripts/compare_rerank_queries.py --db COPY.db --profile SLUG [--k 20] [--out DIR] [--pairs]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(Path(__file__).resolve().parent.parent / ".env")   # reranker device / model settings

from rag_lib.api.settings import get_settings  # noqa: E402
from rag_lib.db import connect  # noqa: E402
from rag_lib.db.repos import candidates as candidates_repo  # noqa: E402
from rag_lib.paper import Paper  # noqa: E402
from rag_lib.scheduler import jobs  # noqa: E402


def _ranked(conn, profile_id: int) -> list[tuple]:
    rows = conn.execute(
        f"""
        SELECT pc.openalex_id, pc.score_raw, p.title, p.abstract, p.year, p.venue, p.doi
        FROM profile_candidates pc
        JOIN papers p USING (openalex_id)
        JOIN profiles pr ON pr.id = pc.profile_id
        WHERE pc.profile_id = ? AND {candidates_repo.PASSES}
        ORDER BY pc.score_raw DESC
        """,
        (profile_id,),
    ).fetchall()
    out = []
    for r in rows:
        paper = Paper(doi=r["doi"], openalex_id=r["openalex_id"], title=r["title"] or "",
                      abstract=r["abstract"] or "", year=r["year"], venue=r["venue"])
        out.append((float(r["score_raw"]), paper, {"score_raw": float(r["score_raw"])}))
    return out


def _rerank(reranker, ranked, profile, mode: str) -> tuple[list[str], list[str]]:
    reranker.query_mode = mode
    queries = reranker._load_queries(profile)
    fresh = [(s, p, dict(bd)) for s, p, bd in ranked]
    res = reranker.rerank(fresh, profile)
    res.sort(key=lambda r: r[2].get("score_reranker_raw", 0.0), reverse=True)
    return [p.openalex_id for _, p, _ in res], queries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", required=True, type=Path)
    ap.add_argument("--profile", required=True, help="interest slug")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("results/rerank_compare"))
    ap.add_argument("--pairs", action="store_true", help="also write a judgement set for probe_step6_judge.py")
    args = ap.parse_args()
    if not args.db.is_file():
        print(f"no such database: {args.db}")
        return 1
    print(f"DB path: {args.db.resolve()} (read-only use)")

    settings = get_settings()
    conn = connect(args.db)
    try:
        row = conn.execute("SELECT * FROM profiles WHERE slug = ? AND is_draft = 0", (args.profile,)).fetchone()
        if row is None:
            print(f"no live interest '{args.profile}'")
            return 1
        profile = jobs._load_profile(conn, row)
        if not profile.interest_terms:
            print("this interest has no research topics yet: generate them first")
            return 1
        ranked = _ranked(conn, int(row["id"]))
        reranker = jobs._build_reranker(row, settings)
        if reranker.name == "noop":
            print("the reranker is disabled in this configuration")
            return 1
        current_mode = reranker.query_mode
        base, base_q = _rerank(reranker, ranked, profile, current_mode)
        terms, terms_q = _rerank(reranker, ranked, profile, "terms")
    finally:
        conn.close()

    titles = {p.openalex_id: p.title for _, p, _ in ranked}
    pos_b = {oa: i + 1 for i, oa in enumerate(base)}
    pos_t = {oa: i + 1 for i, oa in enumerate(terms)}
    k = args.k
    overlap = len(set(base[:k]) & set(terms[:k]))
    print(f"{len(ranked)} papers pass the threshold; top-{k} overlap {overlap}/{k}")
    print(f"queries [{current_mode}]: {base_q}")
    print(f"queries [terms]: {terms_q}")

    out = args.out / args.profile
    out.mkdir(parents=True, exist_ok=True)
    with (out / "rankings.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["openalex_id", "title", f"rank_{current_mode}", "rank_terms", "delta"])
        for oa in sorted(set(base[:k]) | set(terms[:k]), key=lambda x: pos_t[x]):
            w.writerow([oa, titles.get(oa, ""), pos_b[oa], pos_t[oa], pos_b[oa] - pos_t[oa]])
    (out / "summary.json").write_text(json.dumps({
        "profile": args.profile, "n_candidates": len(ranked), "k": k, "overlap": overlap,
        "queries": {current_mode: base_q, "terms": terms_q},
        f"top_{current_mode}": base[:k], "top_terms": terms[:k],
    }, indent=2))
    print(f"wrote {out}/rankings.csv and summary.json")

    if args.pairs:
        root = Path(__file__).resolve().parents[2]       # the deployment dir holding the probes
        sys.path.insert(0, str(root))
        try:
            import probe_step6_pairs as pp
        except ImportError:
            print(f"--pairs needs probe_step6_pairs.py in {root}")
            return 1
        conn = connect(args.db)
        try:
            seeds = [{"title": p.title, "year": p.year, "venue": p.venue,
                      "_text": p.abstract or "", "_source": "openalex" if p.abstract else "title_only"}
                     for p in profile.papers]
            pairs = []
            by_id = {p.openalex_id: p for _, p, _ in ranked}
            for oa in sorted(set(base[:k]) | set(terms[:k])):
                p = by_id[oa]
                cand = {"title": p.title, "year": p.year, "venue": p.venue,
                        "_text": p.abstract or "", "_source": "openalex" if p.abstract else "title_only"}
                pid = f"{args.profile}|profile|{oa.rsplit('/', 1)[-1]}"
                pairs.append({
                    "pair_id": pid, "profile": args.profile, "mode": "profile",
                    "seed_ids": [s.openalex_id for s in profile.papers],
                    "candidate": {"openalex_id": oa, "title": p.title},
                    "join_keys": {f"rank_{current_mode}": pos_b[oa], "rank_terms": pos_t[oa],
                                  f"in_top{k}_{current_mode}": pos_b[oa] <= k, f"in_top{k}_terms": pos_t[oa] <= k},
                    "prompt_user": pp.build_user_prompt(seeds, cand, []),
                })
        finally:
            conn.close()
        doc = {"meta": {"prompt_version": pp.PROMPT_VERSION, "mode": "profile", "scope": f"compare_{current_mode}_vs_terms",
                        "top_k": k, "n_pairs": len(pairs), "rubric": pp.RUBRIC, "system_prompt": pp.SYSTEM_PROMPT},
               "profiles": {}, "pairs": pairs}
        (out / "pairs.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False))
        print(f"wrote {out}/pairs.json ({len(pairs)} pairs) for probe_step6_judge.py --pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
