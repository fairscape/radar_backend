"""Verify reranker output matches what the API returns.

Reads real candidates + selector scores from the DB, reruns MedCPT,
then compares the result against the stored score_blended values.

Usage:
    cd /bigtemp2/nkw3mr/revise_version/radar_backend
    .conda_env/bin/python scripts/verify_reranker.py
    .conda_env/bin/python scripts/verify_reranker.py --profile diabetes
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time

sys.path.insert(0, "/bigtemp2/nkw3mr/revise_version/radar_backend")

from rag_lib.paper import Paper
from rag_lib.profile import Profile
from rag_lib.rerankers.medcpt import MedCPTReranker
from rag_lib.scoring import attach_percentile
from rag_lib.db.repos import (
    papers as papers_repo,
    profiles as profiles_repo,
)

DB_PATH = "/bigtemp2/nkw3mr/radar_deployment/data/radar.db"


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_profile(conn, slug):
    row = conn.execute(
        "SELECT * FROM profiles WHERE slug = ?", (slug,)
    ).fetchone()
    if row is None:
        sys.exit(f"Profile '{slug}' not found")

    profile_id = int(row["id"])
    topic_filters = (
        json.loads(row["topic_filters_json"])
        if row["topic_filters_json"]
        else {}
    )

    seed_ids = profiles_repo.list_seed_openalex_ids(conn, profile_id)
    papers = []
    for oa_id in seed_ids:
        prow = papers_repo.get_by_openalex_id(conn, oa_id)
        if prow is None:
            continue
        topics = papers_repo.decode_topics(prow)
        paper = Paper.from_dict({
            "doi": prow["doi"],
            "openalex_id": prow["openalex_id"],
            "title": prow["title"],
            "abstract": prow["abstract"],
            "year": prow["year"],
            "venue": prow["venue"],
            "primary_topic": topics.get("primary_topic"),
            "topics": topics.get("topics") or [],
            "source": prow["source"],
        })
        papers.append(paper)

    profile = Profile(
        name=row["name"],
        papers=papers,
        topic_filters=topic_filters,
        threshold=row["threshold"],
    )
    return profile, profile_id


def load_db_candidates(conn, profile_id):
    """Load candidates that have score_blended from DB (what the API returns)."""
    rows = conn.execute(
        """
        SELECT pc.openalex_id, pc.score, pc.score_raw, pc.score_blended,
               pc.score_reranker_norm, pc.score_reranker_raw,
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
    return rows


def rebuild_ranked(rows):
    """Reconstruct the selector-output format from DB rows."""
    ranked = []
    for row in rows:
        topics_raw = row["topics_json"]
        topics = []
        if topics_raw:
            try:
                topics = json.loads(topics_raw)
            except json.JSONDecodeError:
                pass

        paper = Paper.from_dict({
            "openalex_id": row["openalex_id"],
            "title": row["title"],
            "abstract": row["abstract"],
            "year": row["year"],
            "venue": row["venue"],
            "doi": row["doi"],
            "topics": topics if isinstance(topics, list) else [],
            "source": row["source"] or "unknown",
        })

        score_raw = float(row["score_raw"]) if row["score_raw"] is not None else float(row["score"])
        breakdown = {
            "score_raw": score_raw,
        }
        ranked.append((score_raw, paper, breakdown))

    # Sort by selector score desc (original selector order)
    ranked.sort(key=lambda r: r[0], reverse=True)
    attach_percentile(ranked)
    return ranked


def main():
    parser = argparse.ArgumentParser(description="Verify reranker vs DB stored results")
    parser.add_argument("--profile", default="diabetes", help="Profile slug")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=1.0)
    args = parser.parse_args()

    conn = connect()
    profile, profile_id = load_profile(conn, args.profile)
    print(f"[1] Profile '{profile.name}': {len(profile.papers)} seeds")

    # Load what's in the DB
    db_rows = load_db_candidates(conn, profile_id)
    if not db_rows:
        sys.exit("No reranked candidates in DB")
    print(f"[2] Loaded {len(db_rows)} reranked candidates from DB")

    # Reconstruct selector output and rerun reranker
    ranked = rebuild_ranked(db_rows)
    print(f"[3] Selector scores: [{ranked[-1][0]:.4f}, {ranked[0][0]:.4f}]")

    reranker = MedCPTReranker(
        device=args.device,
        alpha=args.alpha,
        beta=args.beta,
        max_queries=30,
    )

    queries = reranker._load_queries(profile, conn)
    print(f"[4] UMLS queries ({len(queries)}):")
    for i, q in enumerate(queries):
        print(f"    {i+1:2d}. {q}")

    print(f"\n[5] Running MedCPT cross-encoder...")
    t0 = time.time()
    reranked = reranker.rerank(ranked, profile, conn=conn)
    print(f"    Done in {time.time()-t0:.1f}s")
    print(f"    Diagnostics: {reranker.diagnostics()}")

    # Build DB lookup: openalex_id -> stored blended score
    db_blended = {
        row["openalex_id"]: float(row["score_blended"])
        for row in db_rows
    }
    db_rank = {
        row["openalex_id"]: i + 1
        for i, row in enumerate(db_rows)
    }

    # Compare
    print(f"\n{'='*100}")
    print(f"{'#':>3}  {'Title':<45}  {'DB blended':>10}  {'Rerun blended':>13}  {'Diff':>8}  {'DB rank':>7}  {'Rerun rank':>10}")
    print("-" * 100)

    mismatches = 0
    rank_mismatches = 0
    for i, (score, paper, bd) in enumerate(reranked):
        oid = paper.openalex_id
        rerun_blended = bd.get("score_blended", score)
        stored_blended = db_blended.get(oid, -1)
        diff = abs(rerun_blended - stored_blended)
        stored_rank = db_rank.get(oid, -1)
        rerun_rank = i + 1

        flag = ""
        if diff > 0.001:
            flag = " ← SCORE DIFF"
            mismatches += 1
        if stored_rank != rerun_rank:
            if not flag:
                flag = " ← RANK DIFF"
            rank_mismatches += 1

        title = (paper.title or "")[:45]
        print(f"{rerun_rank:3d}  {title:<45}  {stored_blended:10.4f}  {rerun_blended:13.4f}  {diff:8.4f}  {stored_rank:7d}  {rerun_rank:10d}{flag}")

    print(f"\n{'='*100}")
    print(f"Score mismatches (diff > 0.001): {mismatches} / {len(reranked)}")
    print(f"Rank mismatches:                 {rank_mismatches} / {len(reranked)}")
    if mismatches == 0 and rank_mismatches == 0:
        print("PASS — reranker output matches DB exactly.")
    else:
        print("DIFF — see details above.")

    conn.close()


if __name__ == "__main__":
    main()
