#!/usr/bin/env python3
"""Put every interest's candidate pool back on one scale (one-off).

Before rag_lib/scoring/pool.py, ``score_blended`` and ``score_pct`` were
per-gather-batch values ranked across all batches, so a small nightly
batch's best paper could outrank a big batch's better ones. The scheduler
now rescores the pool after every write; this applies the same rescoring to
pools written before that change.

There is deliberately no default database: ``--db`` must name it, and the
path is printed before anything is written.

Usage:
    python scripts/rescore_pools.py --db /path/to/radar.db [--dry-run]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_lib.scoring.pool import pool_scores, rescore_profile_pool  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", required=True, type=Path, help="database to rescore (no default)")
    ap.add_argument("--alpha", type=float, default=0.4, help="selector weight (RADAR_RERANKER_ALPHA)")
    ap.add_argument("--beta", type=float, default=0.6, help="reranker weight (RADAR_RERANKER_BETA)")
    ap.add_argument("--dry-run", action="store_true", help="report the top of each pool, write nothing")
    args = ap.parse_args()

    if not args.db.is_file():
        print(f"no such database: {args.db}")
        return 1
    print(f"DB path: {args.db.resolve()}  (alpha={args.alpha}, beta={args.beta}, dry_run={args.dry_run})")
    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    profiles = conn.execute("SELECT id, slug FROM profiles ORDER BY id").fetchall()
    for p in profiles:
        rows = [dict(r) for r in conn.execute(
            "SELECT openalex_id, score, score_raw, score_reranker_raw, score_blended "
            "FROM profile_candidates WHERE profile_id = ?", (p["id"],))]
        if not rows:
            continue
        if args.dry_run:
            new = pool_scores(rows, alpha=args.alpha, beta=args.beta)
            key = lambda r: new[r["openalex_id"]]["score_blended"] \
                if new[r["openalex_id"]]["score_blended"] is not None else new[r["openalex_id"]]["score_pct"]
            top = sorted(rows, key=key, reverse=True)[:3]
            print(f"  {p['slug']:34} {len(rows):4} rows  top cos: {[round(r['score_raw'] or 0, 3) for r in top]}")
        else:
            n = rescore_profile_pool(conn, int(p["id"]), alpha=args.alpha, beta=args.beta)
            print(f"  {p['slug']:34} {n:4} rows rescored")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
