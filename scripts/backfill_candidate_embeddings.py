#!/usr/bin/env python3
"""Store vectors for candidates gathered before they were kept, then re-score (one-off).

Until 2026-10-01 a gather computed every candidate's vector and threw it
away, so a refit (seeds added or removed) could not re-score the stored
candidates without embedding them all again. Gathers keep the vectors now;
this embeds the candidates stored before that, under each interest's own
embedding model, and then re-scores every live interest against its
current seeds (rag_lib/scoring/rescore.py).

There is deliberately no default database: ``--db`` must name it, and the
path is printed before anything is written. Loads the embedding model
itself (a GPU or several minutes of CPU, beside the backend's own copy).

Usage:
    python scripts/backfill_candidate_embeddings.py --db /path/to/radar.db [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The embedder's settings (device, model cache) come from the backend's .env.
# RADAR_DB_PATH in it is ignored: --db names the database.
from dotenv import load_dotenv  # noqa: E402
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from rag_lib.api.services.wizard import embed_missing  # noqa: E402
from rag_lib.db import connect  # noqa: E402
from rag_lib.scoring.rescore import rescore_candidates  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", required=True, type=Path, help="database to fill (no default)")
    ap.add_argument("--dry-run", action="store_true", help="count what is missing, write nothing")
    args = ap.parse_args()

    if not args.db.is_file():
        print(f"no such database: {args.db}")
        return 1
    print(f"DB path: {args.db.resolve()}  (dry_run={args.dry_run})")
    conn = connect(args.db)
    try:
        # Per embedding model: an interest scores with its own model's vectors.
        missing = conn.execute(
            """
            SELECT pr.embedding_model AS model, pc.openalex_id AS oa
            FROM profile_candidates pc
            JOIN profiles pr ON pr.id = pc.profile_id
            WHERE NOT EXISTS (
              SELECT 1 FROM paper_embeddings e
              WHERE e.openalex_id = pc.openalex_id AND e.embedding_model = pr.embedding_model
            )
            GROUP BY pr.embedding_model, pc.openalex_id
            """
        ).fetchall()
        by_model: dict[str, list[str]] = {}
        for r in missing:
            by_model.setdefault(r["model"], []).append(r["oa"])
        for model, ids in by_model.items():
            print(f"{model}: {len(ids)} candidates without a vector")
        if args.dry_run:
            return 0

        for model, ids in by_model.items():
            n = embed_missing(conn, model, ids)
            conn.commit()   # embeddings_repo.upsert leaves committing to the caller
            print(f"{model}: embedded {n}")

        live = conn.execute(
            "SELECT id, slug FROM profiles WHERE is_draft = 0 ORDER BY id").fetchall()
        for p in live:
            res = rescore_candidates(conn, int(p["id"]))
            print(f"  {p['slug']}: re-scored {res['n_rescored']}, still without a vector {res['n_skipped']}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
