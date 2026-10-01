#!/usr/bin/env python3
"""Retroactively extract UMLS concepts and map topics for existing papers.

Runs the same extraction every other path runs (``vault._try_extract_umls``
over ``vault._umls_input_text``), so what it stores is what an upload or an
import would have stored. It used to rebuild both itself -- no title
prefix, the abstract only past 100 characters, the body uncapped, no
cache_dir -- so a paper whose abstract said only "T2DM" got no diabetes
concept here, and as the column was then non-NULL nothing ever redid it.

The running backend already does this on its own: the scheduler's
``umls-backfill`` job sweeps seeds every 15 minutes. This script is for a
one-off over papers the sweep does not cover (``--scope all``) or for
re-extracting (``--force``). It loads the UMLS linker itself, which takes
minutes and several GB of memory beside the backend's own copy.

Usage:
    python scripts/backfill_umls.py [--scope seeds|all] [--limit N] [--force]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env before importing settings
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from rag_lib.api.services.vault import _try_extract_umls, _umls_input_text  # noqa: E402
from rag_lib.api.settings import get_settings  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill UMLS data for existing papers")
    parser.add_argument("--scope", choices=("seeds", "all"), default="seeds",
                        help="seed papers only (default) or every paper")
    parser.add_argument("--limit", type=int, default=0, help="Max papers to process (0 = all)")
    parser.add_argument("--force", action="store_true", help="Re-extract even if already populated")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.RADAR_UMLS_ENABLED:
        print("RADAR_UMLS_ENABLED is false — aborting")
        return 1

    conn = sqlite3.connect(str(settings.RADAR_DB_PATH))
    conn.row_factory = sqlite3.Row

    where = []
    if not args.force:
        where.append("p.umls_concepts_json IS NULL")
    if args.scope == "seeds":
        where.append("EXISTS (SELECT 1 FROM profile_seeds s WHERE s.openalex_id = p.openalex_id)")
    query = "SELECT p.openalex_id, p.title, p.abstract, p.body_text FROM papers p"
    if where:
        query += " WHERE " + " AND ".join(where)
    if args.limit > 0:
        query += f" LIMIT {int(args.limit)}"

    rows = conn.execute(query).fetchall()
    print(f"Found {len(rows)} papers to process (scope={args.scope}, force={args.force})")
    n_failed = 0
    for i, row in enumerate(rows, 1):
        # The return value, not the column: with --force a failed re-extract
        # leaves the old value in place, and reading the column called that "ok".
        ok = _try_extract_umls(conn, settings, row["openalex_id"],
                               _umls_input_text(row["title"], row["abstract"], row["body_text"]),
                               force=args.force)
        if not ok:
            n_failed += 1
            print(f"[{i}/{len(rows)}] {row['openalex_id']}  FAILED (see log)")
            continue
        stored = conn.execute("SELECT umls_concepts_json FROM papers WHERE openalex_id = ?",
                              (row["openalex_id"],)).fetchone()[0]
        print(f"[{i}/{len(rows)}] {row['openalex_id']}  {'no concepts' if stored == '[]' else 'ok'}")
    print(f"done: {len(rows) - n_failed} stored, {n_failed} failed")
    conn.close()
    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
