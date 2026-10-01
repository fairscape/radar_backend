#!/usr/bin/env python3
"""Fill authors and publication dates for papers stored without them (one-off).

Until Paper carried ``authors`` and ``publication_date``, every gathered
paper lost both at the OpenAlex -> Paper step: feed cards read "Unknown
authors" and showed <year>-01-01. Gathers store them now; this fetches them
for papers written before, 50 OpenAlex ids per request.

Only NULL columns are written, so a byline an import already stored is never
replaced. Synthetic ids (local:..., prosopia:...) name nothing in OpenAlex
and are skipped. There is deliberately no default database.

Usage:
    python scripts/backfill_authors.py --db /path/to/radar.db [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_lib.openalex_client import OpenAlexClient  # noqa: E402

BATCH = 50


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", required=True, type=Path, help="database to fill (no default)")
    ap.add_argument("--mailto", required=True, help="OpenAlex polite-pool address")
    ap.add_argument("--dry-run", action="store_true", help="fetch and report, write nothing")
    args = ap.parse_args()

    if not args.db.is_file():
        print(f"no such database: {args.db}")
        return 1
    print(f"DB path: {args.db.resolve()}  (dry_run={args.dry_run})")
    conn = sqlite3.connect(str(args.db))
    ids = [r[0] for r in conn.execute(
        "SELECT openalex_id FROM papers "
        "WHERE (authors_json IS NULL OR publication_date IS NULL) "
        "AND openalex_id LIKE 'https://openalex.org/W%'")]
    print(f"{len(ids)} papers missing authors or a date")
    client = OpenAlexClient(mailto=args.mailto)
    n_auth = n_date = 0
    for start in range(0, len(ids), BATCH):
        chunk = [i.rsplit("/", 1)[-1] for i in ids[start:start + BATCH]]
        works = client.paginate_filter("openalex_id:" + "|".join(chunk), limit=BATCH) or []
        for w in works:
            names = [(a.get("author") or {}).get("display_name") for a in (w.get("authorships") or [])]
            names = [n for n in names if n]
            date = w.get("publication_date")
            if args.dry_run:
                n_auth += bool(names)
                n_date += bool(date)
                continue
            cur = conn.execute(
                "UPDATE papers SET authors_json = ? WHERE openalex_id = ? AND authors_json IS NULL",
                (json.dumps(names) if names else None, w["id"]))
            n_auth += cur.rowcount if names else 0
            cur = conn.execute(
                "UPDATE papers SET publication_date = ? WHERE openalex_id = ? AND publication_date IS NULL",
                (date, w["id"]))
            n_date += cur.rowcount if date else 0
        conn.commit()
        print(f"  {min(start + BATCH, len(ids))}/{len(ids)}  ({len(works)} found in this batch)")
    print(f"{'would fill' if args.dry_run else 'filled'}: authors {n_auth}, dates {n_date}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
