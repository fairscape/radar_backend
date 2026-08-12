#!/usr/bin/env python3
"""Retroactively extract UMLS concepts and map topics for existing papers.

Scans all papers with body_text that don't yet have umls_concepts_json
populated, extracts UMLS concepts, maps them to OpenAlex topics, and
updates the DB rows.

Usage:
    python scripts/backfill_umls.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env before importing settings
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import sqlite3
from rag_lib.api.settings import get_settings
from rag_lib.umls.extractor import extract_umls_concepts
from rag_lib.umls.topic_mapper import map_concepts_to_topics


def main():
    parser = argparse.ArgumentParser(description="Backfill UMLS data for existing papers")
    parser.add_argument("--limit", type=int, default=0, help="Max papers to process (0 = all)")
    parser.add_argument("--force", action="store_true", help="Re-extract even if already populated")
    args = parser.parse_args()

    settings = get_settings()
    if not settings.RADAR_UMLS_ENABLED:
        print("RADAR_UMLS_ENABLED is false — aborting")
        sys.exit(1)

    db_path = settings.RADAR_DB_PATH
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Find papers that need processing
    if args.force:
        query = "SELECT openalex_id, title, abstract, body_text FROM papers WHERE body_text IS NOT NULL AND length(body_text) > 0"
    else:
        query = "SELECT openalex_id, title, abstract, body_text FROM papers WHERE body_text IS NOT NULL AND length(body_text) > 0 AND umls_concepts_json IS NULL"

    if args.limit > 0:
        query += f" LIMIT {args.limit}"

    rows = conn.execute(query).fetchall()
    print(f"Found {len(rows)} papers to process")

    if not rows:
        print("Nothing to do")
        conn.close()
        return

    cache_dir = str(settings.RADAR_UMLS_CACHE_DIR)
    total_concepts = 0
    total_mapped = 0

    for i, row in enumerate(rows):
        oa_id = row["openalex_id"]
        title = row["title"] or ""
        text = row["body_text"] or row["abstract"] or ""

        if not text.strip():
            print(f"  [{i+1}/{len(rows)}] {title[:60]:60s} — no text, skipping")
            continue

        # Use abstract if available (cleaner), fall back to body_text
        extract_text = row["abstract"] if row["abstract"] and len(row["abstract"]) > 100 else text

        try:
            concepts = extract_umls_concepts(
                extract_text,
                min_confidence=settings.RADAR_UMLS_MIN_CONFIDENCE,
                spacy_model=settings.RADAR_UMLS_SPACY_MODEL,
                max_concepts=settings.RADAR_UMLS_MAX_CONCEPTS,
            )

            concepts_json = json.dumps([c.to_dict() for c in concepts]) if concepts else None

            mapped = []
            mapped_json = None
            if concepts:
                mapped = map_concepts_to_topics(
                    concepts,
                    min_similarity=settings.RADAR_UMLS_MIN_TOPIC_SIMILARITY,
                    embedding_model=settings.RADAR_UMLS_EMBEDDING_MODEL,
                    cache_dir=cache_dir,
                )
                mapped_json = json.dumps([m.to_dict() for m in mapped]) if mapped else None

            conn.execute(
                "UPDATE papers SET umls_concepts_json = ?, umls_mapped_topics_json = ? "
                "WHERE openalex_id = ?",
                (concepts_json, mapped_json, oa_id),
            )
            conn.commit()

            total_concepts += len(concepts)
            total_mapped += len(mapped)

            print(f"  [{i+1}/{len(rows)}] {title[:60]:60s} — {len(concepts)} concepts, {len(mapped)} mapped")

        except Exception as exc:
            print(f"  [{i+1}/{len(rows)}] {title[:60]:60s} — ERROR: {exc}")

    conn.close()
    print(f"\nDone! Processed {len(rows)} papers: {total_concepts} concepts, {total_mapped} topic mappings")


if __name__ == "__main__":
    main()
