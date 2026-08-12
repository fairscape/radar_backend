#!/usr/bin/env python3
"""Build the OpenAlex topic embedding index for UMLS topic mapping.

Fetches all ~4,500 OpenAlex topics via the public API, embeds each
topic's display_name with the configured embedder (default: specter2),
and saves the result to ``data/umls_cache/``.

Output files:
  - openalex_topics.json       — topic metadata [{id, display_name, subfield, field, domain}, ...]
  - openalex_topic_embeddings.npz — row-normalized float32 embedding matrix

Usage:
    python scripts/build_topic_index.py [--embedding-model specter2] [--output-dir data/umls_cache]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np

# Add parent to path so we can import rag_lib
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_lib.embedders import get_embedder


OPENALEX_TOPICS_URL = "https://api.openalex.org/topics"
PER_PAGE = 200
MAILTO = "radar-tool@example.com"


def fetch_all_topics(mailto: str) -> list[dict]:
    """Fetch all OpenAlex topics via cursor pagination."""
    topics: list[dict] = []
    cursor = "*"
    page = 0

    with httpx.Client(timeout=60.0) as client:
        while True:
            page += 1
            params = {
                "per_page": PER_PAGE,
                "cursor": cursor,
                "mailto": mailto,
            }
            resp = client.get(OPENALEX_TOPICS_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            results = data.get("results", [])
            if not results:
                break

            for r in results:
                topics.append({
                    "id": r.get("id", ""),
                    "display_name": r.get("display_name", ""),
                    "subfield": (r.get("subfield") or {}).get("display_name", ""),
                    "field": (r.get("field") or {}).get("display_name", ""),
                    "domain": (r.get("domain") or {}).get("display_name", ""),
                })

            print(f"  Page {page}: fetched {len(results)} topics (total: {len(topics)})")

            meta = data.get("meta", {})
            next_cursor = meta.get("next_cursor")
            if not next_cursor or next_cursor == cursor:
                break
            cursor = next_cursor

            # Polite delay
            time.sleep(0.1)

    return topics


def embed_topics(topics: list[dict], embedding_model: str) -> np.ndarray:
    """Embed each topic's display_name and return a (N, dim) matrix."""
    embedder = get_embedder(embedding_model)

    embeddings = []
    total = len(topics)
    for i, t in enumerate(topics):
        text = t["display_name"]
        if not text.strip():
            text = t.get("subfield") or t.get("field") or "unknown"
        vec = embedder(text)
        embeddings.append(vec)
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"  Embedded {i + 1}/{total} topics")

    return np.array(embeddings, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Build OpenAlex topic embedding index")
    parser.add_argument(
        "--embedding-model", default="specter2",
        help="Embedder key from rag_lib.embedders registry (default: specter2)",
    )
    parser.add_argument(
        "--output-dir", default="data/umls_cache",
        help="Output directory (default: data/umls_cache)",
    )
    parser.add_argument(
        "--mailto", default=MAILTO,
        help="Email for OpenAlex polite pool",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching OpenAlex topics...")
    topics = fetch_all_topics(args.mailto)
    print(f"Fetched {len(topics)} topics total")

    if not topics:
        print("No topics fetched — aborting")
        sys.exit(1)

    print(f"\nEmbedding topics with '{args.embedding_model}'...")
    embeddings = embed_topics(topics, args.embedding_model)
    print(f"Embedding matrix shape: {embeddings.shape}")

    # Save
    topics_path = output_dir / "openalex_topics.json"
    embeddings_path = output_dir / "openalex_topic_embeddings.npz"

    with open(topics_path, "w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False)
    np.savez_compressed(embeddings_path, embeddings=embeddings)

    print(f"\nSaved:")
    print(f"  {topics_path} ({topics_path.stat().st_size / 1024:.1f} KB)")
    print(f"  {embeddings_path} ({embeddings_path.stat().st_size / 1024:.1f} KB)")
    print("Done!")


if __name__ == "__main__":
    main()
