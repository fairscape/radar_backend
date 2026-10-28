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
import os
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
    """Fetch all OpenAlex topics, ``page=`` by ``page=`` (a ``cursor=``
    past the first page needs an API key since 2026-10)."""
    topics: list[dict] = []
    page = 0

    with httpx.Client(timeout=60.0) as client:
        while True:
            page += 1
            params = {
                "per_page": PER_PAGE,
                "page": page,
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

            count = (data.get("meta") or {}).get("count")
            if len(results) < PER_PAGE or (count is not None and page * PER_PAGE >= count):
                break

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
    from rag_lib.api.settings import get_settings

    # Must be the embedder the mapper uses at runtime, or every search
    # compares vectors from two different models (see rag_lib/umls/cache.py).
    default_model = get_settings().RADAR_UMLS_EMBEDDING_MODEL
    parser.add_argument(
        "--embedding-model", default=default_model,
        help=f"Embedder key from rag_lib.embedders registry (default: RADAR_UMLS_EMBEDDING_MODEL = {default_model})",
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

    from datetime import datetime, timezone

    from rag_lib.umls.cache import META_FILE

    meta_path = output_dir / META_FILE
    # Write beside the live files, then swap all three in: a running backend
    # never sees new topics with old vectors, or vectors without their model.
    tmp_topics = topics_path.with_suffix(".json.tmp")
    tmp_emb = output_dir / "openalex_topic_embeddings.tmp.npz"
    tmp_meta = meta_path.with_suffix(".json.tmp")
    with open(tmp_topics, "w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False)
    np.savez_compressed(tmp_emb, embeddings=embeddings)
    tmp_meta.write_text(json.dumps({
        "embedding_model": args.embedding_model,
        "dim": int(embeddings.shape[1]),
        "n_topics": len(topics),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    os.replace(tmp_topics, topics_path)
    os.replace(tmp_emb, embeddings_path)
    os.replace(tmp_meta, meta_path)

    print(f"\nSaved:")
    print(f"  {topics_path} ({topics_path.stat().st_size / 1024:.1f} KB)")
    print(f"  {embeddings_path} ({embeddings_path.stat().st_size / 1024:.1f} KB)")
    print(f"  {meta_path} (built with {args.embedding_model})")
    print("Done!")


if __name__ == "__main__":
    main()
