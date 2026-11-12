#!/usr/bin/env python3
"""Build the OpenAlex topic embedding index for UMLS topic mapping.

Fetches all ~4,500 OpenAlex topics via the public API, embeds each
topic's display_name with the configured embedder (default: specter2),
and saves the result to ``data/umls_cache/``.

Output files:
  - openalex_topics.json       — topic metadata [{id, display_name, subfield, field, domain}, ...]
  - openalex_topic_embeddings.npz — row-normalized float32 embedding matrix

With ``--rich`` it builds the index the UMLS mapper searches instead: each
topic embedded as its name, OpenAlex's description and its keywords
(``rag_lib.umls.cache.rich_topic_text``), fetched with the configured
OpenAlex key (``RADAR_OPENALEX_API_KEY``), into openalex_topics_rich.json /
openalex_topic_rich_embeddings.npz / openalex_topic_rich_index.meta.json.
``--topics-json PATH`` embeds topics fetched earlier (that same JSON format)
without calling OpenAlex.

Usage:
    python scripts/build_topic_index.py [--embedding-model specter2] [--output-dir data/umls_cache]
    python scripts/build_topic_index.py --rich [--topics-json topics.json] [--output-dir data/umls_cache]
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


def fetch_rich_topics() -> list[dict]:
    """Every OpenAlex topic with its description and keywords."""
    from rag_lib.openalex_client import OpenAlexClient

    client = OpenAlexClient(MAILTO)
    topics: list[dict] = []
    page = 0
    while True:
        page += 1
        data = client.topics_page(page, PER_PAGE)
        results = data.get("results") or []
        for r in results:
            topics.append({
                "id": r.get("id", ""),
                "display_name": r.get("display_name", ""),
                "description": r.get("description") or "",
                "keywords": r.get("keywords") or [],
                "subfield": (r.get("subfield") or {}).get("display_name", ""),
                "field": (r.get("field") or {}).get("display_name", ""),
                "domain": (r.get("domain") or {}).get("display_name", ""),
            })
        print(f"  Page {page}: fetched {len(results)} topics (total: {len(topics)})")
        count = (data.get("meta") or {}).get("count")
        if len(results) < PER_PAGE or (count is not None and page * PER_PAGE >= count):
            break
        time.sleep(0.1)
    print(f"  OpenAlex calls: {client.api_calls}")
    return topics


def topic_text(topic: dict) -> str:
    """The name-only index's text: the display_name (or its subfield)."""
    text = topic["display_name"]
    if not text.strip():
        text = topic.get("subfield") or topic.get("field") or "unknown"
    return text


def embed_topics(topics: list[dict], embedding_model: str, text=topic_text) -> np.ndarray:
    """Embed each topic's ``text(topic)`` and return a (N, dim) matrix."""
    embedder = get_embedder(embedding_model)

    embeddings = []
    total = len(topics)
    for i, t in enumerate(topics):
        vec = embedder(text(t))
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
    parser.add_argument(
        "--rich", action="store_true",
        help="Build the rich index (name + description + keywords) the UMLS mapper searches",
    )
    parser.add_argument(
        "--topics-json", type=Path, default=None,
        help="With --rich: embed topics from this JSON (as --rich saves them) instead of fetching",
    )
    args = parser.parse_args()
    if args.topics_json and not args.rich:
        parser.error("--topics-json needs --rich")

    from rag_lib.umls import cache

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.topics_json:
        topics = json.loads(args.topics_json.read_text(encoding="utf-8"))
        print(f"Read {len(topics)} topics from {args.topics_json}")
    elif args.rich:
        print("Fetching OpenAlex topics with descriptions and keywords...")
        topics = fetch_rich_topics()
    else:
        print(f"Fetching OpenAlex topics...")
        topics = fetch_all_topics(args.mailto)
    print(f"Fetched {len(topics)} topics total")

    if not topics:
        print("No topics fetched — aborting")
        sys.exit(1)

    print(f"\nEmbedding topics with '{args.embedding_model}'...")
    embeddings = embed_topics(topics, args.embedding_model,
                              cache.rich_topic_text if args.rich else topic_text)
    print(f"Embedding matrix shape: {embeddings.shape}")

    # Save
    if args.rich:
        names = (cache.RICH_TOPICS_FILE, cache.RICH_EMBEDDINGS_FILE, cache.RICH_META_FILE)
    else:
        names = (cache.TOPICS_FILE, cache.EMBEDDINGS_FILE, cache.META_FILE)
    topics_path, embeddings_path, meta_path = (output_dir / n for n in names)

    from datetime import datetime, timezone

    # Write beside the live files, then swap all three in: a running backend
    # never sees new topics with old vectors, or vectors without their model.
    tmp_topics = topics_path.with_suffix(".json.tmp")
    tmp_emb = embeddings_path.with_name(embeddings_path.stem + ".tmp.npz")
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
