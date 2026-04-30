"""End-to-end Phase 1B prototype CLI.

Feeds a corpus (CSV manifest or a directory of PDFs) through the full
pipeline:

  1. Build a Profile (Profile.from_csv or Profile.from_pdfs).
  2. Fit a CentroidSelector against it. Print coherence diagnostics.
  3. Print suggested OpenAlex topic IDs from the aggregated filters.
  4. Optional: dry-run over the last N days of OpenAlex output at a
     sweep of thresholds.
  5. Optional: live radar against the top-ranked candidates of the last
     N days, printing the top 20.

Everything hits real OpenAlex when ``--email`` is provided; the
``--embedder`` flag switches between the deterministic hash placeholder
(default, no extra deps) and SPECTER2 (requires ``phase1b`` extras).

Usage::

    python -m cli.ingest --csv manifest.csv --name my_profile \\
        --email you@example.com --out profiles/my_profile.json

    python -m cli.ingest --pdf-dir ./pdfs --name my_profile \\
        --email you@example.com --embedder specter2 --dry-run

    python -m cli.ingest --profile profiles/my_profile.json \\
        --email you@example.com --live
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_lib.db import apply_migrations, connect
from rag_lib.embedders import get_embedder
from rag_lib.gatherers.openalex import OpenAlexGatherer
from rag_lib.openalex_client import OpenAlexClient
from rag_lib.persistence import resolve_user_id, store_profile_from_object
from rag_lib.profile import Profile
from rag_lib.radar import dry_run
from rag_lib.selectors.centroid import CentroidSelector


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="ingest",
        description="End-to-end radar pipeline (build / fit / dry-run / live).",
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=Path, help="CSV manifest (doi,path,title,year,abstract).")
    src.add_argument("--pdf-dir", type=Path, help="Directory of PDFs.")
    src.add_argument("--profile", type=Path,
                     help="Pre-built Profile JSON. Skips build + fit steps unless --live.")
    ap.add_argument("--name", type=str, default=None,
                    help="Profile name. Required with --csv or --pdf-dir.")
    ap.add_argument("--email", type=str, required=True,
                    help="OpenAlex polite-pool mailto.")
    ap.add_argument("--embedder", type=str, default="placeholder-v1",
                    choices=("placeholder-v1", "specter2"),
                    help="Embedding model key.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Where to write the Profile JSON (default: <name>.json).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch last N days from OpenAlex and score at a threshold sweep.")
    ap.add_argument("--live", action="store_true",
                    help="Fetch last N days and print the top 20 ranked candidates.")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--thresholds", type=str, default="0.80,0.85,0.90",
                    help="Comma-separated thresholds for --dry-run.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap candidates fetched from OpenAlex.")
    ap.add_argument("--db", type=Path, default=None,
                    help="If set, persist Profile + seeds + embeddings to this SQLite file.")
    ap.add_argument("--user", type=str, default="demo@example.com",
                    help="Email identifying the owning user (auto-created on first use).")
    return ap.parse_args()


def _build(args, client) -> Profile:
    embedder = get_embedder(args.embedder)
    if args.csv:
        if not args.name:
            sys.exit("--name is required with --csv")
        return Profile.from_csv(
            args.csv, name=args.name, openalex_client=client,
            embedder=embedder, embedding_model=args.embedder,
        )
    if args.pdf_dir:
        if not args.name:
            sys.exit("--name is required with --pdf-dir")
        return Profile.from_pdfs(
            args.pdf_dir, name=args.name, openalex_client=client,
            embedder=embedder, embedding_model=args.embedder,
        )
    return Profile.from_json(args.profile)


def _print_diagnostics(selector: CentroidSelector, profile: Profile) -> None:
    diag = selector.diagnostics()
    print(f"\n== CentroidSelector fit: {profile.name} ==")
    for k in ("n_seed", "coherence_median", "coherence_iqr",
              "coherence_bimodal", "embedding_model"):
        if k in diag:
            v = diag[k]
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")
    cost = selector.cost()
    print(f"  fit wall_seconds: {cost['wall_seconds']:.3f}")


def _print_top_topics(profile: Profile) -> None:
    print(f"\n== Top OpenAlex topic_filters ==")
    for level in ("topics", "subfields", "fields", "domains"):
        items = profile.topic_filters.get(level, [])
        if not items:
            continue
        print(f"  {level}:")
        for t in items[:5]:
            print(f"    {t['id']}  {t['display_name']}  (n={t['count']})")


def _print_dry_run(report: dict) -> None:
    print(f"\n== Dry-run since {report['since']} ==")
    print(f"  fetched: {report['fetched']}")
    print(f"  gatherer cost: {report['gatherer_cost']}")
    print(f"  selector cost: {report['selector_cost']}")
    for thr, r in report["results"].items():
        print(f"  threshold {thr}: {r['count']} candidates")
        for title in r["top_titles"][:3]:
            print(f"    - {title}")


def _print_live(selector, profile, gatherer, days, limit) -> None:
    from rag_lib.radar import _days_ago_iso
    since = _days_ago_iso(days)
    candidates = gatherer.fetch(profile, since, limit=limit)
    ranked = selector.select(candidates, profile)
    print(f"\n== Live radar (top 20 of {len(ranked)}) ==")
    for entry in ranked[:20]:
        s, p = entry[0], entry[1]
        print(f"  {s:.3f}  {p.title}")


def main() -> int:
    args = _parse_args()
    client = OpenAlexClient(mailto=args.email)

    profile = _build(args, client)

    selector = CentroidSelector(embedding_model=args.embedder)
    selector.fit(profile)
    _print_diagnostics(selector, profile)
    _print_top_topics(profile)

    # Persist. Only write if we built from source; reloaded profiles are left alone.
    if args.csv or args.pdf_dir:
        out_path = args.out or Path(f"{profile.name}.json")
        # Commit the fitted centroid + threshold defaults back into the profile.
        profile.selector_config = selector.config()
        profile.gatherer_config = {"type": "openalex", "mailto": args.email}
        profile.to_json(out_path)
        print(f"\nWrote profile -> {out_path}")

    if args.db is not None:
        conn = connect(args.db)
        apply_migrations(conn)
        user_id = resolve_user_id(conn, args.user)
        # Make sure the on-disk Profile has the fitted selector_config when
        # we built from source; reloaded JSONs already carry it.
        if not profile.selector_config:
            profile.selector_config = selector.config()
        profile_id = store_profile_from_object(
            conn, user_id=user_id, profile=profile,
        )
        print(f"\nPersisted profile -> {args.db} (profile_id={profile_id}, user_id={user_id})")

    if args.dry_run:
        gatherer = OpenAlexGatherer(mailto=args.email, client=client)
        thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
        report = dry_run(
            selector, profile, gatherer,
            thresholds=thresholds, days=args.days, limit=args.limit,
        )
        _print_dry_run(report)

    if args.live:
        gatherer = OpenAlexGatherer(mailto=args.email, client=client)
        _print_live(selector, profile, gatherer, args.days, args.limit)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
