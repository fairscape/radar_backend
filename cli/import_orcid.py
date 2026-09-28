"""Create a radar draft from a researcher's ORCID, without the UI.

Runs the same service the ``POST /api/profiles/draft/from-orcid`` route
dispatches to the scheduler, synchronously and with progress on stderr:

    python -m cli.import_orcid 0000-0001-5643-4068 --user you@example.com
    python -m cli.import_orcid 0000-0001-5643-4068 --user you@example.com \\
        --commit --threshold 0.85          # also aggregate topics + commit

Reads settings the way the API does (``.env`` in the working directory,
``RADAR_*`` environment variables); ``--db`` overrides the database path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_lib.api.services import orcid_import, wizard as wizard_service
from rag_lib.api.settings import get_settings
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import gather_runs as gather_runs_repo
from rag_lib.db.repos import profiles as profiles_repo
from rag_lib.db.repos import users as users_repo
from rag_lib.openalex_client import OpenAlexClient
from rag_lib.openalex_tiers import enabled_topic_ids


class _StderrReporter:
    """Minimal stand-in for the scheduler's _ProgressReporter."""

    def step(self, name: str, *, total=None, message=None) -> None:
        print(f"[{name}] {message or ''}{f' (total {total})' if total else ''}", file=sys.stderr)

    def make_embed_tick(self, total=None):
        done = {"n": 0}

        def _tick() -> None:
            done["n"] += 1
            if total and (done["n"] % 5 == 0 or done["n"] == total):
                print(f"[embedding] {done['n']}/{total}", file=sys.stderr)

        return _tick


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="import_orcid", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("orcid", help="ORCID (bare or https://orcid.org/... form)")
    ap.add_argument("--user", required=True, help="RADAR user email the draft belongs to")
    ap.add_argument("--name", default=None, help="radar name (default: OpenAlex display name)")
    ap.add_argument("--mailto", default=None, help="OpenAlex polite-pool email (default: --user)")
    ap.add_argument("--db", type=Path, default=None, help="SQLite path (default: settings)")
    ap.add_argument("--max-seeds", type=int, default=None, help="override RADAR_ORCID_MAX_SEEDS")
    ap.add_argument("--top-k", type=int, default=None, help="override RADAR_ORCID_TOPIC_TOP_K")
    ap.add_argument("--seeds", default="default",
                    help="which fetched works become seeds: default (lead-author, newest N), "
                         "lead (every lead-author work), all (every seed-eligible work), "
                         "or ids:<file> (one OpenAlex id per line)")
    ap.add_argument("--commit", action="store_true", help="aggregate topics and commit the draft")
    ap.add_argument("--threshold", type=float, default=0.85, help="selector threshold for --commit")
    return ap.parse_args()


def _selection(spec: str):
    """``--seeds`` → what run_import expects."""
    if spec.startswith("ids:"):
        return [ln.strip() for ln in Path(spec[4:]).read_text().splitlines() if ln.strip()]
    if spec not in ("default", "lead", "all"):
        raise SystemExit(f"--seeds must be default, lead, all or ids:<file>; got {spec!r}")
    return spec


def main() -> int:
    args = _parse_args()
    settings = get_settings()
    overrides = {}
    if args.db is not None:
        overrides["RADAR_DB_PATH"] = args.db
    if args.max_seeds is not None:
        overrides["RADAR_ORCID_MAX_SEEDS"] = args.max_seeds
    if args.top_k is not None:
        overrides["RADAR_ORCID_TOPIC_TOP_K"] = args.top_k
    if overrides:
        settings = settings.model_copy(update=overrides)

    try:
        orcid = orcid_import.normalize_orcid(args.orcid)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    conn = connect(settings.RADAR_DB_PATH)
    try:
        apply_migrations(conn)
        user = users_repo.upsert(conn, args.user)
        user_id = int(user["id"])
        client = OpenAlexClient(mailto=args.mailto or args.user)

        author = orcid_import.resolve_author(client, orcid)
        if author is None:
            print(f"error: ORCID {orcid} not found on OpenAlex", file=sys.stderr)
            return 1
        name = (args.name or author.display_name).strip()
        print(f"author  : {author.display_name} | {author.institution} | works={author.works_count}",
              file=sys.stderr)

        draft = wizard_service.create_draft(
            conn, user_id=user_id, name=name,
            embedding_model=settings.RADAR_DEFAULT_EMBEDDING_MODEL,
            selector=settings.RADAR_DEFAULT_SELECTOR,
        )
        slug = draft["slug"]
        row = profiles_repo.get_by_slug(conn, user_id, slug)
        profile_id = int(row["id"])
        profiles_repo.set_orcid(conn, profile_id, orcid=orcid, researcher_name=author.display_name)
        run_id = gather_runs_repo.start(conn, profile_id=profile_id, user_id=user_id,
                                        tier_used="orcid_import")
        try:
            result = orcid_import.run_import(
                conn, settings, user_id=user_id, profile_id=profile_id, slug=slug,
                orcid=orcid, name=name, client=client, reporter=_StderrReporter(), author=author,
                selection=_selection(args.seeds),
            )
        except Exception as exc:  # noqa: BLE001
            gather_runs_repo.finish(conn, run_id, n_fetched=0, n_new=0, n_redup=0,
                                    tier_used="orcid_import", error=f"{type(exc).__name__}: {exc}")
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            print(f"draft {slug} left in place; delete it with DELETE /api/profiles/draft/{slug}",
                  file=sys.stderr)
            return 1
        gather_runs_repo.finish(conn, run_id, n_fetched=result.n_fetched, n_new=result.n_seeds,
                                n_redup=0, api_calls=client.api_calls, tier_used="orcid_import",
                                result_json=result.model_dump_json())

        out = {"slug": slug, "run_id": run_id, **result.model_dump()}
        if args.commit:
            tf = wizard_service.aggregate_draft_topics(conn, user_id=user_id, slug=slug)
            committed = wizard_service.commit_draft(
                conn, user_id=user_id, slug=slug, threshold=args.threshold,
                selected_topic_ids=enabled_topic_ids(tf),
            )
            out["committed"] = committed
            out["topics_on"] = [t["display_name"] for t in tf.get("topics", []) if t.get("on", True)]
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
