"""Create a radar draft from a Researcher Profile document, without the UI.

Runs the same service the ``POST /api/profiles/draft/from-profile`` route
dispatches, synchronously and with progress on stderr:

    python -m cli.import_profile profile.jsonld --user you@example.com
    python -m cli.import_profile profile.jsonld --user you@example.com \\
        --dry-run-signals                  # only show expertise/not_interests -> concepts
    python -m cli.import_profile profile.jsonld --user you@example.com \\
        --commit --threshold 0.85          # also aggregate concepts + commit

``--dry-run-signals`` is the tuning tool for RADAR_RP_EXPERTISE_MIN_SIM /
RADAR_RP_NOT_INTEREST_MIN_SIM: it needs ollama (mxbai-embed-large) and the
topic index but touches no database.

Reads settings the way the API does (``.env`` in the working directory,
``RADAR_*`` environment variables); ``--db`` overrides the database path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_lib.api.services import orcid_import, rp_profile_import, wizard as wizard_service
from rag_lib.api.settings import get_settings
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import gather_runs as gather_runs_repo
from rag_lib.db.repos import profiles as profiles_repo
from rag_lib.db.repos import users as users_repo
from rag_lib.openalex_client import OpenAlexClient
from rag_lib.openalex_tiers import enabled_topic_ids


class _StderrReporter:
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
    ap = argparse.ArgumentParser(prog="import_profile", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile", type=Path, help="path to profile.jsonld")
    ap.add_argument("--user", required=True, help="RADAR user email the draft belongs to")
    ap.add_argument("--name", default=None, help="topic name (default: the profile's name)")
    ap.add_argument("--mailto", default=None, help="OpenAlex polite-pool email (default: --user)")
    ap.add_argument("--db", type=Path, default=None, help="SQLite path (default: settings)")
    ap.add_argument("--min-sim", type=float, default=None,
                    help="override both RADAR_RP_*_MIN_SIM floors")
    ap.add_argument("--add-min-sim", type=float, default=None, help="override RADAR_RP_EXPERTISE_ADD_MIN_SIM")
    ap.add_argument("--top-k", type=int, default=None, help="override RADAR_RP_EXPERTISE_TOP_K")
    ap.add_argument("--dry-run-signals", action="store_true",
                    help="print expertise/not_interests -> concept hits and exit (no DB writes)")
    ap.add_argument("--seeds", default="default",
                    help="which fetched works become seeds: default (lead-author, newest N), "
                         "lead (every lead-author work), all (every seed-eligible work), "
                         "or ids:<file> (one OpenAlex id per line)")
    ap.add_argument("--commit", action="store_true", help="aggregate concepts and commit the draft")
    ap.add_argument("--threshold", type=float, default=0.85, help="selector threshold for --commit")
    return ap.parse_args()


def _print_hits(label: str, hits) -> None:
    print(f"\n{label}: {len(hits)} hit(s)")
    for h in hits:
        print(f"  {h.similarity:.3f}  {h.phrase!r:40} -> {h.display_name}  [{h.field or ''}]")


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
    if args.min_sim is not None:
        overrides["RADAR_RP_EXPERTISE_MIN_SIM"] = args.min_sim
        overrides["RADAR_RP_NOT_INTEREST_MIN_SIM"] = args.min_sim
    if args.add_min_sim is not None:
        overrides["RADAR_RP_EXPERTISE_ADD_MIN_SIM"] = args.add_min_sim
    if args.top_k is not None:
        overrides["RADAR_RP_EXPERTISE_TOP_K"] = args.top_k
    if overrides:
        settings = settings.model_copy(update=overrides)

    try:
        profile = rp_profile_import.parse_profile(args.profile.read_text(encoding="utf-8"))
    except (OSError, rp_profile_import.RpProfileError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"profile : {profile.name} | level={profile.level} provenance={profile.provenance} "
          f"orcid={profile.orcid} openalex={profile.openalex_author_id}", file=sys.stderr)
    print(f"          expertise={len(profile.expertise)} not_interests={len(profile.not_interests)} "
          f"collaborators={len(profile.collaborators)}", file=sys.stderr)
    for w in profile.warnings:
        print(f"warning : {w}", file=sys.stderr)

    if args.dry_run_signals:
        ex, w1 = rp_profile_import.map_phrases_to_topics(
            profile.expertise, settings=settings,
            min_sim=settings.RADAR_RP_EXPERTISE_MIN_SIM, top_k=settings.RADAR_RP_EXPERTISE_TOP_K)
        ni, w2 = rp_profile_import.map_phrases_to_topics(
            profile.not_interests, settings=settings,
            min_sim=settings.RADAR_RP_NOT_INTEREST_MIN_SIM, top_k=1)
        for w in w1 + w2:
            print(f"warning : {w}", file=sys.stderr)
        print(f"floors  : expertise>={settings.RADAR_RP_EXPERTISE_MIN_SIM} (switch on, top_k={settings.RADAR_RP_EXPERTISE_TOP_K}) "
              f"add>={settings.RADAR_RP_EXPERTISE_ADD_MIN_SIM} (best hit per phrase) "
              f"not_interest>={settings.RADAR_RP_NOT_INTEREST_MIN_SIM} (best hit per phrase)")
        _print_hits("EXPERTISE -> concepts (switch on if present)", ex)
        best = {}
        for h in ex:
            best.setdefault(h.phrase, h)
        _print_hits("EXPERTISE -> concepts that would be ADDED if absent (best hit >= add floor)",
                    [h for h in best.values() if h.similarity >= settings.RADAR_RP_EXPERTISE_ADD_MIN_SIM])
        _print_hits("NOT_INTERESTS -> concepts (switch off)", ni)
        return 0

    conn = connect(settings.RADAR_DB_PATH)
    try:
        apply_migrations(conn)
        user = users_repo.upsert(conn, args.user)
        user_id = int(user["id"])
        name = (args.name or profile.name).strip()
        client = OpenAlexClient(mailto=args.mailto or args.user)

        author = None
        if profile.orcid is not None:
            author = rp_profile_import.resolve_author_for_profile(client, profile)
            if author is None:
                print(f"error: ORCID {profile.orcid} not found on OpenAlex", file=sys.stderr)
                return 1
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
        rp_profile_import.persist_meta(conn, profile_id, profile.meta(), researcher_name=profile.name)

        if profile.orcid is None:
            print(f"draft {slug} created without seeds (no ORCID in the profile); "
                  f"upload PDFs, the expertise signal applies at Step 3", file=sys.stderr)
            print(json.dumps({"slug": slug, "mode": "pdf"}, indent=2))
            return 0

        profiles_repo.set_orcid(conn, profile_id, orcid=profile.orcid, researcher_name=profile.name)
        run_id = gather_runs_repo.start(conn, profile_id=profile_id, user_id=user_id,
                                        tier_used="orcid_import")
        try:
            result = rp_profile_import.run_profile_import(
                conn, settings, user_id=user_id, profile_id=profile_id, slug=slug,
                profile=profile, name=name, client=client, reporter=_StderrReporter(), author=author,
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
        out = result.model_dump()
        out["slug"] = slug
        print(json.dumps(out, indent=2, ensure_ascii=False))

        if args.commit:
            tf = wizard_service.aggregate_draft_topics(conn, user_id=user_id, slug=slug)
            committed = wizard_service.commit_draft(
                conn, user_id=user_id, slug=slug, threshold=args.threshold,
                selected_topic_ids=enabled_topic_ids(tf),
            )
            print(f"committed: {committed}", file=sys.stderr)
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
