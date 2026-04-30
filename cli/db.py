"""DB management CLI.

Usage::

    python -m cli.db init
    python -m cli.db status
    python -m cli.db upgrade
    python -m cli.db reset --yes
    python -m cli.db profiles --user demo@example.com
    python -m cli.db candidates --profile <slug> --user demo@example.com [--unshown]

The default DB path is ``data/radar.db`` (relative to the working
directory). Override with ``--db PATH`` on any subcommand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_lib.db import apply_migrations, applied_versions, connect
from rag_lib.db.repos import candidates, profiles, users


DEFAULT_DB = Path("data/radar.db")
DEFAULT_USER = "demo@example.com"


def _add_db_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", type=Path, default=DEFAULT_DB,
                   help=f"SQLite path (default: {DEFAULT_DB}).")


def _resolve_user_id(conn, email: str) -> int:
    row = users.get_by_email(conn, email)
    if row is None:
        # Auto-upsert is convenient for the demo; documented in phase-12 plan.
        row = users.upsert(conn, email)
    return int(row["id"])


def cmd_init(args: argparse.Namespace) -> int:
    conn = connect(args.db)
    new = apply_migrations(conn)
    if new:
        print(f"Applied {len(new)} migration(s):")
        for v in new:
            print(f"  + {v}")
    else:
        print("Database is up to date.")
    print(f"DB path: {args.db}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    if not Path(args.db).exists():
        print(f"No database at {args.db}. Run `python -m cli.db init`.")
        return 1
    conn = connect(args.db)
    rows = applied_versions(conn)
    if not rows:
        print("No migrations recorded.")
        return 0
    for version, applied_at in rows:
        print(f"  {version:30s}  applied {applied_at}")
    return 0


def cmd_upgrade(args: argparse.Namespace) -> int:
    return cmd_init(args)


def cmd_reset(args: argparse.Namespace) -> int:
    if not args.yes:
        print("Refusing to delete without --yes.", file=sys.stderr)
        return 2
    p = Path(args.db)
    for path in (p, p.with_name(p.name + "-wal"), p.with_name(p.name + "-shm")):
        if path.exists():
            path.unlink()
            print(f"Removed {path}")
    return 0


def cmd_profiles(args: argparse.Namespace) -> int:
    conn = connect(args.db)
    user_id = _resolve_user_id(conn, args.user)
    rows = profiles.list_for_user(conn, user_id)
    if not rows:
        print(f"No profiles for {args.user}.")
        return 0
    for r in rows:
        coh = r["coherence_median"]
        coh_s = f"{coh:.3f}" if coh is not None else "-"
        thr = r["threshold"]
        thr_s = f"{thr:.3f}" if thr is not None else "-"
        print(
            f"  [{r['id']:3d}] {r['slug']:30s}  "
            f"seeds={r['n_seed']:3d}  coherence={coh_s}  threshold={thr_s}  "
            f"model={r['embedding_model']}"
        )
    return 0


def cmd_candidates(args: argparse.Namespace) -> int:
    conn = connect(args.db)
    user_id = _resolve_user_id(conn, args.user)
    profile_row = profiles.get_by_slug(conn, user_id, args.profile)
    if profile_row is None:
        print(f"No profile '{args.profile}' for {args.user}.", file=sys.stderr)
        return 1
    if args.unshown:
        rows = candidates.unshown_for_profile(conn, profile_row["id"], limit=args.limit)
    else:
        rows = conn.execute(
            """
            SELECT pc.*, p.title, p.year
            FROM profile_candidates pc
            JOIN papers p USING (openalex_id)
            WHERE pc.profile_id = ?
            ORDER BY pc.score DESC
            LIMIT ?
            """,
            (profile_row["id"], args.limit),
        ).fetchall()
    if not rows:
        print("(no candidates)")
        return 0
    for r in rows:
        flag = (
            "S" if r["saved_at"] else
            "D" if r["dismissed_at"] else
            "·" if r["shown_at"] else " "
        )
        print(f"  [{flag}] {r['score']:.3f}  ({r['year'] or '----'})  {r['title'][:90]}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="db", description="RADAR database management.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Create the DB and apply all migrations.")
    _add_db_arg(p_init)
    p_init.set_defaults(func=cmd_init)

    p_status = sub.add_parser("status", help="List applied migrations.")
    _add_db_arg(p_status)
    p_status.set_defaults(func=cmd_status)

    p_upgrade = sub.add_parser("upgrade", help="Apply any pending migrations.")
    _add_db_arg(p_upgrade)
    p_upgrade.set_defaults(func=cmd_upgrade)

    p_reset = sub.add_parser("reset", help="Delete the DB file (irreversible).")
    _add_db_arg(p_reset)
    p_reset.add_argument("--yes", action="store_true", help="Confirm deletion.")
    p_reset.set_defaults(func=cmd_reset)

    p_profiles = sub.add_parser("profiles", help="List profiles for a user.")
    _add_db_arg(p_profiles)
    p_profiles.add_argument("--user", default=DEFAULT_USER)
    p_profiles.set_defaults(func=cmd_profiles)

    p_cands = sub.add_parser("candidates", help="Inspect a profile's candidates.")
    _add_db_arg(p_cands)
    p_cands.add_argument("--user", default=DEFAULT_USER)
    p_cands.add_argument("--profile", required=True, help="Profile slug.")
    p_cands.add_argument("--unshown", action="store_true",
                         help="Restrict to candidates not yet shown / dismissed.")
    p_cands.add_argument("--limit", type=int, default=20)
    p_cands.set_defaults(func=cmd_candidates)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
