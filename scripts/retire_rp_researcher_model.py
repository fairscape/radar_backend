#!/usr/bin/env python3
"""One-off cutover from the RP-document researcher model to this one.

One deployment ran a different researcher model for three weeks: its
migrations 0015_orcid / 0016_rp_profile / 0017_orcid_works /
0018_researchers put the person's identity on the profile row
(``profiles.orcid``, ``researcher_name``, ``rp_meta_json``), kept a
verbatim archive of pasted ``profile.jsonld`` documents in a table also
called ``researchers``, and used ``orcid_works`` as a seed-picking
scratchpad. That model is retired; this one keeps researchers as
first-class rows with ``researcher_papers`` membership.

This is deliberately NOT a migration. The reconciliation only makes
sense against that one database: every statement below names a table
that no other deployment has, and a migration file naming them would
fail on a fresh install, in CI, and in every test that builds a database
from ``migrations/`` -- and since ``apply_migrations`` runs unguarded in
the API's lifespan, that failure is a boot failure. So the work lives
here, the migrations stay exactly as written, and an operator runs this
once, by hand, with the service stopped.

    python scripts/retire_rp_researcher_model.py --db PATH [--dry-run]

Order matters:

    1. retire   rename the legacy tables aside, drop the index whose name
                the new schema needs, forget the retired versions
    2. migrate  apply 0015_seed_similarity and 0016_researchers normally
    3. backfill create one researcher per (user_id, normalized orcid) and
                point the profiles that came from it at that row

Re-running is safe: every step detects whether it has already happened.
Step 1's statements are DDL, which sqlite3 autocommits individually, so
step 1 is *not* one transaction -- a crash between its statements leaves
a mixed state, and the re-run detection is what recovers it. Step 3 is
one transaction.

What is deliberately NOT carried across
---------------------------------------
``researcher_papers`` is left empty. Under the new model that table is
ownership, not history: ``db/repos/vault.py`` and
``db/repos/researchers.py:user_owns_papers`` both read it to decide which
papers a user may see and seed with. ``orcid_works`` was wizard scratch
and fed neither, and 81 of its 454 rows are copies the old rule collapsed
(``dup_of``) or works it excluded (``seed_eligible = 0``). Copying it
would drop hundreds of works the user explicitly declined into their
vault, and would seed future interests with both halves of every
preprint/published pair. Re-importing a researcher costs about two
OpenAlex calls, so the cheap, correct move is to let the user do that.

The two rows in the legacy archive are not carried across either: they
are pasted RP documents, and the wizard path that produced them is gone.
They stay readable in ``_researchers_rp_archive`` for one release.

``gather_runs.researcher_id`` is left NULL on the historical imports, and
this one is worth spelling out because the opposite looks helpful.
``repos/gather_runs.py`` states the invariant: a researcher import has no
profile, a scan has no researcher -- never both. Attaching a researcher
to a run that already has a ``profile_id`` breaks it, and
``gather_runs.researcher_id`` is ``ON DELETE CASCADE``: deleting that
researcher would then take the run rows with it (measured: 5 of 1110 for
one person), which is exactly the history the attachment was meant to
preserve, and contradicts ``delete_researcher``'s contract that a
researcher's papers and interests survive. Worse, once such a run has
``profile_candidates`` children -- and every ordinary gather produces
them -- the cascade hits ``profile_candidates.gather_run_id``, which has
no ``ON DELETE`` clause, and the whole DELETE aborts: the researcher
becomes permanently undeletable. Linking history is not worth that.
"""

from __future__ import annotations

import argparse
import socket
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_lib.api.services.orcid import normalize_orcid  # noqa: E402
from rag_lib.db.migrate import apply_migrations  # noqa: E402


LEGACY_TABLES = {"researchers": "_researchers_rp_archive",
                 "orcid_works": "_orcid_works_legacy"}
# The old index on researchers(user_id, name). SQLite keeps an index's
# name when its table is renamed, so this one would still own the name
# that 0016_researchers needs for its own index.
LEGACY_INDEX = "idx_researchers_user"
# Recorded as applied on this database and nowhere else. Left in place
# they would silently shadow any future migration that reuses the stem --
# 0018_researchers is a plausible future filename given 0016's name -- so
# that file would be skipped here and applied everywhere else.
RETIRED_VERSIONS = ("0015_orcid", "0016_rp_profile",
                    "0017_orcid_works", "0018_researchers")
# Columns step 3 reads or writes. Checked before step 1 touches anything:
# a database with `orcid` but not `researcher_name` (a legacy state caught
# between its own 0015 and 0016) would otherwise fail in step 3, after the
# renames and the migrations had already committed.
REQUIRED_COLUMNS = (("profiles", "orcid"), ("profiles", "researcher_name"))
API_PORT = 8000


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(r["name"] == column
               for r in conn.execute(f"PRAGMA table_info({table})"))


def _backends_using(db: Path) -> list[int]:
    """PIDs of running RADAR backends configured to use ``db``.

    Neither a lock probe nor a check for an open file descriptor can
    answer this. The file is in WAL mode, where ``BEGIN IMMEDIATE``
    conflicts only with another *active* writer, and the API opens a
    connection per request (``api/deps.py:get_db``) and closes it again --
    so between requests it holds no lock and no descriptor. Both checks
    pass against a live service, which is the failure they were meant to
    catch: the watchdog restarts uvicorn mid-cutover, its lifespan calls
    ``apply_migrations`` on the same file, or the in-process scheduler
    inserts into ``gather_runs`` while 0016 is rebuilding it.

    What is stable is the process's *configuration*. Find the uvicorn
    processes, read each one's working directory, and resolve the
    ``RADAR_DB_PATH`` in the ``.env`` it loads from there. That is keyed on
    the database rather than on the port, so rehearsing the cutover
    against a copy while the service serves the real one -- exactly what
    testing looks like -- is allowed.
    """
    target = db.resolve()
    pids: list[int] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            if "uvicorn" not in cmdline or "rag_lib.api.app" not in cmdline:
                continue
            cwd = Path((proc / "cwd").resolve())
            env_file = cwd / ".env"
            configured = "data/radar.db"      # the app's own default
            if env_file.is_file():
                for line in env_file.read_text().splitlines():
                    if line.startswith("RADAR_DB_PATH="):
                        configured = line.split("=", 1)[1].strip()
            if (cwd / configured).resolve() == target:
                pids.append(int(proc.name))
        except (OSError, UnicodeDecodeError, ValueError):
            continue  # process gone, or not ours to inspect
    return pids


def _require_exclusive(db: Path) -> None:
    """Refuse while a running backend is configured to use this database."""
    pids = _backends_using(db)
    if not pids:
        return
    listening = ""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        if s.connect_ex(("127.0.0.1", API_PORT)) == 0:
            listening = f", and something is serving on port {API_PORT}"
    raise SystemExit(
        f"pid(s) {', '.join(map(str, pids))} are running against {db}{listening}.\n"
        "Stop the service, and disable the cron watchdog first so it cannot\n"
        "restart it in the middle of the cutover:\n"
        "    crontab -l > /tmp/radar.cron.bak\n"
        "    crontab -l | grep -v watchdog | crontab -\n"
        "    tmux kill-session -t radar-backend\n"
        "Restore the crontab afterwards with: crontab /tmp/radar.cron.bak"
    )


def check_preconditions(conn: sqlite3.Connection) -> list[str]:
    """Everything step 3 will need, verified before step 1 mutates anything."""
    out: list[str] = []
    if not _has_column(conn, "profiles", "orcid"):
        return ["  profiles.orcid absent -- this is not the database this "
                "script is for; steps 1 and 3 will no-op"]
    missing = [f"{t}.{c}" for t, c in REQUIRED_COLUMNS
               if not _has_column(conn, t, c)]
    if missing:
        raise SystemExit(
            "the legacy model is half-present: " + ", ".join(missing) +
            " missing.\nRefusing to start, because step 3 would fail after "
            "steps 1 and 2 had already committed."
        )
    out.append("  legacy columns present: " +
               ", ".join(f"{t}.{c}" for t, c in REQUIRED_COLUMNS))
    return out


def step_retire(conn: sqlite3.Connection, dry: bool) -> list[str]:
    done: list[str] = []
    for old, new in LEGACY_TABLES.items():
        has_old, has_new = _has_table(conn, old), _has_table(conn, new)
        if has_old and has_new:
            raise SystemExit(
                f"both {old} and {new} exist. The legacy table is still on "
                f"the name the new schema needs, so step 2 would fail with "
                f"'table {old} already exists'. Sort out by hand which one "
                f"is authoritative before re-running."
            )
        if has_new:
            done.append(f"  already retired: {old} -> {new}")
        elif not has_old:
            done.append(f"  absent, nothing to retire: {old}")
        else:
            done.append(f"  rename {old} -> {new}")
            if not dry:
                conn.execute(f"ALTER TABLE {old} RENAME TO {new}")

    # Drop the old index only while it is still the old one. 0016 creates
    # an index of the same name on the new table, and dropping that would
    # be destructive. Renaming a table rewrites its indexes' tbl_name in
    # sqlite_master, and --dry-run has not renamed anything yet, so the
    # table's *shape* is the discriminator that holds in every state: only
    # the new researchers table has a `source` column.
    row = conn.execute(
        "SELECT tbl_name FROM sqlite_master WHERE type='index' AND name=?",
        (LEGACY_INDEX,),
    ).fetchone()
    if row is None:
        done.append(f"  absent: index {LEGACY_INDEX}")
    elif _has_column(conn, row["tbl_name"], "source"):
        done.append(
            f"  keep index {LEGACY_INDEX} (already the new one, on {row['tbl_name']})")
    else:
        done.append(f"  drop index {LEGACY_INDEX} (name needed by 0016)")
        if not dry:
            conn.execute(f"DROP INDEX {LEGACY_INDEX}")

    stale = [v for v in RETIRED_VERSIONS if conn.execute(
        "SELECT 1 FROM _migrations_applied WHERE version=?", (v,)).fetchone()]
    if stale:
        done.append(f"  forget retired versions: {', '.join(stale)}")
        if not dry:
            conn.executemany(
                "DELETE FROM _migrations_applied WHERE version=?",
                [(v,) for v in stale])
            conn.commit()
    else:
        done.append("  no retired versions recorded")
    return done


def step_backfill(conn: sqlite3.Connection, dry: bool) -> list[str]:
    out: list[str] = []
    if not _has_column(conn, "profiles", "orcid"):
        return ["  profiles.orcid absent -- nothing from the old model to carry"]

    # The newest profile's name, not MAX(): MAX over TEXT is alphabetical,
    # so for the case this is meant to handle -- a person renamed between
    # two imports -- it picks the stale spelling about half the time, and
    # nothing later refreshes it (imported_at stays NULL by design).
    pairs = conn.execute(
        """
        SELECT p.user_id, p.orcid, COUNT(*) AS n,
               (SELECT p2.researcher_name FROM profiles p2
                 WHERE p2.user_id = p.user_id AND p2.orcid = p.orcid
                   AND p2.researcher_name IS NOT NULL
                 ORDER BY p2.id DESC LIMIT 1) AS newest_name
          FROM profiles p
         WHERE p.orcid IS NOT NULL AND p.user_id IS NOT NULL
         GROUP BY p.user_id, p.orcid
        """
    ).fetchall()
    out.append(f"  {sum(r['n'] for r in pairs)} profiles over "
               f"{len(pairs)} (user, orcid) pair(s)")

    # researchers.user_id is NOT NULL but profiles.user_id never was
    # (0002_users added it nullable and nothing has enforced it since).
    orphans = conn.execute(
        "SELECT COUNT(*) AS n FROM profiles "
        "WHERE orcid IS NOT NULL AND user_id IS NULL"
    ).fetchone()["n"]
    if orphans:
        out.append(f"  WARNING: {orphans} orcid profile(s) have no user_id and "
                   "are skipped; they keep working without a researcher")

    # Normalize through the app's own function, not a SQL UPPER(TRIM(...)):
    # `key` has to be byte-identical to what a later re-import looks up.
    # services/researchers.py stores key=normalize_orcid(ref), and
    # repos/researchers.py:get_by_key compares with plain `=` on TEXT, so a
    # key that differs by case or an orcid.org prefix means the re-import
    # misses this row and inserts a second copy of the same person.
    plan: list[tuple[int, str, str, str]] = []   # user_id, key, name, raw
    for r in pairs:
        raw = r["orcid"]
        key = normalize_orcid(raw)
        if key is None:
            out.append(f"  WARNING: profiles.orcid {raw!r} is not a valid "
                       "ORCID iD; its profile(s) are skipped")
            continue
        if key != raw:
            out.append(f"  normalized {raw!r} -> {key!r}")
        plan.append((int(r["user_id"]), key, r["newest_name"] or key, raw))

    if dry:
        out.append("  (link counts cannot be previewed before 0016 has run)")
        return out

    created = linked = 0
    for user_id, key, name, raw in plan:
        # OR IGNORE, then read the id back: two profiles of one person
        # stored in different casings collapse to one key here, and a
        # re-run finds the row already present.
        conn.execute(
            """
            INSERT OR IGNORE INTO researchers
                   (user_id, source, key, orcid, name, url, n_papers)
            VALUES (?, 'orcid', ?, ?, ?, ?, 0)
            """,
            (user_id, key, key, name, f"https://orcid.org/{key}"),
        )
        row = conn.execute(
            "SELECT id FROM researchers WHERE user_id=? AND source='orcid' AND key=?",
            (user_id, key),
        ).fetchone()
        if row is None:
            # OR IGNORE swallows a foreign-key failure as readily as a
            # uniqueness one, so an unresolvable user_id would otherwise
            # look like a successful skip.
            raise SystemExit(
                f"could not create or find a researcher for user {user_id} "
                f"orcid {key} -- check that user {user_id} exists in users"
            )
        created += 1
        # Match on the raw value: that is what the profile rows hold.
        linked += conn.execute(
            "UPDATE profiles SET researcher_id=? "
            "WHERE user_id=? AND orcid=? AND researcher_id IS NULL",
            (int(row["id"]), user_id, raw),
        ).rowcount
    conn.commit()

    # An absolute count would hide a partial backfill, so compare against
    # what the plan said.
    total_linked = conn.execute(
        "SELECT COUNT(*) AS n FROM profiles WHERE researcher_id IS NOT NULL"
    ).fetchone()["n"]
    out.append(f"  researchers resolved: {created} of {len(plan)} planned")
    out.append(f"  profiles linked this run: {linked}; "
               f"linked in total: {total_linked}")
    if created != len(plan):
        raise SystemExit("backfill incomplete: see above")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, required=True,
                    help="database file (named explicitly; there is no default)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would happen and change nothing")
    args = ap.parse_args()

    if not args.db.exists():
        print(f"no such database: {args.db}", file=sys.stderr)
        return 2

    if not args.dry_run:
        _require_exclusive(args.db)

    conn = sqlite3.connect(str(args.db))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # The gather_runs rebuild in 0016 is the longest write here; give it the
    # same patience the app uses rather than failing fast on a transient.
    conn.execute("PRAGMA busy_timeout = 30000")

    print(f"database: {args.db}")
    print("\n0. preconditions")
    for line in check_preconditions(conn):
        print(line)

    print("\n1. retire the legacy model")
    for line in step_retire(conn, args.dry_run):
        print(line)

    print("\n2. apply migrations")
    if args.dry_run:
        print("  (skipped in --dry-run)")
    else:
        applied = apply_migrations(conn)
        for v in applied:
            print(f"  + {v}")
        if not applied:
            print("  already up to date")

    print("\n3. backfill researchers")
    for line in step_backfill(conn, args.dry_run):
        print(line)

    if not args.dry_run:
        print("\n4. verify")
        ok = conn.execute("PRAGMA integrity_check").fetchone()[0]
        print(f"  integrity_check:   {ok}")
        viol = conn.execute("PRAGMA foreign_key_check").fetchall()
        print(f"  foreign_key_check: {len(viol)} violation(s)")
        if ok != "ok" or viol:
            return 1
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
