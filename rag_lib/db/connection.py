"""SQLite connection helper.

Single entry point that every reader/writer goes through so foreign-key
enforcement, journaling, and the row factory are consistent. Callers
pass a path; we handle parent-directory creation, pragmas, and row-typing.

WAL keeps the CLI and the FastAPI service from blocking each other when
both touch the same file, so it is what we use on a local disk. It is not
safe everywhere: WAL coordinates its writers through a shared-memory
``-shm`` file, which a network filesystem cannot provide coherently, and
SQLite documents the combination as unsupported -- the failure mode is a
corrupt file, not a stalled write. So the mode is chosen from where the
database actually lives, and ``RADAR_SQLITE_JOURNAL_MODE`` overrides that
choice when the guess is wrong.

Foreign keys are off by default in SQLite -- turn them on every connection.
"""

from __future__ import annotations

import os
import sqlite3
from functools import lru_cache
from pathlib import Path

# Modes SQLite persists in the file header. MEMORY and OFF are deliberately
# absent: both trade away crash durability, which is never what a
# deployment wants from this switch.
_JOURNAL_MODES = {"wal", "delete", "truncate", "persist"}

# Filesystems that cannot back WAL's shared-memory segment. Anything not
# listed is assumed local; a false "local" only costs us the corruption we
# are trying to avoid, so keep this list generous.
#: Filesystem types that are not safe for WAL, plus the ones we cannot
#: judge. ``autofs`` earns its place the hard way: on this deployment's
#: host /proc/mounts lists /p, /u, /s and /a as autofs trigger points with
#: **no** backing nfs line at all, so the longest-prefix match answers
#: "autofs" for every path under them. Classifying that as local put WAL
#: on NFS -- the exact configuration that corrupted the database on
#: 2026-08-28. ``fuse``/``overlay`` are here for the same reason: what is
#: underneath them is not knowable from here, and the cost of guessing
#: "local" wrongly is a corrupt file, while the cost of guessing "remote"
#: wrongly is writers serialising.
_UNSAFE_FSTYPES = {
    "autofs", "fuse", "overlay", "overlayfs", "unknown",
}
_NETWORK_FSTYPES = {
    "nfs", "nfs4", "cifs", "smb3", "smbfs", "afs", "9p",
    "lustre", "gpfs", "beegfs", "ceph", "glusterfs", "fuse.glusterfs",
    "fuse.sshfs", "fuse.s3fs",
}


def _fstype(path: Path) -> str | None:
    """Filesystem type backing ``path``, or None if it can't be determined.

    Walks up to the nearest existing ancestor, since the database file (and
    even its directory) may not exist yet on a first run.
    """
    try:
        mounts = Path("/proc/mounts").read_text().splitlines()
        # resolve() and exists() are inside the guard on purpose: a symlink
        # loop, or an ESTALE/EIO on a dead NFS ancestor, would otherwise
        # raise out of connect() and break every database open in the
        # process -- a guess about journal mode must never do that.
        probe = path.resolve()
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
    except OSError:
        return None  # not Linux, /proc absent, or the path is unreachable
    probe_str = str(probe)

    best = None
    best_len = -1
    for line in mounts:
        parts = line.split()
        if len(parts) < 3:
            continue
        point, fstype = parts[1].replace("\\040", " "), parts[2]
        if probe_str == point or probe_str.startswith(point.rstrip("/") + "/"):
            # >= so that a later mount stacked on the same point wins: an
            # automounter lists the trigger (autofs) before the filesystem
            # it eventually mounts there.
            if len(point) >= best_len:
                best, best_len = fstype, len(point)
    return best


@lru_cache(maxsize=32)
def _default_journal_mode_cached(resolved: str) -> str:
    fs = _fstype(Path(resolved))
    if fs is None or fs in _NETWORK_FSTYPES or fs in _UNSAFE_FSTYPES:
        # Unknown counts as unsafe. WAL on a network filesystem corrupts
        # the file; DELETE on a local one only serialises writers.
        return "delete"
    return "wal"


def _default_journal_mode(path: Path) -> str:
    """Cached: ``connect()`` is the per-request ``get_db`` dependency.

    Without the cache every request re-read /proc/mounts and walked the
    path with ``exists()`` -- and on an autofs tree ``exists()`` *triggers*
    the automounter, so opening a database could stall on an unreachable
    one where before it touched no metadata at all. The answer cannot
    change for a path within a process.
    """
    try:
        key = str(path.resolve())
    except OSError:
        key = str(path)
    return _default_journal_mode_cached(key)


def connect(path: Path | str) -> sqlite3.Connection:
    """Open (or create) the SQLite database at ``path``.

    Creates the parent directory if missing. Sets:
      - ``foreign_keys = ON`` (per-connection in SQLite)
      - ``journal_mode`` -- WAL on a local disk, DELETE on a network
        filesystem, or whatever ``RADAR_SQLITE_JOURNAL_MODE`` names
        (persists across connections)
      - ``row_factory = sqlite3.Row`` so callers can access columns by name
    """
    p = Path(path)

    override = os.environ.get("RADAR_SQLITE_JOURNAL_MODE", "").strip().lower()
    if override and override not in _JOURNAL_MODES:
        raise ValueError(
            f"RADAR_SQLITE_JOURNAL_MODE={override!r} is not one of "
            f"{sorted(_JOURNAL_MODES)}"
        )
    want = override or _default_journal_mode(p)

    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(p),
        detect_types=sqlite3.PARSE_DECLTYPES,
        check_same_thread=False,
        timeout=30.0,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    # journal_mode persists in the file header, so setting it is a one-time
    # job -- but `PRAGMA journal_mode = ...` grabs an exclusive lock even
    # when the mode already matches. Doing that on every connection meant a
    # single in-flight write stalled every other request, readers included,
    # for the full busy timeout. Read the mode first (lock-free) and only
    # write it when the file is genuinely on the wrong mode.
    current = conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
    if current != want:
        # PRAGMA journal_mode returns the mode that is in force afterwards,
        # and leaving WAL needs exclusive access -- so with another
        # connection open this either raises SQLITE_BUSY or comes back
        # still saying "wal". Neither may pass silently: an ignored
        # failure here is a database left in WAL on a filesystem that
        # cannot support it, which is how it corrupted on 2026-08-28.
        try:
            got = conn.execute(f"PRAGMA journal_mode = {want}").fetchone()
        except sqlite3.OperationalError as exc:
            raise RuntimeError(
                f"could not switch {p} from {current!r} to {want!r}: {exc}. "
                "Another connection is open; close every reader and writer "
                "and retry, or set RADAR_SQLITE_JOURNAL_MODE deliberately."
            ) from exc
        landed = (got[0] if got else "").lower()
        if landed != want:
            raise RuntimeError(
                f"{p} is in {landed!r} journal mode, not the {want!r} this "
                "filesystem requires. Another connection is holding it "
                "there; stop them and retry."
            )
    return conn
