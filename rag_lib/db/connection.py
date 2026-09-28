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
from pathlib import Path

# Modes SQLite persists in the file header. MEMORY and OFF are deliberately
# absent: both trade away crash durability, which is never what a
# deployment wants from this switch.
_JOURNAL_MODES = {"wal", "delete", "truncate", "persist"}

# Filesystems that cannot back WAL's shared-memory segment. Anything not
# listed is assumed local; a false "local" only costs us the corruption we
# are trying to avoid, so keep this list generous.
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
    except OSError:
        return None  # not Linux, or /proc not mounted

    probe = path.resolve()
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
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


def _default_journal_mode(path: Path) -> str:
    return "delete" if _fstype(path) in _NETWORK_FSTYPES else "wal"


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
    if conn.execute("PRAGMA journal_mode").fetchone()[0].lower() != want:
        conn.execute(f"PRAGMA journal_mode = {want}")
    return conn
