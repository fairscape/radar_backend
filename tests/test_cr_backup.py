"""deploy/backup.sh after the 2026-11-02 review (FUNCTIONAL_SPEC CR.9, L14).

Runs the script's own functions (it is sourced) against a scratch
database and a local directory standing in for R2: nothing here reaches
the live database, the real vault or the bucket.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "backup.sh"
DEPLOY = Path("/bigtemp/nkw3mr/radar_deployment")
RCLONE = DEPLOY / "bin" / "rclone"
SQLITE3 = DEPLOY / "conda_env312" / "bin" / "sqlite3"

pytestmark = pytest.mark.skipif(not (RCLONE.is_file() and SQLITE3.is_file()), reason="needs the deployment's rclone and sqlite3")


@pytest.fixture
def box(tmp_path):
    vault, remote, local = tmp_path / "vault", tmp_path / "remote", tmp_path / "backups"
    for d in (vault, remote / "vault", local):
        d.mkdir(parents=True)
    conf = tmp_path / "rclone.conf"
    conf.write_text("")
    env = {
        "PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path),
        "RCLONE_CONFIG": str(conf),
        "BACKUP_DB": str(tmp_path / "radar.db"), "BACKUP_VAULT": str(vault),
        "BACKUP_LOCAL": str(local), "BACKUP_REMOTE": str(remote),
        "BACKUP_RCLONE": str(RCLONE), "BACKUP_SQLITE3": str(SQLITE3),
    }
    return {"vault": vault, "remote": remote, "local": local, "db": tmp_path / "radar.db", "env": env}


def _run(box, call, **extra):
    assert box["env"]["BACKUP_DB"] != "/var/tmp/radar-nkw3mr/radar.db"
    r = subprocess.run(["bash", "-c", f'source "{SCRIPT}"; {call}'], env={**box["env"], **extra},
                       capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout + r.stderr


def _files(d: Path):
    return sorted(str(p.relative_to(d)) for p in d.rglob("*") if p.is_file())


def _put(box, name, data):
    p = box["vault"] / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def test_cr9_a_removed_or_replaced_pdf_goes_to_vault_deleted(box):
    for n in ("u1/a.pdf", "u1/b.pdf", "u2/c.pdf"):
        _put(box, n, n.encode())
    code, out = _run(box, "vault_sync 20261102")
    assert code == 0, out
    assert _files(box["remote"] / "vault") == ["u1/a.pdf", "u1/b.pdf", "u2/c.pdf"]
    (box["vault"] / "u1/a.pdf").unlink()
    _put(box, "u1/b.pdf", b"new b")
    code, out = _run(box, "vault_sync 20261103")
    assert code == 0, out
    assert _files(box["remote"] / "vault") == ["u1/b.pdf", "u2/c.pdf"]
    kept = box["remote"] / "vault-deleted" / "20261103"
    assert _files(kept) == ["u1/a.pdf", "u1/b.pdf"]
    assert (kept / "u1/b.pdf").read_bytes() == b"u1/b.pdf"


@pytest.mark.parametrize("left", [0, 1])
def test_cr9_a_vault_with_less_than_half_the_remote_files_still_syncs_and_pins(box, left):
    """The user's choice (2026-11-02): no refusal. The sync runs; what it
    removes is kept in vault-pinned/<day>/, never pruned, and reported."""
    for i in range(4):
        _put(box, f"u1/{i}.pdf", b"x%d" % i)
    assert _run(box, "vault_sync 20261102")[0] == 0
    for i in range(left, 4):
        (box["vault"] / f"u1/{i}.pdf").unlink()
    code, out = _run(box, "vault_sync 20261103")
    assert code == 0, out
    assert "WARN" in out and "vault shrank" in out, out
    assert len(_files(box["remote"] / "vault")) == left
    assert len(_files(box["remote"] / "vault-pinned" / "20261103")) == 4 - left
    assert not (box["remote"] / "vault-deleted").exists()
    assert "vault shrank: %d files here, 4 on R2" % left in (box["local"] / "pinned.log").read_text()


def test_cr9_pinned_vault_files_are_never_pruned(box):
    import datetime as dt

    old = (dt.date.today() - dt.timedelta(days=400)).strftime("%Y%m%d")
    p = box["remote"] / "vault-pinned" / old / "u1" / "x.pdf"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"x")
    assert _run(box, "prune_vault_deleted")[0] == 0
    assert p.is_file()


def test_cr9_half_the_files_still_sync(box):
    for i in range(4):
        _put(box, f"u1/{i}.pdf", b"x%d" % i)
    assert _run(box, "vault_sync 20261102")[0] == 0
    for i in (2, 3):
        (box["vault"] / f"u1/{i}.pdf").unlink()
    assert _run(box, "vault_sync 20261103")[0] == 0


def test_cr9_vault_deleted_is_kept_sixty_days_by_its_name(box):
    import datetime as dt

    old = (dt.date.today() - dt.timedelta(days=61)).strftime("%Y%m%d")
    recent = (dt.date.today() - dt.timedelta(days=59)).strftime("%Y%m%d")
    for d in (old, recent):
        p = box["remote"] / "vault-deleted" / d / "u1" / "x.pdf"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"x")
    (box["remote"] / "vault-deleted" / "notes").mkdir()
    code, out = _run(box, "prune_vault_deleted")
    assert code == 0, out
    assert sorted(p.name for p in (box["remote"] / "vault-deleted").iterdir()) == sorted([recent, "notes"])


def _db(box, n_papers):
    p = box["db"]
    p.unlink(missing_ok=True)
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE papers (openalex_id TEXT)")
    c.execute("CREATE TABLE profiles (id INTEGER)")
    c.executemany("INSERT INTO papers VALUES (?)", [(f"W{i}",) for i in range(n_papers)])
    c.commit()
    c.close()


def _backup(box, n, **extra):
    import time

    _db(box, n)
    code, out = _run(box, "do_backup", **extra)
    time.sleep(1.1)          # the next snapshot must be the newer by mtime
    return code, out


def _snapshots(box):
    return len(list((box["local"] / "hourly").glob("radar-*.db.gz")))


@pytest.mark.parametrize("before,after,pinned", [(9, 0, True), (100, 0, True), (100, 9, True), (100, 10, False), (1, 1, False), (0, 0, False)])
def test_l14_a_shrink_keeps_backing_up_and_pins_the_copy_before_it(box, before, after, pinned):
    """The user's choice (2026-11-02): the snapshot is kept either way; a
    shrink pins the previous one in pinned/, which nothing prunes."""
    code, out = _backup(box, before)
    assert code == 0, out
    # Two snapshots in one minute share a name; move the first aside.
    first = next((box["local"] / "hourly").glob("radar-*.db.gz"))
    first = first.rename(first.with_name("radar-20000101-0000.db.gz"))
    code, out = _backup(box, after)
    assert code == 0, out
    assert _snapshots(box) == 2
    kept = list((box["local"] / "pinned").glob("*.db.gz")) if (box["local"] / "pinned").exists() else []
    if pinned:
        assert [k.name for k in kept] == [first.name], out
        assert k_bytes_equal(kept[0], first)
        assert "WARN" in out and "database shrank" in out, out
        assert f"previous one had {before}" in (box["local"] / "pinned.log").read_text()
        # And on to R2 with the next push (here: the local "remote").
        assert (box["remote"] / "pinned" / first.name).is_file()
    else:
        assert kept == [] and "shrank" not in out, out


def k_bytes_equal(a, b):
    return a.read_bytes() == b.read_bytes()


def test_l14_the_pinned_copy_survives_the_rolling_window(box, monkeypatch):
    """KEEP_HOURLY snapshots later, the hourly pre-shrink copy is gone; the
    pinned one is not."""
    code, out = _backup(box, 50)
    first = next((box["local"] / "hourly").glob("radar-*.db.gz"))
    first = first.rename(first.with_name("radar-20000101-0000.db.gz"))
    _backup(box, 0)
    code, out = _run(box, "KEEP_HOURLY=1; KEEP_DAILY=1; do_backup")
    assert code == 0, out
    assert not first.exists()
    assert (box["local"] / "pinned" / first.name).is_file()


def test_l5_restore_refuses_to_run_with_overrides(box):
    """Review L5: restore also stops and restarts the live service, which
    BACKUP_* cannot redirect -- with an override set it does nothing."""
    snap = box["local"] / "x.db.gz"
    snap.write_bytes(b"")
    code, out = _run(box, f'do_restore "{snap}" --yes; echo "returned $?"')
    assert "returned 2" in out and "refuses to run" in out, out
    assert "tmux" not in out and "停后端" not in out
