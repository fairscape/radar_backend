"""Class sweep: a score shown beside a bar agrees with the server's decision.

Rounds 8 and 9 found the Threshold tab, the wizard trial, the seed band
and the page's own toFixed each disagreeing with the feed, one at a time.
The rule (``calibration``): bars on the 3-place grid, scores sent rounded
down to 4 places and shown rounded down to 3. Here every bar the server
produces is checked to be on the grid, and the rule itself is checked
against raw ``>=`` over many scores.
"""

from __future__ import annotations

import math
import random

import numpy as np

from rag_lib import calibration
from rag_lib.api.mappers import floor4


def shown(x: float) -> float:
    """``fmtScore`` in lib/format.ts, over what the API sends."""
    return math.floor(floor4(x) * 1000 + 1e-9) / 1000


def on_grid(b: float) -> bool:
    return abs(b * 1000 - round(b * 1000)) < 1e-6


def test_shown_score_at_or_over_shown_bar_iff_the_server_passes_it():
    rng = random.Random(7)
    bars = [round(rng.uniform(0.3, 0.99), 3) for _ in range(300)]
    for b in bars:
        for s in [b, b - 1e-5, b - 4e-5, b + 1e-5, b - 0.0004, b + 0.0009, b - 0.0011] + \
                 [rng.uniform(b - 0.002, b + 0.002) for _ in range(30)]:
            assert (shown(s) >= b) == (s >= b), (s, b)


def test_every_bar_the_server_makes_is_on_the_grid():
    rng = np.random.default_rng(3)
    for _ in range(200):
        vecs = rng.normal(size=(rng.integers(2, 9), 16))
        band = calibration.seed_similarity_band(vecs)
        assert on_grid(band["min"]), band["min"]
        scores = list(rng.uniform(0.5, 0.95, size=40))
        for floor in (band["min"], None, 0.99, 0.1):
            assert on_grid(calibration.suggest_threshold(floor, scores))
    assert on_grid(calibration.bar(0.90049))
    assert calibration.bar(None) is None


def test_the_band_floor_never_rises_over_a_seed():
    rng = np.random.default_rng(5)
    for _ in range(100):
        band = calibration.seed_similarity_band(rng.normal(size=(5, 8)))
        assert all(v >= band["min"] for v in band["values"])


def test_stored_thresholds_are_snapped(tmp_path):
    from rag_lib.db import connect
    from rag_lib.db.migrate import apply_migrations
    from rag_lib.db.repos import profiles as profiles_repo, users as users_repo

    conn = connect(tmp_path / "r.db")
    apply_migrations(conn)
    uid = users_repo.upsert(conn, email="a@example.com")
    uid = uid["id"] if not isinstance(uid, int) else uid
    pid = profiles_repo.upsert(conn, user_id=uid, name="X", embedding_model="m", n_seed=0, threshold=0.90049)
    assert conn.execute("SELECT threshold FROM profiles WHERE id = ?", (pid,)).fetchone()[0] == 0.9
    profiles_repo.update_threshold(conn, pid, 0.81234)
    assert conn.execute("SELECT threshold FROM profiles WHERE id = ?", (pid,)).fetchone()[0] == 0.812


def test_migration_snaps_rows_already_stored(tmp_path):
    import importlib.util
    import pathlib
    import sqlite3

    path = pathlib.Path(calibration.__file__).parent / "db" / "migrations" / "0025_bars_on_grid.py"
    spec = importlib.util.spec_from_file_location("m0025", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE profiles (id INTEGER PRIMARY KEY, threshold REAL, seed_sim_min REAL)")
    conn.execute("INSERT INTO profiles VALUES (1, 0.9, 0.8562069600396984), (2, 0.90049, NULL), (3, NULL, 0.9)")
    mod.migrate(conn)
    assert conn.execute("SELECT threshold, seed_sim_min FROM profiles ORDER BY id").fetchall() == [
        (0.9, 0.856), (0.9, None), (None, 0.9)]
