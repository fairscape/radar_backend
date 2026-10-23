"""0025_bars_on_grid -- stored thresholds and seed-band floors on the 3-place grid.

Scores are shown rounded down to 3 places and compared raw with the bar
(``calibration``, BAR_PLACES). With a bar of more places the two disagreed:
a paper shown at the band's floor was not "Like your seeds". New writes go
through ``calibration.bar`` / ``bar_floor``; this snaps the rows already
stored the same way.
"""

from __future__ import annotations

import math
import sqlite3


def migrate(conn: sqlite3.Connection) -> None:
    rows = conn.execute("SELECT id, threshold, seed_sim_min FROM profiles").fetchall()
    for pid, thr, band_min in rows:
        conn.execute(
            "UPDATE profiles SET threshold = ?, seed_sim_min = ? WHERE id = ?",
            (None if thr is None else round(float(thr), 3),
             None if band_min is None else math.floor(float(band_min) * 1000 + 1e-9) / 1000,
             pid),
        )
