"""Review 18 (b): the wizard trial's distribution rounds to nearest, the feed floors.

The wizard's slider counts ``scores.filter(s => s >= value)`` (WizardPage.tsx,
ui/domain.tsx) and its paper list ``p.score >= value``. The trial sends
``round(raw, 4)``; the feed admits ``score_raw >= threshold`` on the raw
value, and the Threshold tab of a live interest sends ``floor4``. So a paper
at 0.89996 counts as passing a 0.9 bar in the wizard, the sweep row for 0.9
(compared on the raw value) leaves it out, and after commit the feed does too.
"""

from __future__ import annotations

import types

from rag_lib.api.mappers import floor4
from rag_lib.api.services import wizard
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import users as users_repo
from rag_lib.paper import Paper
from rag_lib.scoring.pool import passes


class _Selector:
    def fit(self, profile):
        pass

    def select(self, papers, profile, threshold=None):
        raws = {"https://openalex.org/W1": 0.89996, "https://openalex.org/W2": 0.95}
        out = [(raws[p.openalex_id], p, {"score_raw": raws[p.openalex_id], "score_pct": 0.0}) for p in papers]
        return sorted(out, key=lambda e: e[0], reverse=True)


class _Gatherer:
    last_tier_used = "stub"
    last_filter_str = ""

    def fetch(self, profile, since, limit=None):
        return [Paper(doi=None, openalex_id="https://openalex.org/W1", title="Just under"),
                Paper(doi=None, openalex_id="https://openalex.org/W2", title="Clear")]

    def cost(self):
        return {"api_calls": 0}


def test_trial_distribution_counts_a_paper_the_feed_will_not_show(tmp_path, monkeypatch):
    conn = connect(tmp_path / "radar.db")
    apply_migrations(conn)
    uid = int(users_repo.upsert(conn, "u@example.com")["id"])
    conn.execute(
        "INSERT INTO profiles (user_id, name, slug, embedding_model, topic_filters_json, n_seed, is_draft, threshold)"
        " VALUES (?, 'D', 'd', 'placeholder-v1', '{}', 0, 1, 0.7)", (uid,))
    conn.commit()
    row = conn.execute("SELECT * FROM profiles WHERE slug = 'd'").fetchone()

    fake_profile = types.SimpleNamespace(
        papers=[object()], topic_filters={}, embedding_model="placeholder-v1",
        seed_embeddings=lambda model: [],
    )
    monkeypatch.setattr(wizard, "_load_profile", lambda conn, row: fake_profile)
    monkeypatch.setattr(wizard, "_new_selector_for_draft", lambda row, profile, **k: _Selector())
    settings = types.SimpleNamespace(RADAR_GATHER_MAX_CANDIDATES=None, RADAR_DEFAULT_MAILTO="x@example.com")

    res = wizard.compute_dry_run_for_profile(
        conn, settings, profile_row=row, slug="d", gatherer=_Gatherer(), thresholds=[0.9])

    thr = 0.9
    sweep_n = next(r.n for r in res.sweep if r.th == thr)
    slider_n = sum(1 for s in res.scores if s >= thr)          # WizardPage.tsx:1200
    list_n = sum(1 for p in res.papers if p.score >= thr)      # ThresholdPaperList.tsx:17
    feed_n = sum(1 for raw in (0.89996, 0.95) if passes(raw, thr))
    assert sweep_n == feed_n == 1
    # Live interest's Threshold tab would show floor4(0.89996) = 0.8999.
    assert floor4(0.89996) < thr
    assert slider_n == feed_n, f"slider says {slider_n} pass at {thr}, the feed shows {feed_n} (scores={res.scores})"
    assert list_n == feed_n, f"paper list marks {list_n} as passing, the feed shows {feed_n}"
    conn.close()


def test_the_trial_takes_what_a_scan_takes(tmp_path, monkeypatch):
    """FUNCTIONAL_SPEC S.1: no 1000-paper cap on the trial -- each topic's
    most cited, as a scan -- unless RADAR_GATHER_MAX_CANDIDATES sets one."""
    conn = connect(tmp_path / "radar.db")
    apply_migrations(conn)
    uid = int(users_repo.upsert(conn, "u@example.com")["id"])
    conn.execute(
        "INSERT INTO profiles (user_id, name, slug, embedding_model, topic_filters_json, n_seed, is_draft, threshold)"
        " VALUES (?, 'D', 'd', 'placeholder-v1', '{}', 0, 1, 0.7)", (uid,))
    conn.commit()
    row = conn.execute("SELECT * FROM profiles WHERE slug = 'd'").fetchone()
    fake_profile = types.SimpleNamespace(
        papers=[object()], topic_filters={}, embedding_model="placeholder-v1",
        seed_embeddings=lambda model: [],
    )
    monkeypatch.setattr(wizard, "_load_profile", lambda conn, row: fake_profile)
    monkeypatch.setattr(wizard, "_new_selector_for_draft", lambda row, profile, **k: _Selector())
    asked = []

    class Recording(_Gatherer):
        def fetch(self, profile, since, limit=None):
            asked.append(limit)
            return super().fetch(profile, since, limit=limit)

    for cap in (None, 150):
        settings = types.SimpleNamespace(RADAR_GATHER_MAX_CANDIDATES=cap, RADAR_DEFAULT_MAILTO="x@example.com")
        wizard.compute_dry_run_for_profile(
            conn, settings, profile_row=row, slug="d", gatherer=Recording(), thresholds=[0.9])
    assert asked == [None, 150]
    conn.close()
