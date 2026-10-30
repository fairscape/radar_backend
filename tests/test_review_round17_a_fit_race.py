"""Review 17 (a): a seed attached while a refit is fitting is recorded as fitted but is not in the centroid.

Spec E: for a saved interest the race is now a copy saved during a scan's refit (refused).
"""

from __future__ import annotations

import json

from rag_lib.api.services import wizard as wizard_service
from rag_lib.db import connect
from rag_lib.db.repos import profiles as profiles_repo
from tests.edit_helpers import edit_saved, refit
from tests.test_review_round12_journeys import (  # noqa: F401 -- fixtures
    _feed, _go_live, _ok, _one, _q, _request, _scan, app, env,
)


def _state(env, slug):
    row = _one(env, "SELECT * FROM profiles WHERE slug=?", slug)
    seeds = sorted(r[0] for r in _q(env, "SELECT openalex_id FROM profile_seeds WHERE profile_id=?", row["id"]))
    cfg = json.loads(row["selector_config_json"])
    return row, seeds, cfg


def test_seed_saved_during_a_scans_refit_is_refused(app, env, monkeypatch):
    """Spec E: a saved interest is refitted only by a scan (at its start,
    under the scan lock), and its seeds change only when an Edit-interest
    copy is saved. The old race -- a seed attached while the refit fits,
    recorded as fitted but not in the centroid -- becomes: a copy saved
    while the scan's refit fits. The save is refused (E.6 R11) and writes
    nothing; saved after the scan, the next scan fits the seed."""
    from rag_lib.api.services import edit_copy

    slug, rid = _go_live(app, env, name="Fit Race", threshold=0.0)
    papers = [p["id"] for p in _ok(_request(app, "GET", f"/api/researchers/{rid}"))["papers"]]
    late = papers[0]
    # Remove ``late`` and save: the fit is stale until the next scan refits.
    with edit_saved(lambda m, p, **k: _request(app, m, p, **k), slug) as copy:
        _ok(_request(app, "DELETE", f"/api/profiles/{copy}/seeds", params={"openalex_id": late}))
    row, seeds, cfg = _state(env, slug)
    assert late not in seeds and late in cfg["fit_seed_ids"]

    real = wizard_service._new_selector_for_draft
    fired = []

    def _selector_then_other_request_saves(prow, profile, **kw):
        sel = real(prow, profile, **kw)
        if not fired and int(prow["id"]) == row["id"]:
            # Another request (a second tab) opens Edit interest, adds the
            # seed back and saves while this scan's refit is fitting.
            other = connect(env)
            try:
                c = edit_copy.open_copy(other, user_id=1, slug=slug)
                profiles_repo.attach_seed(other, int(c["id"]), late)
                try:
                    edit_copy.save_copy(other, user_id=1, slug=c["slug"])
                except wizard_service.ScanRunning:
                    fired.append(c["slug"])
                else:
                    fired.append(None)
            finally:
                other.close()
        return sel

    monkeypatch.setattr(wizard_service, "_new_selector_for_draft", _selector_then_other_request_saves)
    _scan(app, slug)
    monkeypatch.setattr(wizard_service, "_new_selector_for_draft", real)
    assert fired and fired[0] is not None, "the save under the scan's refit was not refused"
    row, seeds, cfg = _state(env, slug)
    assert late not in seeds                                      # nothing written
    assert sorted(cfg["fit_seed_ids"]) == seeds                   # the fit is what it says
    # Saved once the scan is over; the next scan fits it.
    _ok(_request(app, "POST", f"/api/profiles/draft/{fired[0]}/save-edit"))
    row, seeds, cfg = _state(env, slug)
    assert late in seeds and late not in cfg["fit_seed_ids"]
    _scan(app, slug)
    row2, seeds2, cfg2 = _state(env, slug)
    assert sorted(cfg2["fit_seed_ids"]) == seeds2
    # A refit by hand shows what the centroid should be.
    conn = connect(env)
    try:
        refit(conn, 1, slug)
    finally:
        conn.close()
    row3, _, _ = _state(env, slug)
    assert row2["centroid"] == row3["centroid"], "the scan kept a centroid without the saved seed"


def test_seed_attached_during_commit_is_never_fitted(app, env, monkeypatch):
    from tests.test_review_round12_journeys import _import_researcher
    rid = _import_researcher(app)
    papers = [p["id"] for p in _ok(_request(app, "GET", f"/api/researchers/{rid}"))["papers"]]
    late = papers[0]
    slug = _ok(_request(app, "POST", f"/api/researchers/{rid}/interests",
                        json={"name": "Commit Race", "openalex_ids": papers[1:]}))["draft"]["slug"]
    on = [t["id"] for t in _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics")) if t["on"]]
    real = wizard_service._new_selector_for_draft
    fired = []

    def _hook(prow, profile, **kw):
        sel = real(prow, profile, **kw)
        if not fired and kw.get("threshold") is not None:
            fired.append(1)
            other = connect(env)
            try:
                profiles_repo.attach_seed(other, int(prow["id"]), late)
            finally:
                other.close()
        return sel

    monkeypatch.setattr(wizard_service, "_new_selector_for_draft", _hook)
    prof = _ok(_request(app, "POST", "/api/profiles", json={"slug": slug, "threshold": 0.0, "selected_topic_ids": on}))
    monkeypatch.setattr(wizard_service, "_new_selector_for_draft", real)
    row, seeds, cfg = _state(env, slug)
    det = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))
    print("COMMIT FIT", len(cfg["fit_seed_ids"]), "seeds", len(seeds), "matrix", len(cfg.get("seed_matrix") or []),
          "Profile.seeds", prof["seeds"], "detail seeds", len(det["seeds"]))
    _scan(app, slug)
    row2, _, _ = _state(env, slug)
    # A refit by hand (the route is refused for a saved interest, spec E).
    conn = connect(env)
    try:
        refit(conn, 1, slug)
    finally:
        conn.close()
    row3, _, _ = _state(env, slug)
    assert row2["centroid"] == row3["centroid"], "scan kept a centroid without the late seed"
