"""Review 17 (a): a seed attached while a refit is fitting is recorded as fitted but is not in the centroid."""

from __future__ import annotations

import json

from rag_lib.api.services import wizard as wizard_service
from rag_lib.db import connect
from rag_lib.db.repos import profiles as profiles_repo
from tests.test_review_round12_journeys import (  # noqa: F401 -- fixtures
    _feed, _go_live, _ok, _one, _q, _request, _scan, app, env,
)


def _state(env, slug):
    row = _one(env, "SELECT * FROM profiles WHERE slug=?", slug)
    seeds = sorted(r[0] for r in _q(env, "SELECT openalex_id FROM profile_seeds WHERE profile_id=?", row["id"]))
    cfg = json.loads(row["selector_config_json"])
    return row, seeds, cfg


def test_seed_attached_during_refit_is_never_fitted(app, env, monkeypatch):
    slug, rid = _go_live(app, env, name="Fit Race", threshold=0.0)
    papers = [p["id"] for p in _ok(_request(app, "GET", f"/api/researchers/{rid}"))["papers"]]
    late = papers[0]
    _ok(_request(app, "DELETE", f"/api/profiles/{slug}/seeds", params={"openalex_id": late}))
    _scan(app, slug)
    row, seeds, cfg = _state(env, slug)
    assert late not in seeds and sorted(cfg["fit_seed_ids"]) == seeds

    real = wizard_service._new_selector_for_draft
    fired = []

    def _selector_then_other_request_attaches(prow, profile, **kw):
        sel = real(prow, profile, **kw)
        if not fired:
            fired.append(1)
            # Another request (an upload to the Seeds tab, "add as seed" in a
            # second tab) attaches a seed while this refit is fitting.
            other = connect(env)
            try:
                profiles_repo.attach_seed(other, int(prow["id"]), late)
            finally:
                other.close()
        return sel

    monkeypatch.setattr(wizard_service, "_new_selector_for_draft", _selector_then_other_request_attaches)
    _ok(_request(app, "POST", f"/api/profiles/{slug}/refit"))
    monkeypatch.setattr(wizard_service, "_new_selector_for_draft", real)

    row, seeds, cfg = _state(env, slug)
    assert late in seeds
    fitted_matrix = cfg.get("seed_matrix")
    print("FIT_SEEDS", cfg["fit_seed_ids"], "n seeds", len(seeds), "n_seed col", row["n_seed"],
          "matrix rows", len(fitted_matrix) if isinstance(fitted_matrix, list) else None)
    stale_centroid = row["centroid"]
    # The next scan should notice that ``late`` is not in the fit. It does not.
    _scan(app, slug)
    row2, _, _ = _state(env, slug)
    after_scan_centroid = row2["centroid"]
    # A refit by hand shows what the centroid should have been.
    _ok(_request(app, "POST", f"/api/profiles/{slug}/refit"))
    row3, _, _ = _state(env, slug)
    print("CENTROID changed by scan:", after_scan_centroid != stale_centroid,
          "changed by manual refit:", row3["centroid"] != after_scan_centroid,
          "n_seed col", row2["n_seed"], "->", row3["n_seed"])
    assert after_scan_centroid == row3["centroid"], (
        "the stored fit says it includes the late seed, the centroid does not, and the scan kept it")


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
    _ok(_request(app, "POST", f"/api/profiles/{slug}/refit"))
    row3, _, _ = _state(env, slug)
    assert row2["centroid"] == row3["centroid"], "scan kept a centroid without the late seed"
