"""Review 17 (a): random walks over a live interest, invariants checked after each step."""

from __future__ import annotations

import json
import random

import pytest

from tests.test_review_round12_journeys import (  # noqa: F401 -- fixtures
    _feed, _go_live, _ok, _one, _q, _request, _scan, _upload, app, env,
)
from tests.edit_helpers import edit_saved
from tests.test_review_round17_a_journeys import _give_topics


def _edit(app, slug, **kw):
    """Edit interest on ``slug``: a saved interest changes only on a copy,
    saved in one go (FUNCTIONAL_SPEC E)."""
    return edit_saved(lambda m, p, **k: _request(app, m, p, **k), slug, **kw)


def _check(app, env, slug, log, fitted):
    prof = _ok(_request(app, "GET", f"/api/profiles/{slug}"))
    det = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))
    lst = {p["key"]: p for p in _ok(_request(app, "GET", "/api/profiles"))}[slug]
    seeds = {s["id"] for s in det["seeds"]}
    assert prof["seeds"] == len(seeds) == lst["seeds"], (log[-1], prof["seeds"], len(seeds), lst["seeds"])
    row = _one(env, "SELECT * FROM profiles WHERE slug=?", slug)
    cfg = json.loads(row["selector_config_json"])
    # Spec E: saving new seeds does not refit; the next scan does. The fit
    # is the one the last scan (or the wizard) made, whatever was saved since.
    assert sorted(cfg["fit_seed_ids"]) == sorted(fitted), (log[-1], "fit is not the last scan's")
    assert prof["threshold"] == pytest.approx(row["threshold"])
    feed = _feed(app, slug)
    thr = row["threshold"]
    yf, yt = row["year_from"], row["year_to"]
    hist = _ok(_request(app, "POST", f"/api/profiles/{slug}/dry-run"))
    passing = [s for s in hist["scores"] if s >= thr]
    fresh = [c for c in feed["cards"] if c.get("state") is None]
    triaged = [c for c in feed["cards"] if c.get("state") is not None]
    assert len(feed["cards"]) == len(passing), (log[-1], len(feed["cards"]), len(passing))
    for c in feed["cards"]:
        assert c["id"] not in seeds, (log[-1], "seed in feed", c["id"])
        assert c["similarity"] is None or c["similarity"] >= round(thr, 4) - 1e-4, (log[-1], c["similarity"], thr)
        y = int(c["date"][:4]) if c["date"] else None
        assert y is None or ((yf is None or y >= yf) and (yt is None or y <= yt)), (log[-1], y, yf, yt)
    low = _feed(app, slug, bucket="low")["cards"]
    assert len(low) == len(hist["scores"]) - len(passing), (log[-1], len(low), len(hist["scores"]), len(passing))
    # Saved state agrees between the feed and the counts.
    # Over the papers the feed lists: not seeds, inside the year range.
    from rag_lib.db.repos.candidates import IN_YEARS, NOT_SEED
    n_saved_db = _one(env, "SELECT COUNT(*) n FROM profile_candidates pc JOIN profiles pr ON pr.id = pc.profile_id"
                      " JOIN papers p ON p.openalex_id = pc.openalex_id"
                      f" WHERE pc.profile_id=? AND pc.saved_at IS NOT NULL AND {NOT_SEED} AND {IN_YEARS}", row["id"])["n"]
    assert prof["saves30"] == n_saved_db
    thr_view = _ok(_request(app, "GET", f"/api/profiles/{slug}/feedback-threshold"))
    assert thr_view["threshold"] == pytest.approx(thr)
    # Topics: detail = stored
    stored = json.loads(row["topic_filters_json"] or "{}")
    assert {t["id"]: bool(t.get("on", True)) for t in stored.get("topics") or []} == {t["id"]: t["on"] for t in det["topics"]}
    # No open runs left behind.
    open_runs = _q(env, "SELECT id, tier_used FROM gather_runs WHERE finished_at IS NULL")
    assert not open_runs, (log[-1], [tuple(r) for r in open_runs])
    return feed, fresh, triaged, seeds


class _VaryingGatherer:
    """Each fetch: some papers seen before, some new, with years."""
    name = "varying"
    calls = [0]

    def __init__(self, rnd, **_):
        self.rnd = rnd
        self.last_tier_used = None

    def _papers(self, n):
        from rag_lib.paper import Paper
        self.calls[0] += 1
        out = []
        for i in range(n):
            k = self.rnd.randint(0, 40)
            out.append(Paper(doi=f"10.walk/{k}", openalex_id=f"https://openalex.org/W9{k:04d}",
                             title=f"Walk paper {k}", abstract=f"walk abstract {k} " * (1 + k % 5),
                             year=2014 + (k % 13), source="fixture"))
        return out

    def fetch(self, profile, since, *, limit=None):
        return self._papers(min(12, limit or 12))

    def fetch_years(self, profile, year_from, year_to, *, per_year):
        return [p for p in self._papers(20) if year_from <= p.year <= year_to]

    def cost(self):
        return {"api_calls": 1}


@pytest.mark.parametrize("seed", list(range(1, 13)))
def test_random_walk(app, env, seed, monkeypatch):
    from rag_lib.scheduler import jobs as scheduler_jobs
    rnd = random.Random(seed)
    grnd = random.Random(seed * 1000)
    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer", lambda **kw: _VaryingGatherer(grnd))
    slug, rid = _go_live(app, env, name=f"Walk {seed}", threshold=0.0)
    _give_topics(env, slug)
    rpapers = [p["id"] for p in _ok(_request(app, "GET", f"/api/researchers/{rid}"))["papers"]]
    log = ["start"]
    _scan(app, slug)
    log.append("scan")
    n_up = 0

    def _seed_ids():
        return {s["id"] for s in _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))["seeds"]}

    fitted = _seed_ids()            # the wizard's commit fitted these
    for step in range(40):
        feed, fresh, triaged, seeds = _check(app, env, slug, log, fitted)
        op = rnd.choice(["scan", "backfill", "thr", "years", "rm_seed", "add_seed", "upload",
                         "save", "dismiss", "undo", "topics", "rtopics", "refit", "feed_to_seed"])
        if op == "scan":
            _scan(app, slug)
            fitted = _seed_ids()    # a scan fits on the seeds saved since (spec E)
        elif op == "backfill":
            y0 = rnd.choice([2016, 2019, 2021])
            _ok(_request(app, "POST", f"/api/profiles/{slug}/backfill", json={"year_from": y0, "year_to": None}))
            fitted = _seed_ids()    # a backfill is a scan: it refits first
        elif op == "thr":
            hist = _ok(_request(app, "POST", f"/api/profiles/{slug}/dry-run"))["scores"]
            t = max(0.0, round(rnd.choice(hist), 3)) if hist else 0.0
            op = f"thr {t}"
            with _edit(app, slug) as copy:
                _ok(_request(app, "PATCH", f"/api/profiles/{copy}", json={"threshold": t}))
        elif op == "years":
            yrs = rnd.choice([None, (2018, None), (None, 2023), (2020, 2024), (2015, 2019)])
            op = f"years {yrs}"
            body = {"year_from": None, "year_to": None} if yrs is None else {"year_from": yrs[0], "year_to": yrs[1]}
            with _edit(app, slug) as copy:
                _ok(_request(app, "PATCH", f"/api/profiles/{copy}", json={"years": body}))
        elif op == "rm_seed" and len(seeds) > 1:
            s = rnd.choice(sorted(seeds))
            keep = rnd.random() < 0.8      # sometimes left without saving
            op = f"rm_seed {s}" + ("" if keep else " (discarded)")
            with _edit(app, slug, save=keep) as copy:
                _ok(_request(app, "DELETE", f"/api/profiles/{copy}/seeds", params={"openalex_id": s}))
            assert (s in _seed_ids()) is not keep, (op, "seed removal saved/discarded wrongly")
        elif op == "add_seed":
            s = rnd.choice(rpapers)
            op = f"add_seed {s}"
            with _edit(app, slug) as copy:
                _ok(_request(app, "POST", f"/api/profiles/{copy}/seeds", json={"openalex_ids": [s]}))
        elif op == "upload":
            n_up += 1
            with _edit(app, slug) as copy:
                _ok(_upload(app, f"walk-{seed}-{n_up}".encode(), f"w{n_up}.pdf", copy))
        elif op in ("save", "dismiss") and feed["cards"]:
            c = rnd.choice(feed["cards"])
            op = f"{op} {c['id']}"
            _ok(_request(app, "POST", f"/api/radar/cards/{op.split()[0]}", json={"card_id": c["id"], "profile": slug}))
        elif op == "undo" and triaged:
            c = rnd.choice(triaged)
            which = "save" if c["state"] == "saved" else "dismiss"
            op = f"undo {which} {c['id']}"
            r = _ok(_request(app, "POST", f"/api/radar/cards/{which}", json={"card_id": c["id"], "profile": slug, "on": False}))
            assert r["state"] is None
        elif op == "topics":
            with _edit(app, slug) as copy:
                prev = _ok(_request(app, "GET", f"/api/profiles/{copy}/topics/preview"))
                ids = [t["id"] for t in prev]
                pick = rnd.sample(ids, max(1, rnd.randint(1, len(ids)))) if ids else []
                op = f"topics {pick}"
                _ok(_request(app, "PUT", f"/api/profiles/{copy}/topics", json={"selected_topic_ids": pick}))
        elif op == "rtopics":
            en = rnd.choice([True, False])
            op = f"rtopics {en}"
            with _edit(app, slug) as copy:
                r = _request(app, "PUT", f"/api/profiles/{copy}/researcher-topics", json={"enabled": en})
                assert r.status_code in (200, 409), r.text
        elif op == "refit":
            # Spec E: a saved interest is not refitted on request; the next
            # scan refits it. The route refuses and changes nothing.
            r = _request(app, "POST", f"/api/profiles/{slug}/refit")
            assert r.status_code == 409, r.text
        elif op == "feed_to_seed" and fresh:
            c = rnd.choice(fresh)
            op = f"feed_to_seed {c['id']}"
            # Spec E (R1): the feed's "add as seed" is gone; the route refuses,
            # and the paper becomes a seed through Edit interest.
            r = _request(app, "POST", f"/api/profiles/{slug}/seeds", json={"openalex_ids": [c["id"]]})
            assert r.status_code == 409, r.text
            with _edit(app, slug) as copy:
                _ok(_request(app, "POST", f"/api/profiles/{copy}/seeds", json={"openalex_ids": [c["id"]]}))
        else:
            op = f"skip {op}"
        log.append(op)
    _check(app, env, slug, log, fitted)
    # A last scan fits on the seeds saved since the one before.
    _scan(app, slug)
    log.append("final scan")
    _check(app, env, slug, log, _seed_ids())
    print("LOG", log)
