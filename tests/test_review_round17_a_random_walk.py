"""Review 17 (a): random walks over a live interest, invariants checked after each step."""

from __future__ import annotations

import json
import random

import pytest

from tests.test_review_round12_journeys import (  # noqa: F401 -- fixtures
    _feed, _go_live, _ok, _one, _q, _request, _scan, _upload, app, env,
)
from tests.test_review_round17_a_journeys import _give_topics


def _check(app, env, slug, log):
    prof = _ok(_request(app, "GET", f"/api/profiles/{slug}"))
    det = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))
    lst = {p["key"]: p for p in _ok(_request(app, "GET", "/api/profiles"))}[slug]
    seeds = {s["id"] for s in det["seeds"]}
    assert prof["seeds"] == len(seeds) == lst["seeds"], (log[-1], prof["seeds"], len(seeds), lst["seeds"])
    row = _one(env, "SELECT * FROM profiles WHERE slug=?", slug)
    cfg = json.loads(row["selector_config_json"])
    assert sorted(cfg["fit_seed_ids"]) == sorted(seeds), (log[-1], "fit seeds stale")
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
    for step in range(40):
        feed, fresh, triaged, seeds = _check(app, env, slug, log)
        op = rnd.choice(["scan", "backfill", "thr", "years", "rm_seed", "add_seed", "upload",
                         "save", "dismiss", "undo", "topics", "rtopics", "refit", "feed_to_seed"])
        if op == "scan":
            _scan(app, slug)
        elif op == "backfill":
            y0 = rnd.choice([2016, 2019, 2021])
            _ok(_request(app, "POST", f"/api/profiles/{slug}/backfill", json={"year_from": y0, "year_to": None}))
        elif op == "thr":
            hist = _ok(_request(app, "POST", f"/api/profiles/{slug}/dry-run"))["scores"]
            t = max(0.0, round(rnd.choice(hist), 3)) if hist else 0.0
            op = f"thr {t}"
            _ok(_request(app, "PATCH", f"/api/profiles/{slug}", json={"threshold": t}))
        elif op == "years":
            yrs = rnd.choice([None, (2018, None), (None, 2023), (2020, 2024), (2015, 2019)])
            op = f"years {yrs}"
            body = {"year_from": None, "year_to": None} if yrs is None else {"year_from": yrs[0], "year_to": yrs[1]}
            _ok(_request(app, "PATCH", f"/api/profiles/{slug}", json={"years": body}))
        elif op == "rm_seed" and len(seeds) > 1:
            s = rnd.choice(sorted(seeds))
            op = f"rm_seed {s}"
            _ok(_request(app, "DELETE", f"/api/profiles/{slug}/seeds", params={"openalex_id": s}))
        elif op == "add_seed":
            s = rnd.choice(rpapers)
            op = f"add_seed {s}"
            _ok(_request(app, "POST", f"/api/profiles/{slug}/seeds", json={"openalex_ids": [s]}))
        elif op == "upload":
            n_up += 1
            _ok(_upload(app, f"walk-{seed}-{n_up}".encode(), f"w{n_up}.pdf", slug))
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
            prev = _ok(_request(app, "GET", f"/api/profiles/{slug}/topics/preview"))
            ids = [t["id"] for t in prev]
            pick = rnd.sample(ids, max(1, rnd.randint(1, len(ids)))) if ids else []
            op = f"topics {pick}"
            _ok(_request(app, "PUT", f"/api/profiles/{slug}/topics", json={"selected_topic_ids": pick}))
        elif op == "rtopics":
            en = rnd.choice([True, False])
            op = f"rtopics {en}"
            r = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": en})
            assert r.status_code in (200, 409), r.text
        elif op == "refit":
            _ok(_request(app, "POST", f"/api/profiles/{slug}/refit"))
        elif op == "feed_to_seed" and fresh:
            c = rnd.choice(fresh)
            op = f"feed_to_seed {c['id']}"
            _ok(_request(app, "POST", f"/api/profiles/{slug}/seeds", json={"openalex_ids": [c["id"]]}))
        else:
            op = f"skip {op}"
        log.append(op)
    _check(app, env, slug, log)
    print("LOG", log)
