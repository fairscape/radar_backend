"""Review 17 (a): random walks over a draft, then commit; the live interest must match the draft."""

from __future__ import annotations

import json
import random

import pytest

from tests.test_review_round12_journeys import (  # noqa: F401 -- fixtures
    _feed, _import_researcher, _ok, _one, _q, _request, _scan, _upload, app, env,
)
from tests.test_review_round17_a_journeys import _give_topics, _set_paper_topics


@pytest.mark.parametrize("seed", list(range(1, 16)))
def test_draft_walk(app, env, seed):
    rnd = random.Random(seed)
    rid = _import_researcher(app)
    rpapers = [p["id"] for p in _ok(_request(app, "GET", f"/api/researchers/{rid}"))["papers"]]
    if rnd.random() < 0.5:
        slug = _ok(_request(app, "POST", f"/api/researchers/{rid}/interests",
                            json={"name": f"DW {seed}", "openalex_ids": rpapers[:3]}))["draft"]["slug"]
    else:
        slug = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": f"DW {seed}"}))["slug"]
    _give_topics(env, slug) if _q(env, "SELECT 1 FROM profile_seeds ps JOIN profiles p ON p.id=ps.profile_id WHERE p.slug=?", slug) else None
    log = []
    n_up = 0
    last_topics = None
    for step in range(25):
        op = rnd.choice(["upload", "add", "rm", "get_topics", "put_topics", "rtopics", "dry", "coh", "umls"])
        seeds = [r[0] for r in _q(env, "SELECT ps.openalex_id FROM profile_seeds ps JOIN profiles p ON p.id=ps.profile_id WHERE p.slug=?", slug)]
        if op == "upload":
            n_up += 1
            up = _ok(_upload(app, f"dw-{seed}-{n_up}".encode(), f"d{n_up}.pdf", slug))
            _set_paper_topics(env, up["id"], [f"T4{seed:02d}{n_up:02d}", "T29999"])
        elif op == "add":
            pick = rnd.sample(rpapers, rnd.randint(1, len(rpapers)))
            _ok(_request(app, "POST", f"/api/profiles/draft/{slug}/seeds", json={"openalex_ids": pick}))
        elif op == "rm" and seeds:
            s = rnd.choice(seeds)
            path = rnd.choice(["new", "old"])
            if path == "new":
                _ok(_request(app, "DELETE", f"/api/profiles/{slug}/seeds", params={"openalex_id": s}))
            else:
                from urllib.parse import quote
                _ok(_request(app, "DELETE", f"/api/profiles/draft/{slug}/seeds/{quote(s, safe='')}"))
            op = f"rm {path} {s}"
        elif op == "get_topics":
            last_topics = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))
        elif op == "put_topics":
            ts = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))
            if ts:
                pick = rnd.sample([t["id"] for t in ts], rnd.randint(1, len(ts)))
                r = _ok(_request(app, "PUT", f"/api/profiles/{slug}/topics", json={"selected_topic_ids": pick}))
                assert {t["id"] for t in r if t["on"]} == set(pick), (log, pick, r)
                again = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))
                assert {t["id"] for t in again if t["on"]} == set(pick), (log, "GET after PUT differs", pick, again)
                op = f"put_topics {pick}"
        elif op == "rtopics":
            en = rnd.choice([True, False])
            r = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": en})
            assert r.status_code in (200, 400, 409), r.text
            op = f"rtopics {en} -> {r.status_code}"
        elif op == "dry":
            ts = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))
            pick = rnd.sample([t["id"] for t in ts], rnd.randint(1, len(ts))) if ts else None
            years = rnd.choice([None, (2018, None), (2020, 2024)])
            body = {"days": 30, "selected_topic_ids": pick}
            if years:
                body.update(year_from=years[0], year_to=years[1])
            run = _ok(_request(app, "POST", f"/api/profiles/draft/{slug}/dry-run", json=body))
            st = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/dry-run/{run['run_id']}"))
            assert st["run"]["finished_at"] and not st["run"]["error"], st["run"]
            if pick:
                after = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))
                assert {t["id"] for t in after if t["on"]} == set(pick), (log, "dry-run selection not stored", pick, after)
            op = f"dry {pick} {years}"
        elif op == "coh":
            r = _request(app, "POST", f"/api/profiles/draft/{slug}/coherence")
            assert r.status_code == 200, r.text
            assert r.json()["n"] == len(seeds), (r.json()["n"], len(seeds))
        elif op == "umls":
            _ok(_request(app, "POST", f"/api/profiles/{slug}/umls"))
        else:
            op = f"skip {op}"
        log.append(op)
        drafts = {d["slug"]: d for d in _ok(_request(app, "GET", "/api/profiles/drafts"))}
        seeds_now = _q(env, "SELECT ps.openalex_id FROM profile_seeds ps JOIN profiles p ON p.id=ps.profile_id WHERE p.slug=?", slug)
        assert drafts[slug]["n_seeds"] == len(seeds_now), (log,)
        docs = {d["id"] for d in _ok(_request(app, "GET", "/api/vault/docs", params={"tag": slug}))}
        assert docs == {r[0] for r in seeds_now}, (log, docs, [r[0] for r in seeds_now])
    seeds = sorted(r[0] for r in _q(env, "SELECT ps.openalex_id FROM profile_seeds ps JOIN profiles p ON p.id=ps.profile_id WHERE p.slug=?", slug))
    if not seeds:
        _ok(_upload(app, f"dw-{seed}-final".encode(), "final.pdf", slug))
        seeds = sorted(r[0] for r in _q(env, "SELECT ps.openalex_id FROM profile_seeds ps JOIN profiles p ON p.id=ps.profile_id WHERE p.slug=?", slug))
    ts = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))
    pick = rnd.sample([t["id"] for t in ts], rnd.randint(1, len(ts))) if ts else []
    years = rnd.choice([None, {"year_from": 2019, "year_to": None}])
    prof = _ok(_request(app, "POST", "/api/profiles", json={"slug": slug, "threshold": 0.05, "selected_topic_ids": pick, "years": years}))
    det = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))
    assert sorted(s["id"] for s in det["seeds"]) == seeds
    assert prof["seeds"] == len(seeds), (prof["seeds"], len(seeds))
    assert {t["id"] for t in det["topics"] if t["on"]} == set(pick), (log, pick, det["topics"])
    assert prof["threshold"] == 0.05
    assert (prof["yearFrom"], prof["yearTo"]) == ((2019, None) if years else (None, None))
    row = _one(env, "SELECT * FROM profiles WHERE slug=?", slug)
    assert sorted(json.loads(row["selector_config_json"])["fit_seed_ids"]) == seeds
    prev = _ok(_request(app, "GET", f"/api/profiles/{slug}/topics/preview"))
    assert {t["id"] for t in prev if t["on"]} == set(pick), (log, "preview differs", prev)
    _scan(app, slug)
    runs = _ok(_request(app, "GET", f"/api/profiles/{slug}/runs"))
    assert runs[0]["error"] is None
    print("LOG", log)
