"""Review 17 (a): journeys with state carried between steps."""

from __future__ import annotations

from tests.test_review_round12_journeys import (  # noqa: F401 -- fixtures
    B, _feed, _go_live, _import_researcher, _ok, _one, _q, _request, _scan, _upload, app, env,
)


def _pid(env, slug):
    return _one(env, "SELECT id FROM profiles WHERE slug = ?", slug)["id"]


def _draft_with_uploads(app, name="Upl"):
    slug = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": name}))["slug"]
    _ok(_upload(app, b"seed-one-" + name.encode(), "one.pdf", slug))
    _ok(_upload(app, b"seed-two-" + name.encode(), "two.pdf", slug))
    return slug


def _give_topics(env, slug, n=4):
    """Each seed carries topics T2000k..: k = its index mod n, plus T29999."""
    import json
    from rag_lib.db import connect
    conn = connect(env)
    try:
        pid = conn.execute("SELECT id FROM profiles WHERE slug=?", (slug,)).fetchone()[0]
        seeds = [r[0] for r in conn.execute("SELECT openalex_id FROM profile_seeds WHERE profile_id=? ORDER BY openalex_id", (pid,))]
        for i, s in enumerate(seeds):
            def blk(tid, name):
                return {"id": tid, "display_name": name, "score": 0.9,
                        "subfield": {"id": "SF1", "display_name": "Sub"}, "field": {"id": "F1", "display_name": "F"},
                        "domain": {"id": "D1", "display_name": "D"}}
            k = i % n
            t = [blk(f"T2000{k}", f"Topic {k}"), blk("T29999", "Shared")]
            conn.execute("UPDATE papers SET topics_json=? WHERE openalex_id=?",
                         (json.dumps({"primary_topic": t[0], "topics": t}), s))
        conn.commit()
    finally:
        conn.close()


def _draft_from_researcher(app, name="R", env=None):
    rid = _import_researcher(app)
    papers = _ok(_request(app, "GET", f"/api/researchers/{rid}"))["papers"]
    ids = [p["id"] for p in papers]
    slug = _ok(_request(app, "POST", f"/api/researchers/{rid}/interests",
                        json={"name": name, "openalex_ids": ids}))["draft"]["slug"]
    if env is not None:
        _give_topics(env, slug)
    return slug, rid, ids


def test_topics_saved_then_dryrun_other_then_commit_third(app, env):
    slug, rid, ids = _draft_from_researcher(app, "Topic Dance", env)
    topics = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))
    tids = [t["id"] for t in topics]
    print("TOPICS", [(t["id"], t["on"]) for t in topics])
    assert len(tids) >= 3, tids
    a, b, c = tids[0], tids[1], tids[2]
    # Save {a} on the draft.
    saved = _ok(_request(app, "PUT", f"/api/profiles/{slug}/topics", json={"selected_topic_ids": [a]}))
    assert {t["id"] for t in saved if t["on"]} == {a}
    again = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))
    assert {t["id"] for t in again if t["on"]} == {a}
    # Dry-run with {b}.
    run = _ok(_request(app, "POST", f"/api/profiles/draft/{slug}/dry-run",
                       json={"days": 30, "selected_topic_ids": [b], "year_from": 2018, "year_to": 2022}))
    st = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/dry-run/{run['run_id']}"))
    assert st["run"]["finished_at"] and not st["run"]["error"], st["run"]
    after = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))
    assert {t["id"] for t in after if t["on"]} == {b}
    # Commit with {b, c}.
    prof = _ok(_request(app, "POST", "/api/profiles", json={
        "slug": slug, "threshold": 0.1, "selected_topic_ids": [b, c],
        "cron": "0 4 * * *", "tz": "UTC", "years": {"year_from": 2018, "year_to": 2022}}))
    assert prof["yearFrom"] == 2018 and prof["yearTo"] == 2022 and prof["threshold"] == 0.1
    detail = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))
    assert {t["id"] for t in detail["topics"] if t["on"]} == {b, c}, detail["topics"]
    assert detail["profile"]["seeds"] == len(detail["seeds"]) == len(ids)
    # Edit-page preview agrees.
    prev = _ok(_request(app, "GET", f"/api/profiles/{slug}/topics/preview"))
    assert {t["id"] for t in prev if t["on"]} == {b, c}
    assert not any(t["new"] for t in prev)
    # Scan, feed only within years.
    _scan(app, slug)
    feed = _feed(app, slug)
    for card in feed["cards"]:
        y = int(card["date"][:4]) if card["date"] else None
        assert y is None or 2018 <= y <= 2022, card["date"]
    runs = _ok(_request(app, "GET", f"/api/profiles/{slug}/runs"))
    print("RUNS", [(r["tier_used"], r["n_fetched"], r["n_new"]) for r in runs])
    # Years cleared on edit.
    p = _ok(_request(app, "PATCH", f"/api/profiles/{slug}", json={"years": {"year_from": None, "year_to": None}}))
    assert p["yearFrom"] is None and p["yearTo"] is None
    feed2 = _feed(app, slug)
    assert len(feed2["cards"]) >= len(feed["cards"])


def test_seeds_change_after_dryrun_then_commit(app, env):
    slug, rid, ids = _draft_from_researcher(app, "Seed Shift", env)
    topics = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))
    on = [t["id"] for t in topics if t["on"]]
    run = _ok(_request(app, "POST", f"/api/profiles/draft/{slug}/dry-run", json={"days": 30, "selected_topic_ids": on}))
    _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/dry-run/{run['run_id']}"))
    # Remove a seed and add an upload after the trial.
    _ok(_request(app, "DELETE", f"/api/profiles/{slug}/seeds", params={"openalex_id": ids[0]}))
    up = _ok(_upload(app, b"late-seed", "late.pdf", slug))
    prof = _ok(_request(app, "POST", "/api/profiles", json={
        "slug": slug, "threshold": 0.2, "selected_topic_ids": on, "years": None}))
    detail = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))
    seed_ids = {s["id"] for s in detail["seeds"]}
    assert ids[0] not in seed_ids and up["id"] in seed_ids
    assert prof["seeds"] == len(seed_ids) == len(ids)
    pid = _pid(env, slug)
    import json
    cfg = json.loads(_one(env, "SELECT selector_config_json FROM profiles WHERE id=?", pid)[0])
    assert sorted(cfg["fit_seed_ids"]) == sorted(seed_ids)
    drafts = _ok(_request(app, "GET", "/api/profiles/drafts"))
    assert slug not in [d["slug"] for d in drafts]
    lst = _ok(_request(app, "GET", "/api/profiles"))
    assert [p["key"] for p in lst] == [slug]
    assert lst[0]["seeds"] == prof["seeds"]


def test_restart_between_steps(app, env):
    from rag_lib.api.app import create_app
    slug = _draft_with_uploads(app, "Restart")
    topics = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))
    # "restart": new app, same DB
    app2 = create_app()
    app2.dependency_overrides.update(app.dependency_overrides)
    drafts = _ok(_request(app2, "GET", "/api/profiles/drafts"))
    assert [d["slug"] for d in drafts] == [slug] and drafts[0]["n_seeds"] == 2
    t2 = _ok(_request(app2, "GET", f"/api/profiles/draft/{slug}/topics"))
    assert [(t["id"], t["on"]) for t in t2] == [(t["id"], t["on"]) for t in topics]
    on = [t["id"] for t in t2 if t["on"]]
    _ok(_request(app2, "POST", "/api/profiles", json={"slug": slug, "threshold": 0.0, "selected_topic_ids": on}))
    app3 = create_app()
    app3.dependency_overrides.update(app.dependency_overrides)
    _scan(app3, slug)
    feed = _feed(app3, slug)
    assert feed["cards"]
    cid = feed["cards"][0]["id"]
    _ok(_request(app3, "POST", "/api/radar/cards/save", json={"card_id": cid, "profile": slug}))
    app4 = create_app()
    app4.dependency_overrides.update(app.dependency_overrides)
    f2 = _feed(app4, slug)
    st = {c["id"]: c.get("state") for c in f2["cards"]}
    print("STATES", f2["states"], st.get(cid))
    p = _ok(_request(app4, "GET", f"/api/profiles/{slug}"))
    assert p["saves30"] == 1


def test_save_dismiss_undo_counts_agree(app, env):
    slug, _ = _go_live(app, env, name="Triage", threshold=0.0)
    _scan(app, slug)
    feed = _feed(app, slug)
    cards = feed["cards"]
    assert len(cards) >= 3
    c1, c2, c3 = cards[0]["id"], cards[1]["id"], cards[2]["id"]
    r = _ok(_request(app, "POST", "/api/radar/cards/save", json={"card_id": c1, "profile": slug}))
    assert r["state"] == "saved"
    r = _ok(_request(app, "POST", "/api/radar/cards/dismiss", json={"card_id": c2, "profile": slug}))
    assert r["state"] == "dismissed"
    # dismiss a saved card
    r = _ok(_request(app, "POST", "/api/radar/cards/dismiss", json={"card_id": c1, "profile": slug}))
    print("DISMISS SAVED", r)
    # undo the dismiss of c2
    r = _ok(_request(app, "POST", "/api/radar/cards/dismiss", json={"card_id": c2, "profile": slug, "on": False}))
    assert r["state"] is None
    # save then unsave c3
    _ok(_request(app, "POST", "/api/radar/cards/save", json={"card_id": c3, "profile": slug}))
    r = _ok(_request(app, "POST", "/api/radar/cards/save", json={"card_id": c3, "profile": slug, "on": False}))
    assert r["state"] is None
    p = _ok(_request(app, "GET", f"/api/profiles/{slug}"))
    fb = _ok(_request(app, "GET", f"/api/profiles/{slug}/feedback"))
    thr = _ok(_request(app, "GET", f"/api/profiles/{slug}/feedback-threshold"))
    f2 = _feed(app, slug)
    print("PROFILE", p["saves30"], p["dismisses30"], "FB", [(e.get("action"), e.get("openalex_id") or e.get("id")) for e in fb], "THR", thr, "STATES", f2["states"])
    states = f2["states"]
    n_saved = sum(1 for v in states.values() if v == "saved")
    n_dis = sum(1 for v in states.values() if v == "dismissed")
    assert p["saves30"] == n_saved == thr["n_saved"]
    assert p["dismisses30"] == n_dis == thr["n_dismissed"]


def test_threshold_edit_feed_matches_histogram(app, env):
    slug, _ = _go_live(app, env, name="Thr", threshold=0.0)
    _scan(app, slug)
    hist = _ok(_request(app, "POST", f"/api/profiles/{slug}/dry-run"))
    scores = sorted(hist["scores"])
    print("HIST", hist["n"], scores, "papers", len(hist["papers"]))
    feed0 = _feed(app, slug)
    assert len(feed0["cards"]) == sum(1 for x in scores if x >= 0.0)
    thr = scores[len(scores) // 2]
    p = _ok(_request(app, "PATCH", f"/api/profiles/{slug}", json={"threshold": thr}))
    assert p["threshold"] == thr
    feed1 = _feed(app, slug)
    expect = sum(1 for s in scores if s >= thr)
    print("THR", thr, "expect", expect, "feed", len(feed1["cards"]), [c["similarity"] for c in feed1["cards"]])
    assert len(feed1["cards"]) == expect
    hist2 = _ok(_request(app, "POST", f"/api/profiles/{slug}/dry-run"))
    assert sorted(hist2["scores"]) == scores
    # buckets add up
    hi = _feed(app, slug, bucket="high")["cards"]
    med = _feed(app, slug, bucket="medium")["cards"]
    low = _feed(app, slug, bucket="low")["cards"]
    print("BUCKETS", len(hi), len(med), len(low))
    assert len(hi) + len(med) == len(feed1["cards"])
    assert len(low) == len(scores) - expect
    # Rescan: runs, candidatesScored agree
    r = _scan(app, slug)
    runs = _ok(_request(app, "GET", f"/api/profiles/{slug}/runs"))
    print("SCAN", r, [(x["tier_used"], x["n_fetched"], x["n_new"], x["error"]) for x in runs])
    feed2 = _feed(app, slug)
    assert feed2["candidatesScored"] == runs[0]["n_fetched"]


def test_live_seed_edit_rescores_feed(app, env):
    slug, rid = _go_live(app, env, name="Seeds Live", threshold=0.0)
    _scan(app, slug)
    f0 = {c["id"]: c["similarity"] for c in _feed(app, slug)["cards"]}
    detail = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))
    seeds = [s["id"] for s in detail["seeds"]]
    assert detail["profile"]["seeds"] == len(seeds)
    _ok(_request(app, "DELETE", f"/api/profiles/{slug}/seeds", params={"openalex_id": seeds[0]}))
    d2 = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))
    assert d2["profile"]["seeds"] == len(d2["seeds"]) == len(seeds) - 1
    lst = {p["key"]: p for p in _ok(_request(app, "GET", "/api/profiles"))}
    assert lst[slug]["seeds"] == len(seeds) - 1
    f1 = {c["id"]: c["similarity"] for c in _feed(app, slug)["cards"]}
    print("SIM", f0, f1)
    assert f0 != f1
    # The removed seed comes back.
    added = _ok(_request(app, "POST", f"/api/profiles/{slug}/seeds", json={"openalex_ids": [seeds[0]]}))
    print("ADDED", added)
    assert added["attached"] == 1
    f2 = {c["id"]: c["similarity"] for c in _feed(app, slug)["cards"]}
    for k in f0:
        if f0[k] is not None and k in f2:
            assert abs(f0[k] - f2[k]) < 1e-3, (k, f0[k], f2[k])
    # Upload a seed to a live interest.
    up = _ok(_upload(app, b"live-upload", "lu.pdf", slug))
    d3 = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))
    assert up["id"] in {s["id"] for s in d3["seeds"]}
    assert d3["profile"]["seeds"] == len(d3["seeds"]) == len(seeds) + 1
    import json
    cfg = json.loads(_one(env, "SELECT selector_config_json FROM profiles WHERE slug=?", slug)[0])
    assert sorted(cfg["fit_seed_ids"]) == sorted(s["id"] for s in d3["seeds"])
    # The feed never shows a seed.
    feed = _feed(app, slug)
    assert not ({c["id"] for c in feed["cards"]} & {s["id"] for s in d3["seeds"]})


def _code(app, method, path, **kw):
    try:
        return _request(app, method, path, **kw).status_code
    except Exception as exc:  # noqa: BLE001
        return f"unhandled {type(exc).__name__}: {exc}"


def test_empty_draft_after_removing_every_seed(app, env):
    slug = _draft_with_uploads(app, "Emptied")
    docs = _ok(_request(app, "GET", "/api/vault/docs", params={"tag": slug}))
    for d in docs:
        _ok(_request(app, "DELETE", f"/api/profiles/{slug}/seeds", params={"openalex_id": d["id"]}))
    assert _ok(_request(app, "GET", f"/api/vault/docs", params={"tag": slug})) == []
    drafts = _ok(_request(app, "GET", "/api/profiles/drafts"))
    assert drafts[0]["n_seeds"] == 0 and drafts[0]["phase"] == "empty"
    print("COH", _code(app, "POST", f"/api/profiles/draft/{slug}/coherence"))
    topics = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))
    print("TOPICS", topics)
    run = _ok(_request(app, "POST", f"/api/profiles/draft/{slug}/dry-run", json={"days": 30}))
    st = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/dry-run/{run['run_id']}"))
    print("DRY", st["result"])
    code = _code(app, "POST", "/api/profiles", json={"slug": slug, "threshold": 0.1, "selected_topic_ids": []})
    print("COMMIT", code)
    assert code in (400, 409), code


def test_wizard_routes_on_a_live_interest_and_live_routes_on_a_draft(app, env):
    slug, rid = _go_live(app, env, name="Live Routes")
    for m, p, kw in [
        ("GET", f"/api/profiles/draft/{slug}/topics", {}),
        ("POST", f"/api/profiles/draft/{slug}/coherence", {}),
        ("POST", f"/api/profiles/draft/{slug}/dry-run", {"json": {"days": 30}}),
        ("DELETE", f"/api/profiles/draft/{slug}", {}),
        ("POST", "/api/profiles", {"json": {"slug": slug, "threshold": 0.3, "selected_topic_ids": ["T10001"]}}),
    ]:
        c = _code(app, m, p, **kw)
        print("LIVE", m, p, c)
        assert c == 404, (m, p, c)
    d = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": "Just Draft"}))["slug"]
    _ok(_upload(app, b"jd", "jd.pdf", d))
    for m, p, kw in [
        ("POST", f"/api/profiles/{d}/gather-now", {}),
        ("POST", f"/api/profiles/{d}/backfill", {"json": {"year_from": 2020}}),
        ("POST", f"/api/profiles/{d}/refit", {}),
        ("DELETE", f"/api/profiles/{d}", {}),
        ("GET", f"/api/profiles/{d}/topic-yield", {}),
        ("GET", f"/api/profiles/{d}/reranker-comparison", {}),
        ("POST", f"/api/profiles/{d}/dry-run", {}),
        ("PATCH", f"/api/profiles/{d}", {"json": {"threshold": 0.4}}),
        ("PATCH", f"/api/profiles/{d}/schedule", {"json": {"enabled": True}}),
        ("GET", f"/api/profiles/{d}/feedback-threshold", {}),
        ("GET", f"/api/radar/daily?profile={d}", {}),
    ]:
        c = _code(app, m, p, **kw)
        print("DRAFT", m, p, c)
        assert not isinstance(c, str) and c < 500, (m, p, c)
    assert _ok(_request(app, "GET", "/api/profiles/drafts"))[0]["slug"] == d


def test_delete_interest_sharing_a_paper(app, env):
    a, rid = _go_live(app, env, name="Share A", threshold=0.0)
    papers = _ok(_request(app, "GET", f"/api/researchers/{rid}"))["papers"]
    ids = [p["id"] for p in papers]
    b = _ok(_request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": "Share B", "openalex_ids": ids}))["draft"]["slug"]
    on = [t["id"] for t in _ok(_request(app, "GET", f"/api/profiles/draft/{b}/topics")) if t["on"]]
    _ok(_request(app, "POST", "/api/profiles", json={"slug": b, "threshold": 0.0, "selected_topic_ids": on}))
    _scan(app, a)
    _scan(app, b)
    fa = _feed(app, a)["cards"]
    fb = _feed(app, b)["cards"]
    shared = sorted({c["id"] for c in fa} & {c["id"] for c in fb})
    assert shared
    x = shared[0]
    _ok(_request(app, "POST", "/api/radar/cards/save", json={"card_id": x, "profile": a}))
    _ok(_request(app, "POST", "/api/radar/cards/dismiss", json={"card_id": x, "profile": b}))
    allf = _feed(app)
    print("ALL", [(c["id"], c["profile"], c.get("state")) for c in allf["cards"] if c["id"] == x], allf["states"])
    _ok(_request(app, "DELETE", f"/api/profiles/{a}"))
    fb2 = _feed(app, b, )
    print("B after", fb2["states"])
    assert fb2["states"].get(x) == "dismissed"
    allf2 = _feed(app)
    assert {c["profile"] for c in allf2["cards"]} == {b}
    r = _ok(_request(app, "GET", f"/api/researchers/{rid}"))["researcher"]
    print("RES", r)
    assert r["n_interests"] == 1
    # seeds of the deleted interest's tag gone from the vault
    tags = _ok(_request(app, "GET", "/api/vault/tags"))
    print("TAGS", tags)
    assert a not in tags
    # Feedback of b intact, a's routes 404
    assert _code(app, "GET", f"/api/profiles/{a}/feedback") == 404
    assert _ok(_request(app, "GET", f"/api/profiles/{a}")) is None


def test_forget_researcher_with_live_and_draft_interests(app, env):
    slug, rid = _go_live(app, env, name="Forget Live")
    papers = _ok(_request(app, "GET", f"/api/researchers/{rid}"))["papers"]
    d = _ok(_request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": "Forget Draft", "openalex_ids": [papers[0]["id"]]}))["draft"]["slug"]
    _ok(_request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}))
    _ok(_request(app, "GET", f"/api/profiles/draft/{d}/topics"))
    _ok(_request(app, "PUT", f"/api/profiles/{d}/researcher-topics", json={"enabled": True}))
    _ok(_request(app, "DELETE", f"/api/researchers/{rid}"))
    v1 = _ok(_request(app, "GET", f"/api/profiles/{slug}/researcher-topics"))
    v2 = _ok(_request(app, "GET", f"/api/profiles/{d}/researcher-topics"))
    print("V", v1, v2)
    assert v1["researcher_id"] is None and v2["researcher_id"] is None
    det = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))
    print("LIVE TOPICS", [(t["id"], t["on"], t.get("profile"), t.get("source")) for t in det["topics"]])
    assert not any(t.get("profile") for t in det["topics"])
    dt = _ok(_request(app, "GET", f"/api/profiles/draft/{d}/topics"))
    print("DRAFT TOPICS", [(t["id"], t["on"], t.get("profile"), t.get("source")) for t in dt])
    assert not any(t.get("profile") for t in dt)
    assert any(t["on"] for t in dt)
    # seeds stay
    assert det["profile"]["seeds"] == len(det["seeds"]) > 0
    drafts = _ok(_request(app, "GET", "/api/profiles/drafts"))
    assert drafts[0]["researcher_name"] is None and drafts[0]["n_seeds"] == 1
    # The live interest still scans.
    _scan(app, slug)
    runs = _ok(_request(app, "GET", f"/api/profiles/{slug}/runs"))
    assert runs[0]["error"] is None, runs[0]


def _set_paper_topics(env, oa, tids):
    import json
    from rag_lib.db import connect
    blks = [{"id": t, "display_name": f"Name {t}", "score": 0.9,
             "subfield": {"id": "SF1", "display_name": "Sub"}, "field": {"id": "F1", "display_name": "F"},
             "domain": {"id": "D1", "display_name": "D"}} for t in tids]
    conn = connect(env)
    try:
        conn.execute("UPDATE papers SET topics_json=? WHERE openalex_id=?",
                     (json.dumps({"primary_topic": blks[0], "topics": blks}), oa))
        conn.commit()
    finally:
        conn.close()


def test_edit_topics_after_seed_changes(app, env):
    slug, rid, ids = _draft_from_researcher(app, "Edit Topics", env)
    topics = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))
    tids = sorted(t["id"] for t in topics)
    print("T", tids)
    keep = [t for t in tids if t != "T20001"]
    _ok(_request(app, "POST", "/api/profiles", json={"slug": slug, "threshold": 0.0, "selected_topic_ids": keep}))
    det = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))
    assert {t["id"] for t in det["topics"] if t["on"]} == set(keep)
    prev = _ok(_request(app, "GET", f"/api/profiles/{slug}/topics/preview"))
    assert [(t["id"], t["on"], t["new"]) for t in sorted(prev, key=lambda t: t["id"])] == \
        [(t["id"], t["on"], False) for t in sorted(det["topics"], key=lambda t: t["id"])]
    # A new seed with a new topic.
    up = _ok(_upload(app, b"new-topic-seed", "nt.pdf", slug))
    _set_paper_topics(env, up["id"], ["T30000"])
    prev = {t["id"]: t for t in _ok(_request(app, "GET", f"/api/profiles/{slug}/topics/preview"))}
    print("PREV", {k: (v["on"], v["new"]) for k, v in prev.items()})
    assert prev["T30000"]["new"] is True and prev["T30000"]["on"] is False
    det = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))
    assert "T30000" not in {t["id"] for t in det["topics"]}   # preview wrote nothing
    saved = _ok(_request(app, "PUT", f"/api/profiles/{slug}/topics", json={"selected_topic_ids": keep + ["T30000"]}))
    assert {t["id"] for t in saved if t["on"]} == set(keep) | {"T30000"}
    det = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))
    assert {t["id"] for t in det["topics"] if t["on"]} == set(keep) | {"T30000"}
    prev = {t["id"]: t for t in _ok(_request(app, "GET", f"/api/profiles/{slug}/topics/preview"))}
    assert not any(t["new"] for t in prev.values())
    # The new seed goes again: its topic leaves the list.
    _ok(_request(app, "DELETE", f"/api/profiles/{slug}/seeds", params={"openalex_id": up["id"]}))
    prev = {t["id"]: t for t in _ok(_request(app, "GET", f"/api/profiles/{slug}/topics/preview"))}
    print("PREV2", {k: (v["on"], v["new"]) for k, v in prev.items()})
    det = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))
    print("DET2", [(t["id"], t["on"]) for t in det["topics"]])
    # Re-adding the seed: the topic was switched on by the user; does it come back on?
    _ok(_request(app, "POST", f"/api/profiles/{slug}/seeds", json={"openalex_ids": [up["id"]]}))
    prev = {t["id"]: t for t in _ok(_request(app, "GET", f"/api/profiles/{slug}/topics/preview"))}
    print("PREV3", {k: (v["on"], v["new"]) for k, v in prev.items()})
    # Scan and topic-yield.
    _scan(app, slug)
    ty = _ok(_request(app, "GET", f"/api/profiles/{slug}/topic-yield"))
    print("TY", ty)


def test_dry_run_through_the_scheduler_path(app, env, monkeypatch):
    """The production path: no injected gatherer, the job runs under the scheduler."""
    from rag_lib.api.routers.profiles import get_wizard_gatherer
    from rag_lib.api.services import wizard as wizard_service
    from tests.fixture_gatherer import FixtureGatherer

    from rag_lib.api import app as app_module
    from rag_lib.api import settings as settings_module
    app.dependency_overrides.pop(get_wizard_gatherer)
    monkeypatch.setenv("RADAR_SCHEDULER_ENABLED", "true")
    settings_module.get_settings.cache_clear()
    monkeypatch.setattr(app_module, "build_scheduler", lambda settings: app.state.inline)
    monkeypatch.setattr(app_module, "start_scheduler", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "stop_scheduler", lambda *a, **k: None)
    monkeypatch.setattr(wizard_service, "OpenAlexGatherer", lambda **kw: FixtureGatherer(seed=42, n_candidates=12))
    slug, rid, ids = _draft_from_researcher(app, "Sched Dry", env)
    topics = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))
    pick = [topics[0]["id"]]
    run = _ok(_request(app, "POST", f"/api/profiles/draft/{slug}/dry-run",
                       json={"days": 30, "selected_topic_ids": pick, "year_from": 2019, "year_to": 2021}))
    st = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/dry-run/{run['run_id']}"))
    print("RUN", st["run"])
    assert st["run"]["finished_at"] and not st["run"]["error"], st["run"]
    assert st["result"] and len(st["result"]["scores"]) > 0
    for p in st["result"]["papers"]:
        assert p["year"] is None or 2019 <= p["year"] <= 2021
    after = _ok(_request(app, "GET", f"/api/profiles/draft/{slug}/topics"))
    assert {t["id"] for t in after if t["on"]} == set(pick)
    # The trial is not a scan: commit and Scan now are free.
    _ok(_request(app, "POST", "/api/profiles", json={"slug": slug, "threshold": 0.0, "selected_topic_ids": pick,
                                                     "years": {"year_from": 2019, "year_to": 2021}}))
    r = _ok(_request(app, "POST", f"/api/profiles/{slug}/backfill", json={"year_from": 2019, "year_to": 2021}))
    runs = _ok(_request(app, "GET", f"/api/profiles/{slug}/runs"))
    print("RUNS", [(x["tier_used"], x["n_fetched"], x["n_new"], x["error"]) for x in runs])
    assert runs[0]["error"] is None
    feed = _feed(app, slug)
    print("FEED", len(feed["cards"]), feed["candidatesScored"])
    assert feed["cards"]


class _DroppingScheduler:
    """Accepts jobs and never runs them: the backend died before they started."""

    def __init__(self):
        self.jobs = []

    def add_job(self, func, *a, args=(), kwargs=None, **k):
        self.jobs.append((func, args, kwargs, k))

    def get_job(self, *a, **k):
        return None

    def remove_job(self, *a, **k):
        return None


def _scheduler_on(app, monkeypatch, sched_holder):
    from rag_lib.api import app as app_module
    from rag_lib.api import settings as settings_module
    monkeypatch.setenv("RADAR_SCHEDULER_ENABLED", "true")
    settings_module.get_settings.cache_clear()
    monkeypatch.setattr(app_module, "build_scheduler", lambda settings: sched_holder[0])
    monkeypatch.setattr(app_module, "start_scheduler", lambda *a, **k: None)
    monkeypatch.setattr(app_module, "stop_scheduler", lambda *a, **k: None)


def test_restart_mid_import_then_resume(app, env, monkeypatch):
    from rag_lib.api.routers.prosopia import get_import_openalex_client
    from rag_lib.api.services import prosopia as prosopia_service

    holder = [_DroppingScheduler()]
    _scheduler_on(app, monkeypatch, holder)
    stub = app.state.openalex
    app.dependency_overrides.pop(get_import_openalex_client)
    monkeypatch.setattr(prosopia_service, "OpenAlexClient", lambda **kw: stub)
    start = _ok(_request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan", "name": "Restarted"}))
    slug, run_id = start["draft_slug"], start["run_id"]
    # Before the restart... but every _request runs the lifespan, i.e. a restart.
    st = _ok(_request(app, "GET", f"/api/import/prosopia/{run_id}"))
    print("ST", st["run"])
    assert st["run"]["finished_at"] and st["run"]["error"]
    drafts = _ok(_request(app, "GET", "/api/profiles/drafts"))
    print("DRAFTS", drafts)
    assert drafts[0]["phase"] == "failed" and drafts[0]["import_error"]
    rs = _ok(_request(app, "GET", "/api/researchers"))
    print("RS", rs)
    assert rs and rs[0]["importing"] is False
    # Retry into the same draft, with the jobs running this time.
    holder[0] = app.state.inline
    again = _ok(_request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan", "into": slug}))
    print("AGAIN", again)
    st2 = _ok(_request(app, "GET", f"/api/import/prosopia/{again['run_id']}"))
    print("ST2", st2["run"], st2["result"])
    assert st2["run"]["error"] is None
    drafts = _ok(_request(app, "GET", "/api/profiles/drafts"))
    print("DRAFTS2", drafts)
    assert drafts[0]["slug"] == slug and drafts[0]["phase"] == "seeded" and drafts[0]["n_seeds"] == 5
    assert drafts[0]["import_error"] is None
    rs = _ok(_request(app, "GET", "/api/researchers"))
    assert len(rs) == 1 and rs[0]["n_papers"] == 5 and rs[0]["n_interests"] == 1, rs
    view = _ok(_request(app, "GET", f"/api/profiles/{slug}/researcher-topics"))
    assert view["researcher_id"] == rs[0]["id"]


def test_restart_mid_researcher_import_then_build(app, env, monkeypatch):
    from rag_lib.api.routers.prosopia import get_import_openalex_client
    from rag_lib.api.services import prosopia as prosopia_service

    holder = [_DroppingScheduler()]
    _scheduler_on(app, monkeypatch, holder)
    stub = app.state.openalex
    app.dependency_overrides.pop(get_import_openalex_client)
    monkeypatch.setattr(prosopia_service, "OpenAlexClient", lambda **kw: stub)
    r = _ok(_request(app, "POST", "/api/researchers/import", json={"source": "prosopia", "ref": "sheffield-nathan"}))
    rid = r["researcher_id"]
    det = _ok(_request(app, "GET", f"/api/researchers/{rid}"))
    print("DET", det["researcher"], len(det["papers"]))
    assert det["researcher"]["importing"] is False and det["researcher"]["last_error"]
    holder[0] = app.state.inline
    r2 = _ok(_request(app, "POST", "/api/researchers/import", json={"source": "prosopia", "ref": "sheffield-nathan"}))
    assert r2["researcher_id"] == rid
    det = _ok(_request(app, "GET", f"/api/researchers/{rid}"))
    print("DET2", det["researcher"], len(det["papers"]))
    assert det["researcher"]["last_error"] is None and det["researcher"]["n_papers"] == 5
    d = _ok(_request(app, "POST", f"/api/researchers/{rid}/interests", json={"name": "After Restart"}))
    assert d["n_seeds"] == 5


def test_draft_seed_count_agrees_across_endpoints(app, env):
    slug = _draft_with_uploads(app, "Count Draft")
    drafts = {d["slug"]: d for d in _ok(_request(app, "GET", "/api/profiles/drafts"))}
    lst = {p["key"]: p for p in _ok(_request(app, "GET", "/api/profiles"))}
    one = _ok(_request(app, "GET", f"/api/profiles/{slug}"))
    det = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))
    print("COUNTS drafts", drafts[slug]["n_seeds"], "list", lst[slug]["seeds"], "get", one["seeds"],
          "detail.profile", det["profile"]["seeds"], "detail.seeds", len(det["seeds"]), "isDraft", one["isDraft"])
    assert drafts[slug]["n_seeds"] == len(det["seeds"]) == 2
    assert lst[slug]["seeds"] == one["seeds"] == det["profile"]["seeds"] == 2
