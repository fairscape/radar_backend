"""Review 17 (b): numbers the API reports that disagree with each other.

Each test drives the HTTP routes the frontend calls and compares two of
them (or one of them with the rule it claims to apply).
"""

from __future__ import annotations

from rag_lib.db import connect
from tests.test_review_round12_journeys import (  # noqa: F401
    env, app, _feed, _go_live, _ok, _one, _q, _request, _scan, _upload,
)


def _exec(env, sql, *params):
    conn = connect(env)
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def _pid(env, slug):
    return _one(env, "SELECT id FROM profiles WHERE slug = ?", slug)["id"]


def test_below_the_bar_paper_reads_as_at_the_bar(app, env):
    """A paper 0.00003 under the threshold: the feed files it under "Below
    the bar" but reports its similarity as exactly the threshold (delta
    +0.000), and the Threshold tab counts it as passing (score >= value)."""
    slug, _ = _go_live(app, env, name="Edge", threshold=0.0)
    _scan(app, slug)
    _ok(_request(app, "PATCH", f"/api/profiles/{slug}", json={"threshold": 0.5}))
    pid = _pid(env, slug)
    _exec(env, "UPDATE profile_candidates SET score_raw = 0.8 WHERE profile_id = ?", pid)
    oa = _q(env, "SELECT openalex_id FROM profile_candidates WHERE profile_id = ? LIMIT 1", pid)[0][0]
    _exec(env, "UPDATE profile_candidates SET score_raw = 0.49997 WHERE profile_id = ? AND openalex_id = ?", pid, oa)

    in_feed = {c["id"] for c in _feed(app, slug)["cards"]}
    assert oa not in in_feed                                   # the feed: below the bar
    low = {c["id"]: c for c in _feed(app, slug, bucket="low")["cards"]}
    card = low[oa]
    assert card["bucket"] == "low"
    # Card similarity is what the frontend subtracts the threshold from.
    assert card["similarity"] < 0.5, f"below-bar card reports similarity {card['similarity']} (raw 0.49997)"


def test_threshold_tab_counts_a_failing_paper_as_passing(app, env):
    slug, _ = _go_live(app, env, name="Edge2", threshold=0.0)
    _scan(app, slug)
    _ok(_request(app, "PATCH", f"/api/profiles/{slug}", json={"threshold": 0.5}))
    pid = _pid(env, slug)
    _exec(env, "UPDATE profile_candidates SET score_raw = 0.8 WHERE profile_id = ?", pid)
    oa = _q(env, "SELECT openalex_id FROM profile_candidates WHERE profile_id = ? LIMIT 1", pid)[0][0]
    _exec(env, "UPDATE profile_candidates SET score_raw = 0.49997 WHERE profile_id = ? AND openalex_id = ?", pid, oa)

    feed_n = len(_feed(app, slug)["cards"])
    dry = _ok(_request(app, "POST", f"/api/profiles/{slug}/dry-run"))
    # ThresholdPaperList / ThresholdHistogram: papers.filter(p => p.score >= value)
    tab_n = sum(1 for p in dry["papers"] if p["score"] >= 0.5)
    assert tab_n == feed_n, f"Threshold tab says {tab_n} pass at the saved threshold, the feed has {feed_n}"


def test_dismissed_paper_made_a_seed_still_counts_as_dismissed(app, env):
    """Interest card / detail stat "N dismissed (30 d)" vs the feed's
    "Show saved and dismissed (N)" vs the Threshold tab's feedback count."""
    slug, _ = _go_live(app, env, name="Seeded", threshold=0.0)
    _scan(app, slug)
    card = _feed(app, slug)["cards"][0]
    _ok(_request(app, "POST", "/api/radar/cards/dismiss", json={"card_id": card["id"], "profile": slug, "on": True}))
    _ok(_request(app, "POST", f"/api/profiles/{slug}/seeds", json={"openalex_ids": [card["id"]]}))

    prof = next(p for p in _ok(_request(app, "GET", "/api/profiles")) if p["key"] == slug)
    feed = _feed(app, slug)["cards"] + _feed(app, slug, bucket="low")["cards"]
    feed_dismissed = sum(1 for c in feed if c.get("state") == "dismissed")
    ft = _ok(_request(app, "GET", f"/api/profiles/{slug}/feedback-threshold"))
    assert (prof["dismisses30"], feed_dismissed, ft["n_dismissed"]) == (feed_dismissed,) * 3, (
        prof["dismisses30"], feed_dismissed, ft["n_dismissed"])


def test_below_the_bar_across_interests_is_not_closest_to_the_bar_first(app, env):
    """"Below the bar, closest to it first" -- across interests it sorts by
    raw similarity, so a paper far under a high threshold precedes one just
    under a low threshold."""
    a, _ = _go_live(app, env, name="High", threshold=0.0)
    b = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": "Low"}))["slug"]
    _ok(_upload(app, b"low-seed", "low.pdf", b))
    kicked = _ok(_request(app, "POST", f"/api/profiles/draft/{b}/dry-run", json={"days": 7}))
    _ok(_request(app, "GET", f"/api/profiles/draft/{b}/dry-run/{kicked['run_id']}"))
    _ok(_request(app, "POST", "/api/profiles", json={"slug": b, "threshold": 0.0, "selected_topic_ids": []}))
    _scan(app, a)
    _scan(app, b)
    _ok(_request(app, "PATCH", f"/api/profiles/{a}", json={"threshold": 0.9}))
    _ok(_request(app, "PATCH", f"/api/profiles/{b}", json={"threshold": 0.5}))
    pa, pb = _pid(env, a), _pid(env, b)
    _exec(env, "UPDATE profile_candidates SET score_raw = 0.95 WHERE profile_id = ?", pa)
    _exec(env, "UPDATE profile_candidates SET score_raw = 0.95 WHERE profile_id = ?", pb)
    far = _q(env, "SELECT openalex_id FROM profile_candidates WHERE profile_id = ? LIMIT 1", pa)[0][0]
    near = _q(env, "SELECT openalex_id FROM profile_candidates WHERE profile_id = ? AND openalex_id != ? LIMIT 1", pb, far)[0][0]
    _exec(env, "UPDATE profile_candidates SET score_raw = 0.80 WHERE profile_id = ? AND openalex_id = ?", pa, far)   # 0.10 under
    _exec(env, "UPDATE profile_candidates SET score_raw = 0.49 WHERE profile_id = ? AND openalex_id = ?", pb, near)  # 0.01 under

    order = [(c["profile"], c["id"]) for c in _feed(app, bucket="low")["cards"]]
    assert order.index((b, near)) < order.index((a, far)), order
