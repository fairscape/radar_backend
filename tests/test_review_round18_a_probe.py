"""Review 18 (a): every route against draft / live / other-user / missing keys; no 500s."""

from __future__ import annotations

from tests.test_review_round12_journeys import (  # noqa: F401 -- fixtures
    B, _go_live, _import_researcher, _ok, _one, _q, _request, _scan, _upload, app, env,
)


def test_no_500_across_object_states(app, env):
    live, rid = _go_live(app, env, "Live One")
    _scan(app, live)
    draft = _ok(_request(app, "POST", "/api/profiles/draft", json={"name": "Just Draft"}))["slug"]
    _ok(_upload(app, b"draft-seed", "d.pdf", draft))
    other_live, other_rid = _go_live(app, env, "Bs Live", headers=B)
    seed = _q(env, "SELECT openalex_id FROM profile_seeds LIMIT 1")[0][0]
    keys = {"live": live, "draft": draft, "missing": "nope-nope", "other": other_live}
    bad = []
    for label, k in keys.items():
        calls = [
            ("GET", f"/api/profiles/{k}", {}),
            ("GET", f"/api/profiles/{k}/detail", {}),
            ("PATCH", f"/api/profiles/{k}", {"json": {"threshold": 0.2}}),
            ("PATCH", f"/api/profiles/{k}", {"json": {"years": {"year_from": 2019, "year_to": None}}}),
            ("POST", f"/api/profiles/{k}/refit", {}),
            ("POST", f"/api/profiles/{k}/refit?autotune=1", {}),
            ("POST", f"/api/profiles/{k}/dry-run", {"json": {}}),
            ("POST", f"/api/profiles/{k}/recompute-coherence", {}),
            ("POST", f"/api/profiles/{k}/recompute-topics", {}),
            ("GET", f"/api/profiles/{k}/topics/preview", {}),
            ("PUT", f"/api/profiles/{k}/topics", {"json": {"selected_topic_ids": []}}),
            ("GET", f"/api/profiles/{k}/researcher-topics", {}),
            ("PUT", f"/api/profiles/{k}/researcher-topics", {"json": {"enabled": True}}),
            ("PUT", f"/api/profiles/{k}/researcher-topics", {"json": {"enabled": False}}),
            ("POST", f"/api/profiles/{k}/seeds", {"json": {"openalex_ids": [seed]}}),
            ("DELETE", f"/api/profiles/{k}/seeds", {"params": {"openalex_id": seed}}),
            ("GET", f"/api/profiles/{k}/runs", {}),
            ("GET", f"/api/profiles/{k}/feedback", {}),
            ("GET", f"/api/profiles/{k}/feedback-threshold", {}),
            ("GET", f"/api/profiles/{k}/terms", {}),
            ("GET", f"/api/profiles/{k}/umls", {}),
            ("GET", f"/api/profiles/{k}/topic-yield", {}),
            ("GET", f"/api/profiles/{k}/reranker-comparison", {}),
            ("POST", f"/api/profiles/{k}/backfill", {"json": {"year_from": 2020}}),
            ("GET", f"/api/profiles/draft/{k}/topics", {}),
            ("POST", f"/api/profiles/draft/{k}/coherence", {}),
            ("POST", f"/api/profiles/draft/{k}/dry-run", {"json": {"days": 30}}),
            ("POST", f"/api/profiles/draft/{k}/seeds", {"json": {"openalex_ids": [seed]}}),
            ("POST", "/api/profiles", {"json": {"slug": k, "threshold": 0.1, "selected_topic_ids": []}}),
            ("GET", f"/api/radar/daily?profile={k}", {}),
            ("POST", "/api/radar/cards/save", {"json": {"card_id": seed, "profile": k, "on": True}}),
            ("GET", f"/api/vault/docs?tag={k}", {}),
        ]
        for method, path, kw in calls:
            r = _request(app, method, path, **kw)
            print(label, method, path, r.status_code, r.text[:120])
            if r.status_code >= 500:
                bad.append((label, method, path, r.status_code, r.text[:200]))
    for rk, r_id in {"own": rid, "other": other_rid, "missing": 999999}.items():
        for method, path, kw in [
            ("GET", f"/api/researchers/{r_id}", {}),
            ("GET", f"/api/researchers/{r_id}/suggestions", {}),
            ("GET", f"/api/researchers/{r_id}/works", {}),
            ("PATCH", f"/api/researchers/{r_id}", {"json": {"mark": "self"}}),
            ("POST", f"/api/researchers/{r_id}/interests", {"json": {"name": "X"}}),
            ("PUT", "/api/researchers/order", {"json": {"ids": [r_id]}}),
        ]:
            r = _request(app, method, path, **kw)
            print("researcher", rk, method, path, r.status_code, r.text[:120])
            if r.status_code >= 500:
                bad.append((rk, method, path, r.status_code, r.text[:200]))
    for k in (live,):
        r = _request(app, "DELETE", f"/api/profiles/{k}", headers=B)
        print("B deletes A's", r.status_code)
        assert r.status_code == 404
    assert not bad, bad
