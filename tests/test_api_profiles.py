"""Profiles API tests.

Stand up the FastAPI app against a tmp SQLite seeded with one profile +
50 candidates + a finished gather run, then exercise the read endpoints
and the placeholder write endpoints.
"""

from __future__ import annotations

import asyncio
import sqlite3

import httpx
import numpy as np
import pytest

from rag_lib.api import settings as settings_module
from rag_lib.db import apply_migrations, connect, encode_vector
from rag_lib.db.repos import (
    candidates as candidates_repo,
    embeddings as embeddings_repo,
    gather_runs as gather_runs_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    users as users_repo,
)


def _seed_db(db_path) -> dict:
    """Create one profile + 12 seed papers + 50 candidate papers."""
    conn = connect(db_path)
    apply_migrations(conn)
    user = users_repo.upsert(conn, "demo@example.com")
    user_id = int(user["id"])

    rng = np.random.default_rng(42)
    profile_id = profiles_repo.upsert(
        conn,
        user_id=user_id,
        name="provenance / fairscape",
        embedding_model="placeholder-v1",
        n_seed=12,
        topic_filters={"topics": [
            {"id": "T11431", "display_name": "Research Data Provenance and Evidence", "count": 11},
            {"id": "T10123", "display_name": "FAIR Data Principles", "count": 9},
        ]},
        threshold=0.85,
        coherence_median=0.78,
        coherence_iqr=0.08,
        coherence_bimodal=False,
    )

    # 12 seed papers + their embeddings.
    for i in range(12):
        oa = f"W{1000000 + i}"
        papers_repo.upsert(conn, {
            "openalex_id": oa,
            "title": f"Seed paper {i+1}: provenance topic {i}",
            "year": 2024,
            "venue": "JAMIA",
            "source": "user_csv",
            "primary_topic": {"id": "T11431", "display_name": "Research Data Provenance and Evidence"},
            "topics": [{"id": "T10123", "display_name": "FAIR Data Principles"}],
        })
        v = rng.standard_normal(64).astype("float32")
        v /= (np.linalg.norm(v) + 1e-9)
        embeddings_repo.upsert(conn, oa, "placeholder-v1", v)
        profiles_repo.attach_seed(conn, profile_id, oa)

    # 50 candidates with descending scores.
    run_id = gather_runs_repo.start(
        conn, profile_id=profile_id, user_id=user_id,
        since_date="2026-03-26", filter_string="topics.id:T11431",
        tier_used="must-have-AND",
    )
    for i in range(50):
        oa = f"W{2000000 + i}"
        score = 0.95 - (i * 0.01)
        papers_repo.upsert(conn, {
            "openalex_id": oa,
            "doi": f"10.1234/foo.{i}" if i % 2 == 0 else None,
            "title": f"Candidate paper {i+1}: about something",
            "abstract": "We present a method for " + (" ".join(["lorem"] * (40 + i))),
            "year": 2026,
            "venue": "bioRxiv" if i % 3 == 0 else "Nature Methods",
            "publication_date": "2026-04-15",
            "source": "openalex_gatherer",
            "primary_topic": (
                {"id": "T11431", "display_name": "Research Data Provenance and Evidence"}
                if i % 2 == 0 else
                {"id": "T99999", "display_name": "Something else"}
            ),
            "topics": [{"id": "T10123", "display_name": "FAIR Data Principles"}] if i % 4 == 0 else [],
        })
        candidates_repo.insert_dedup(
            conn,
            profile_id=profile_id,
            openalex_id=oa,
            score=score,
            tier_used="must-have-AND",
            gather_run_id=run_id,
        )
    gather_runs_repo.finish(
        conn, run_id,
        n_fetched=50, n_new=50, n_redup=0, api_calls=3,
    )
    conn.close()
    return {"user_id": user_id, "profile_id": profile_id}


@pytest.fixture()
def app(tmp_path, monkeypatch):
    db = tmp_path / "radar.db"
    monkeypatch.setenv("RADAR_DB_PATH", str(db))
    monkeypatch.setenv("RADAR_VAULT_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("RADAR_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("RADAR_LOG_JSON", "false")
    settings_module.get_settings.cache_clear()

    _seed_db(db)

    from rag_lib.api.app import create_app

    app = create_app()
    yield app

    settings_module.get_settings.cache_clear()


def _request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(method, path, **kwargs)
    return asyncio.run(_run())


def test_list_profiles_returns_seeded_profile(app):
    resp = _request(app, "GET", "/api/profiles")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert isinstance(body, list) and len(body) == 1
    p = body[0]
    assert p["key"] == "provenance-fairscape"
    assert p["name"] == "provenance / fairscape"
    assert p["seeds"] == 12
    assert p["threshold"] == 0.85
    assert isinstance(p["hue"], int) and 0 <= p["hue"] < 360
    # Health derives from coherence + 30d engagement; coherence=0.78 OK,
    # but no saves/dismisses yet → warn.
    # Calibrated bands (rag_lib.calibration): 0.78 sits below the
    # similarity of random same-field papers, so it reads as "mixed".
    assert p["health"] == "err"
    assert p["coherenceLabel"] == "mixed"
    assert isinstance(p["agreement"], int)


def test_get_profile_by_key(app):
    resp = _request(app, "GET", "/api/profiles/provenance-fairscape")
    assert resp.status_code == 200, resp.text
    p = resp.json()
    assert p["key"] == "provenance-fairscape"


def test_get_profile_unknown_key_returns_null(app):
    resp = _request(app, "GET", "/api/profiles/does-not-exist")
    assert resp.status_code == 200
    assert resp.json() is None


def test_profile_detail_shape(app):
    resp = _request(app, "GET", "/api/profiles/provenance-fairscape/detail")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body.keys()) >= {
        "profile", "seeds", "topics", "sweep",
        "coherenceBins", "coherenceStats",
        "feedbackLog", "feedbackMoreCount",
    }
    assert body["profile"]["key"] == "provenance-fairscape"
    assert len(body["seeds"]) == 12
    assert body["seeds"][0]["idx"] == 1
    assert {"id", "name", "count", "on"} <= set(body["topics"][0].keys())
    assert isinstance(body["coherenceBins"], list) and len(body["coherenceBins"]) == 16
    assert body["feedbackLog"] == [] and body["feedbackMoreCount"] == 0


def test_refit_response_shape(app):
    resp = _request(app, "POST", "/api/profiles/provenance-fairscape/refit")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["key"] == "provenance-fairscape"
    assert isinstance(body["cost"], str)


def test_refit_unknown_profile_404(app):
    resp = _request(app, "POST", "/api/profiles/does-not-exist/refit")
    assert resp.status_code == 404


def test_dry_run_reports_candidate_count(app):
    resp = _request(app, "POST", "/api/profiles/provenance-fairscape/dry-run")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert body["key"] == "provenance-fairscape"
    assert body["n"] == 50
    # Per-candidate scores power the histogram + slider on the
    # profile detail page; one score per persisted candidate.
    assert isinstance(body["scores"], list)
    assert len(body["scores"]) == 50


# --- seeds of any interest, and a real refit ---------------------------------
#
# Until 2026-10-01 /refit returned a fixed placeholder and fitted nothing,
# the scheduler kept scoring with the centroid fitted at commit, a live
# interest's seeds could not be removed at all, and a draft's could not be
# removed when their id was the usual https://openalex.org/W… (the encoded
# slashes were decoded before routing, so the route never matched).

SLUG = "provenance-fairscape"


def _profile_row(tmp_path):
    import json as _json

    conn = connect(tmp_path / "radar.db")
    row = conn.execute(
        "SELECT selector_config_json, n_seed FROM profiles WHERE slug = ?", (SLUG,),
    ).fetchone()
    seeds = [r[0] for r in conn.execute(
        "SELECT ps.openalex_id FROM profile_seeds ps JOIN profiles p ON p.id = ps.profile_id "
        "WHERE p.slug = ?", (SLUG,))]
    conn.close()
    cfg = _json.loads(row[0]) if row[0] else {}
    return cfg.get("centroid"), row[1], seeds


def test_refit_fits_the_current_seeds(app, tmp_path):
    resp = _request(app, "POST", f"/api/profiles/{SLUG}/refit")
    assert resp.status_code == 200, resp.text
    assert resp.json()["cost"].endswith("12 vecs")  # was always "0.00 s · 0 vecs"
    centroid, n_seed, _ = _profile_row(tmp_path)
    assert centroid is not None and len(centroid) == 64
    assert n_seed == 12


def test_removing_a_seed_from_a_live_interest_refits_it(app, tmp_path):
    _request(app, "POST", f"/api/profiles/{SLUG}/refit")
    before, _, _ = _profile_row(tmp_path)
    resp = _request(app, "DELETE", f"/api/profiles/{SLUG}/seeds", params={"openalex_id": "W1000000"})
    assert resp.status_code == 200, resp.text
    after, n_seed, seeds = _profile_row(tmp_path)
    assert "W1000000" not in seeds and n_seed == 11
    assert after != before, "the centroid must follow the seeds"


def test_a_url_form_seed_can_be_removed_on_either_route(app, tmp_path):
    slug = _request(app, "POST", "/api/profiles/draft", json={"name": "url seeds"}).json()["slug"]
    conn = connect(tmp_path / "radar.db")
    pid = conn.execute("SELECT id FROM profiles WHERE slug = ?", (slug,)).fetchone()[0]
    for oa in ("https://openalex.org/W3000001", "https://openalex.org/W3000002"):
        papers_repo.upsert(conn, {"openalex_id": oa, "title": oa, "source": "user_pdf"})
        profiles_repo.attach_seed(conn, pid, oa)
    conn.close()

    old = _request(app, "DELETE", f"/api/profiles/draft/{slug}/seeds/https%3A%2F%2Fopenalex.org%2FW3000001")
    assert old.status_code == 200, old.text
    new = _request(app, "DELETE", f"/api/profiles/{slug}/seeds",
                   params={"openalex_id": "https://openalex.org/W3000002"})
    assert new.status_code == 200, new.text


def test_the_last_seed_of_a_live_interest_is_kept(app, tmp_path):
    conn = connect(tmp_path / "radar.db")
    pid = conn.execute("SELECT id FROM profiles WHERE slug = ?", (SLUG,)).fetchone()[0]
    for i in range(1, 12):
        profiles_repo.detach_seed(conn, pid, f"W{1000000 + i}")
    conn.close()
    resp = _request(app, "DELETE", f"/api/profiles/{SLUG}/seeds", params={"openalex_id": "W1000000"})
    assert resp.status_code == 409, resp.text
    assert _profile_row(tmp_path)[2] == ["W1000000"]


def test_a_feed_paper_can_become_a_seed_of_a_live_interest(app, tmp_path):
    """A candidate has no stored vector; attaching it must embed it, or the
    fit would silently leave it out."""
    # The fixture's seed vectors are hand-made 64-d noise; re-embed them with
    # the profile's own embedder so every vector in the fit comes from one
    # model, as it always does in production.
    from rag_lib.embedders import get_embedder

    embed = get_embedder("placeholder-v1")
    conn = connect(tmp_path / "radar.db")
    for i in range(12):
        oa = f"W{1000000 + i}"
        embeddings_repo.upsert(conn, oa, "placeholder-v1", embed(f"Seed paper {i + 1}"))
    conn.close()
    _request(app, "POST", f"/api/profiles/{SLUG}/refit")
    before, _, _ = _profile_row(tmp_path)
    resp = _request(app, "POST", f"/api/profiles/{SLUG}/seeds", json={"openalex_ids": ["W2000000"]})
    assert resp.status_code == 200, resp.text
    # The candidates have no stored vector except the new seed itself.
    assert resp.json() == {"attached": 1, "rejected": [], "rescored": 1}
    conn = connect(tmp_path / "radar.db")
    assert embeddings_repo.has(conn, "W2000000", "placeholder-v1")
    conn.close()
    after, n_seed, seeds = _profile_row(tmp_path)
    assert "W2000000" in seeds and n_seed == 13
    assert after != before


def test_a_paper_that_is_not_the_users_is_refused(app, tmp_path):
    conn = connect(tmp_path / "radar.db")
    papers_repo.upsert(conn, {"openalex_id": "W4000000", "title": "someone else's", "source": "openalex_gatherer"})
    conn.close()
    resp = _request(app, "POST", f"/api/profiles/{SLUG}/seeds", json={"openalex_ids": ["W4000000"]})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"attached": 0, "rejected": ["W4000000"], "rescored": None}


def test_a_feed_paper_attached_to_a_draft_is_embedded_at_once(app, tmp_path):
    """A draft is not re-fitted on attach -- it is fitted at commit -- so the
    attach itself has to embed. (On a live interest the refit embeds too,
    which is why the live test above cannot catch a missing embed here.)"""
    slug = _request(app, "POST", "/api/profiles/draft",
                    json={"name": "from the feed", "embedding_model": "placeholder-v1"}).json()["slug"]
    resp = _request(app, "POST", f"/api/profiles/{slug}/seeds", json={"openalex_ids": ["W2000001"]})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"attached": 1, "rejected": [], "rescored": None}
    conn = connect(tmp_path / "radar.db")
    assert embeddings_repo.has(conn, "W2000001", "placeholder-v1")
    conn.close()


def test_committing_a_draft_registers_its_gather_job_at_once(app, tmp_path, monkeypatch):
    """The schedule row used to be written and the job left for 'the next
    scheduler boot', so a new interest had no nightly scan until a restart."""
    import rag_lib.scheduler.runner as runner

    registered: list[int] = []
    monkeypatch.setattr(runner, "register_gather_job",
                        lambda scheduler, **kw: registered.append(kw["profile_id"]) or True)
    slug = _request(app, "POST", "/api/profiles/draft",
                    json={"name": "to commit", "embedding_model": "placeholder-v1"}).json()["slug"]
    _request(app, "POST", f"/api/profiles/{slug}/seeds", json={"openalex_ids": ["W2000002", "W2000003"]})
    resp = _request(app, "POST", "/api/profiles",
                    json={"slug": slug, "threshold": 0.5, "selected_topic_ids": []})
    assert resp.status_code == 200, resp.text
    conn = connect(tmp_path / "radar.db")
    pid = conn.execute("SELECT id FROM profiles WHERE slug = ?", (slug,)).fetchone()[0]
    conn.close()
    assert registered == [pid]


def test_a_seed_uploaded_into_a_live_interest_refits_it(app, tmp_path):
    """The Seeds tab's PDF drop zone attaches through the upload path, which
    wrote the seed row and never re-fitted -- listed, but not scored against."""
    from rag_lib.api.services import vault as vault_service
    from rag_lib.embedders import get_embedder

    embed = get_embedder("placeholder-v1")
    conn = connect(tmp_path / "radar.db")
    for i in range(12):
        embeddings_repo.upsert(conn, f"W{1000000 + i}", "placeholder-v1", embed(f"Seed paper {i + 1}"))
    papers_repo.upsert(conn, {"openalex_id": "W5000000", "title": "An uploaded seed", "source": "user_pdf"})
    embeddings_repo.upsert(conn, "W5000000", "placeholder-v1", embed("An uploaded seed"))
    conn.close()
    _request(app, "POST", f"/api/profiles/{SLUG}/refit")
    before, _, _ = _profile_row(tmp_path)

    conn = connect(tmp_path / "radar.db")
    uid = conn.execute("SELECT id FROM users WHERE email = 'demo@example.com'").fetchone()[0]
    vault_service._attach_to_profile(conn, uid, SLUG, "W5000000")
    conn.close()

    after, n_seed, seeds = _profile_row(tmp_path)
    assert "W5000000" in seeds and n_seed == 13
    assert after != before


def test_a_commit_with_a_bad_cron_is_refused_before_anything_changes(app, tmp_path):
    """The cron used to be checked only when registering the job -- after the
    draft had gone live -- so a bad one answered 500 to a done commit."""
    slug = _request(app, "POST", "/api/profiles/draft",
                    json={"name": "bad cron", "embedding_model": "placeholder-v1"}).json()["slug"]
    _request(app, "POST", f"/api/profiles/{slug}/seeds", json={"openalex_ids": ["W2000004"]})
    resp = _request(app, "POST", "/api/profiles",
                    json={"slug": slug, "threshold": 0.5, "selected_topic_ids": [], "cron": "0 4 * *"})
    assert resp.status_code == 400, resp.text
    conn = connect(tmp_path / "radar.db")
    assert conn.execute("SELECT is_draft FROM profiles WHERE slug = ?", (slug,)).fetchone()[0] == 1
    conn.close()


def test_removing_a_seed_is_refused_before_anything_changes_if_nothing_could_be_fitted(app, tmp_path, monkeypatch):
    """The seed row used to be deleted first and the refit run after; a refit
    that then failed answered an error with the seed already gone."""
    from rag_lib.api.services import wizard as wizard_service

    monkeypatch.setattr(wizard_service, "embed_missing", lambda *a, **k: 0)
    conn = connect(tmp_path / "radar.db")
    conn.execute("DELETE FROM paper_embeddings WHERE openalex_id <> 'W1000000'")
    conn.commit()
    uid = conn.execute("SELECT id FROM users WHERE email = 'demo@example.com'").fetchone()[0]
    with pytest.raises(ValueError):
        wizard_service.remove_seed(conn, user_id=uid, slug=SLUG, openalex_id="W1000000")
    conn.close()
    assert "W1000000" in _profile_row(tmp_path)[2]


def test_a_failed_attach_is_repaired_by_trying_again(app, tmp_path, monkeypatch):
    """The seed row commits before embedding runs. If embedding failed, a
    retry found it 'already attached' and never embedded or re-fitted it."""
    from rag_lib.api.services import researchers as researchers_service
    from rag_lib.api.services import wizard as wizard_service
    from rag_lib.embedders import get_embedder

    embed = get_embedder("placeholder-v1")
    conn = connect(tmp_path / "radar.db")
    for i in range(12):
        embeddings_repo.upsert(conn, f"W{1000000 + i}", "placeholder-v1", embed(f"Seed paper {i + 1}"))
    uid = conn.execute("SELECT id FROM users WHERE email = 'demo@example.com'").fetchone()[0]
    wizard_service.refit_profile(conn, user_id=uid, slug=SLUG)
    before = _profile_row(tmp_path)[0]

    real = wizard_service.embed_missing
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("embedder down")
        return real(*a, **k)
    monkeypatch.setattr(wizard_service, "embed_missing", flaky)

    with pytest.raises(RuntimeError):
        researchers_service.attach_seeds(conn, user_id=uid, slug=SLUG, openalex_ids=["W2000005"])
    assert not embeddings_repo.has(conn, "W2000005", "placeholder-v1")
    researchers_service.attach_seeds(conn, user_id=uid, slug=SLUG, openalex_ids=["W2000005"])
    assert embeddings_repo.has(conn, "W2000005", "placeholder-v1")
    conn.close()
    assert _profile_row(tmp_path)[0] != before


def test_a_bad_timezone_alone_is_refused_on_commit_and_on_schedule_edit(app, tmp_path):
    slug = _request(app, "POST", "/api/profiles/draft",
                    json={"name": "bad tz", "embedding_model": "placeholder-v1"}).json()["slug"]
    _request(app, "POST", f"/api/profiles/{slug}/seeds", json={"openalex_ids": ["W2000006"]})
    resp = _request(app, "POST", "/api/profiles",
                    json={"slug": slug, "threshold": 0.5, "selected_topic_ids": [], "tz": "Bad/Zone"})
    assert resp.status_code == 400, resp.text
    resp = _request(app, "PATCH", f"/api/profiles/{SLUG}/schedule", json={"tz": "Bad/Zone"})
    assert resp.status_code == 400, resp.text
    conn = connect(tmp_path / "radar.db")
    stored = [r[0] for r in conn.execute("SELECT tz FROM profile_schedules")]
    conn.close()
    assert "Bad/Zone" not in stored


def test_moving_the_threshold_changes_the_feed_at_once(app, tmp_path):
    """The threshold used to apply only when a scan stored papers, so editing
    it changed nothing the user could see. The feed now applies it on read,
    and the threshold tab lists every stored paper, above and below it."""
    conn = connect(tmp_path / "radar.db")
    conn.execute("UPDATE profile_candidates SET score_raw = score")   # 0.95 .. 0.46
    conn.commit()
    pid = conn.execute("SELECT id FROM profiles WHERE slug = ?", (SLUG,)).fetchone()[0]
    conn.close()

    def feed():
        c = connect(tmp_path / "radar.db")
        try:
            rows = candidates_repo.top_for_profile(c, pid, limit=100)
            return len(rows), [r["score_pct"] for r in rows]
        finally:
            c.close()

    assert _request(app, "PATCH", f"/api/profiles/{SLUG}", json={"threshold": 0.795}).status_code == 200
    n, pcts = feed()
    assert n == 16                                       # 0.95 .. 0.80
    assert max(pcts) == 1.0 and min(pcts) == 0.0         # ranked among themselves
    assert _request(app, "PATCH", f"/api/profiles/{SLUG}", json={"threshold": 0.595}).status_code == 200
    assert feed()[0] == 36

    body = _request(app, "POST", f"/api/profiles/{SLUG}/dry-run").json()
    assert len(body["papers"]) == 50                     # the tab lists all of them
    assert body["papers"][0]["score"] == 0.95 and body["papers"][-1]["score"] == 0.46
    assert body["papers"][0]["title"].startswith("Candidate paper 1")


def test_adding_a_seed_rescores_the_papers_already_found(app, tmp_path):
    """A refit moved the centroid but left every stored candidate with the
    similarity it had against the old one, so a new seed changed nothing in
    the feed until the next morning's scan."""
    import json as _json

    from rag_lib.embedders import get_embedder

    embed = get_embedder("placeholder-v1")
    conn = connect(tmp_path / "radar.db")
    for i in range(12):
        embeddings_repo.upsert(conn, f"W{1000000 + i}", "placeholder-v1", embed(f"Seed paper {i + 1}"))
    for i in range(50):                                # stored as a gather now stores them
        embeddings_repo.upsert(conn, f"W{2000000 + i}", "placeholder-v1", embed(f"Candidate paper {i + 1}"))
    conn.close()
    assert _request(app, "POST", f"/api/profiles/{SLUG}/refit").status_code == 200

    def scores():
        c = connect(tmp_path / "radar.db")
        try:
            return dict(c.execute(
                "SELECT pc.openalex_id, pc.score_raw FROM profile_candidates pc "
                "JOIN profiles p ON p.id = pc.profile_id WHERE p.slug = ?", (SLUG,)).fetchall())
        finally:
            c.close()

    before = scores()
    resp = _request(app, "POST", f"/api/profiles/{SLUG}/seeds", json={"openalex_ids": ["W2000007"]})
    assert resp.status_code == 200, resp.text
    assert resp.json()["rescored"] == 50
    after = scores()

    c = connect(tmp_path / "radar.db")
    cfg = _json.loads(c.execute("SELECT selector_config_json FROM profiles WHERE slug = ?", (SLUG,)).fetchone()[0])
    centroid = np.asarray(cfg["centroid"], dtype=float)
    for oa, got in after.items():                       # each = cosine to the *new* centroid
        v = np.asarray(embeddings_repo.get(c, oa, "placeholder-v1"), dtype=float)
        want = float(centroid @ v / (np.linalg.norm(centroid) * np.linalg.norm(v)))
        assert abs(got - want) < 1e-6, oa
    c.close()
    assert after != before
    assert after["W2000007"] == max(after.values())    # the new seed is now the closest paper


def test_a_saved_interests_topics_can_be_previewed_and_switched(app, tmp_path):
    """Topics were fixed after setup: no route saved an on/off choice for a
    live interest, and recompute-topics wrote as it read, switching every
    newly found topic on before the user had seen it."""
    import json as _json

    def stored():
        c = connect(tmp_path / "radar.db")
        try:
            tf = _json.loads(c.execute(
                "SELECT topic_filters_json FROM profiles WHERE slug = ?", (SLUG,)).fetchone()[0])
        finally:
            c.close()
        return {t["id"]: t.get("on", True) for t in tf["topics"]}

    # As if saved before the seeds brought in T10123.
    c = connect(tmp_path / "radar.db")
    c.execute("UPDATE profiles SET topic_filters_json = ? WHERE slug = ?", (_json.dumps(
        {"topics": [{"id": "T11431", "display_name": "Provenance", "count": 11, "on": True}]}), SLUG))
    c.commit()
    c.close()

    resp = _request(app, "GET", f"/api/profiles/{SLUG}/topics/preview")
    assert resp.status_code == 200, resp.text
    topics = {t["id"]: t for t in resp.json()}
    assert topics["T11431"]["on"] is True and topics["T11431"]["new"] is False
    assert topics["T10123"]["on"] is False and topics["T10123"]["new"] is True   # new starts off
    assert stored() == {"T11431": True}                                          # nothing written

    resp = _request(app, "PUT", f"/api/profiles/{SLUG}/topics", json={"selected_topic_ids": ["T10123"]})
    assert resp.status_code == 200, resp.text
    assert stored() == {"T11431": False, "T10123": True}

    resp = _request(app, "PUT", f"/api/profiles/{SLUG}/topics", json={"selected_topic_ids": []})
    assert resp.status_code == 409
    assert stored() == {"T11431": False, "T10123": True}                         # unchanged
