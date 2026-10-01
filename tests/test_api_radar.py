"""Radar API tests.

Reuses the seed helper from test_api_profiles to stand up a tmp DB,
then exercises ``/api/radar/daily`` filter combinations and the
save/dismiss toggle.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from rag_lib.api import settings as settings_module
from tests.test_api_profiles import _seed_db


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


def test_daily_returns_cards_sorted_by_score(app):
    resp = _request(app, "GET", "/api/radar/daily")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["candidatesScored"] == 50
    assert isinstance(body["cards"], list) and len(body["cards"]) > 0
    scores = [c["score"] for c in body["cards"]]
    assert scores == sorted(scores, reverse=True)
    assert all({"id", "title", "score", "bucket", "profile"} <= set(c.keys()) for c in body["cards"])
    assert body["cards"][0]["profile"] == "provenance-fairscape"


def test_daily_bucket_filter(app):
    resp = _request(app, "GET", "/api/radar/daily?bucket=high")
    assert resp.status_code == 200, resp.text
    cards = resp.json()["cards"]
    # Seeded scores start at 0.95 and drop by 0.01; high bucket = >=0.85.
    assert all(c["bucket"] == "high" for c in cards)
    assert all(c["score"] >= 0.85 for c in cards)


def test_daily_profile_filter_unknown_returns_empty(app):
    resp = _request(app, "GET", "/api/radar/daily?profile=nope")
    assert resp.status_code == 200
    body = resp.json()
    assert body["cards"] == []


def test_save_toggle_round_trip(app):
    # First call → saved
    daily = _request(app, "GET", "/api/radar/daily").json()
    card_id = daily["cards"][0]["id"]
    r1 = _request(app, "POST", "/api/radar/cards/save", json={"card_id": card_id})
    assert r1.status_code == 200, r1.text
    assert r1.json()["state"] == "saved"
    # Second call → toggled back to None
    r2 = _request(app, "POST", "/api/radar/cards/save", json={"card_id": card_id})
    assert r2.json()["state"] is None


def test_save_then_dismiss_clears_save(app):
    daily = _request(app, "GET", "/api/radar/daily").json()
    card_id = daily["cards"][1]["id"]
    _request(app, "POST", "/api/radar/cards/save", json={"card_id": card_id})
    r = _request(app, "POST", "/api/radar/cards/dismiss", json={"card_id": card_id})
    assert r.status_code == 200
    assert r.json()["state"] == "dismissed"

    # Daily should now report this card as dismissed in the states map.
    daily2 = _request(app, "GET", "/api/radar/daily").json()
    assert daily2["states"].get(card_id) == "dismissed"


def test_save_unknown_card_404(app):
    r = _request(app, "POST", "/api/radar/cards/save", json={"card_id": "W9999999"})
    assert r.status_code == 404


# --- one paper, two interests --------------------------------------------------
#
# A card is a paper *in an interest*. Save/dismiss used to resolve the paper
# alone to whichever interest fetched it last, and the feed's ``states`` was
# keyed by paper id, so with a paper in two interests the card on screen,
# the state it showed and the row a click changed could all differ.

PAPER = "W2000000"


def _second_interest(tmp_path, *, saved_in_first: bool = False) -> None:
    from rag_lib.db import connect
    from rag_lib.db.repos import candidates as candidates_repo
    from rag_lib.db.repos import gather_runs as gather_runs_repo
    from rag_lib.db.repos import profiles as profiles_repo

    conn = connect(tmp_path / "radar.db")
    user_id = conn.execute("SELECT id FROM users WHERE email = 'demo@example.com'").fetchone()[0]
    pid = profiles_repo.upsert(conn, user_id=user_id, name="other interest",
                               embedding_model="placeholder-v1", n_seed=1, threshold=0.85)
    run = gather_runs_repo.start(conn, profile_id=pid, user_id=user_id, since_date="2026-03-26",
                                 filter_string="x", tier_used="must-have-AND")
    candidates_repo.insert_dedup(conn, profile_id=pid, openalex_id=PAPER, score=0.99,
                                 tier_used="must-have-AND", gather_run_id=run)
    if saved_in_first:
        conn.execute("UPDATE profile_candidates SET saved_at = datetime('now') "
                     "WHERE openalex_id = ? AND profile_id = (SELECT id FROM profiles WHERE slug = 'provenance-fairscape')",
                     (PAPER,))
        conn.commit()
    conn.close()


def _saved(tmp_path) -> dict:
    from rag_lib.db import connect

    conn = connect(tmp_path / "radar.db")
    rows = conn.execute(
        "SELECT p.slug, pc.saved_at IS NOT NULL FROM profile_candidates pc "
        "JOIN profiles p ON p.id = pc.profile_id WHERE pc.openalex_id = ?", (PAPER,)).fetchall()
    conn.close()
    return {slug: bool(s) for slug, s in rows}


def test_save_lands_on_the_interest_the_card_was_shown_under(app, tmp_path):
    _second_interest(tmp_path)
    resp = _request(app, "POST", "/api/radar/cards/save",
                    json={"card_id": PAPER, "profile": "provenance-fairscape"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["state"] == "saved"
    # "other interest" fetched the paper later; it must be left alone.
    assert _saved(tmp_path) == {"provenance-fairscape": True, "other-interest": False}


def test_saving_in_one_interest_never_unsaves_it_in_another(app, tmp_path):
    _second_interest(tmp_path, saved_in_first=True)
    resp = _request(app, "POST", "/api/radar/cards/save",
                    json={"card_id": PAPER, "profile": "other-interest"})
    assert resp.json()["state"] == "saved"
    assert _saved(tmp_path) == {"provenance-fairscape": True, "other-interest": True}


def test_each_card_carries_its_own_interests_state(app, tmp_path):
    _second_interest(tmp_path, saved_in_first=True)
    cards = [c for c in _request(app, "GET", "/api/radar/daily").json()["cards"] if c["id"] == PAPER]
    assert {c["profile"]: c["state"] for c in cards} == {
        "provenance-fairscape": "saved", "other-interest": None,
    }


def test_feed_cards_carry_the_stored_byline(app, tmp_path):
    """Card.authors was hardcoded to [] -- every card read 'Unknown authors'."""
    from rag_lib.db import connect

    conn = connect(tmp_path / "radar.db")
    conn.execute("UPDATE papers SET authors_json = ? WHERE openalex_id = ?",
                 ('["Ada Lovelace", "Alan Turing"]', PAPER))
    conn.commit()
    conn.close()
    card = next(c for c in _request(app, "GET", "/api/radar/daily").json()["cards"] if c["id"] == PAPER)
    assert card["authors"] == ["Ada Lovelace", "Alan Turing"]
