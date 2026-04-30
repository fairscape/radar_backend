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
