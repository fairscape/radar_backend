"""Smoke test for the FastAPI skeleton.

Boots the app in-process via ``httpx.ASGITransport`` (no network port,
no uvicorn) and asserts the health endpoints. Migrations run against a
tmp DB so the test never touches ``data/radar.db``.

Tests are written sync and drive the AsyncClient via ``asyncio.run`` so
the suite doesn't need ``pytest-asyncio``.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from rag_lib.api import settings as settings_module


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("RADAR_DB_PATH", str(tmp_path / "radar.db"))
    monkeypatch.setenv("RADAR_VAULT_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("RADAR_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("RADAR_LOG_JSON", "false")
    settings_module.get_settings.cache_clear()

    from rag_lib.api.app import create_app  # import after env is set

    app = create_app()
    yield app

    settings_module.get_settings.cache_clear()


def _get(app, path: str) -> httpx.Response:
    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.get(path)
    return asyncio.run(_run())


def test_api_health_returns_ok(app):
    resp = _get(app, "/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body and isinstance(body["version"], str)


def test_root_health_returns_ok(app):
    resp = _get(app, "/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_chat_history_returns_empty_for_fresh_user(app):
    # Phase 9 implemented chat history; an empty vault means no turns yet.
    resp = _get(app, "/api/chat/history")
    assert resp.status_code == 200
    assert resp.json() == []
