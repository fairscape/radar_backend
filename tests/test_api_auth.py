"""Auth — Phase 12.

Covers the three behaviors of ``get_current_user`` with the new
``X-User-Email`` header:

  - header absent + ``RADAR_REQUIRE_AUTH=False`` → falls back to user 1
  - header absent + ``RADAR_REQUIRE_AUTH=True``  → 401
  - header present                                → upsert + 200
  - two emails see disjoint profiles (per-user isolation)

Plus the new ``/api/users/me`` GET + PATCH endpoints.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from rag_lib.api import settings as settings_module
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import (
    profiles as profiles_repo,
    users as users_repo,
)


def _seed_db(db_path) -> None:
    conn = connect(db_path)
    apply_migrations(conn)
    conn.close()


@pytest.fixture()
def app(tmp_path, monkeypatch):
    db = tmp_path / "radar.db"
    monkeypatch.setenv("RADAR_DB_PATH", str(db))
    monkeypatch.setenv("RADAR_VAULT_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("RADAR_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("RADAR_LOG_JSON", "false")
    monkeypatch.setenv("RADAR_SCHEDULER_ENABLED", "false")
    monkeypatch.setenv("RADAR_REQUIRE_AUTH", "false")
    settings_module.get_settings.cache_clear()
    _seed_db(db)

    from rag_lib.api.app import create_app

    yield create_app()
    settings_module.get_settings.cache_clear()


@pytest.fixture()
def app_auth_required(tmp_path, monkeypatch):
    db = tmp_path / "radar.db"
    monkeypatch.setenv("RADAR_DB_PATH", str(db))
    monkeypatch.setenv("RADAR_VAULT_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("RADAR_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("RADAR_LOG_JSON", "false")
    monkeypatch.setenv("RADAR_SCHEDULER_ENABLED", "false")
    monkeypatch.setenv("RADAR_REQUIRE_AUTH", "true")
    settings_module.get_settings.cache_clear()
    _seed_db(db)

    from rag_lib.api.app import create_app

    yield create_app()
    settings_module.get_settings.cache_clear()


def _request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(method, path, **kwargs)
    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Header behavior
# ---------------------------------------------------------------------------


def test_no_header_fallback_to_demo_when_auth_optional(app):
    resp = _request(app, "GET", "/api/users/me")
    assert resp.status_code == 200, resp.text
    assert resp.json()["email"] == "demo@example.com"


def test_no_header_401_when_auth_required(app_auth_required):
    resp = _request(app_auth_required, "GET", "/api/users/me")
    assert resp.status_code == 401
    assert "X-User-Email" in resp.json()["detail"]


def test_header_upserts_unknown_email(app_auth_required, tmp_path):
    resp = _request(
        app_auth_required, "GET", "/api/users/me",
        headers={"X-User-Email": "alice@example.com"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["email"] == "alice@example.com"
    assert body["mailto"] == "alice@example.com"

    # The upsert is durable — the row is now in the DB.
    conn = connect(tmp_path / "radar.db")
    try:
        row = users_repo.get_by_email(conn, "alice@example.com")
        assert row is not None
    finally:
        conn.close()


def test_header_lowercased_before_upsert(app_auth_required):
    resp = _request(
        app_auth_required, "GET", "/api/users/me",
        headers={"X-User-Email": "Alice@Example.COM"},
    )
    assert resp.status_code == 200
    assert resp.json()["email"] == "alice@example.com"


def test_invalid_email_400(app_auth_required):
    resp = _request(
        app_auth_required, "GET", "/api/users/me",
        headers={"X-User-Email": "not-an-email"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Per-user isolation
# ---------------------------------------------------------------------------


def test_two_emails_see_disjoint_profiles(app_auth_required, tmp_path):
    # Alice creates a draft, then commits.
    resp = _request(
        app_auth_required, "POST", "/api/profiles/draft",
        headers={"X-User-Email": "alice@example.com"},
        json={"name": "Alice profile"},
    )
    assert resp.status_code == 200, resp.text
    alice_slug = resp.json()["slug"]

    # Manually flip the draft to live so it shows up in Alice's list.
    conn = connect(tmp_path / "radar.db")
    try:
        alice = users_repo.get_by_email(conn, "alice@example.com")
        assert alice is not None
        row = profiles_repo.get_by_slug(conn, int(alice["id"]), alice_slug)
        assert row is not None
        conn.execute(
            "UPDATE profiles SET is_draft = 0 WHERE id = ?",
            (int(row["id"]),),
        )
        conn.commit()
    finally:
        conn.close()

    # Alice sees her profile.
    resp = _request(
        app_auth_required, "GET", "/api/profiles",
        headers={"X-User-Email": "alice@example.com"},
    )
    assert resp.status_code == 200
    keys = [p["key"] for p in resp.json()]
    assert alice_slug in keys

    # Bob, in a separate request, sees an empty list.
    resp = _request(
        app_auth_required, "GET", "/api/profiles",
        headers={"X-User-Email": "bob@example.com"},
    )
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# /api/users/me PATCH
# ---------------------------------------------------------------------------


def test_patch_me_updates_mailto(app_auth_required):
    headers = {"X-User-Email": "alice@example.com"}
    _request(app_auth_required, "GET", "/api/users/me", headers=headers)
    resp = _request(
        app_auth_required, "PATCH", "/api/users/me",
        headers=headers,
        json={"mailto": "alice+lab@example.com"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["mailto"] == "alice+lab@example.com"


def test_patch_me_rejects_garbage_mailto(app_auth_required):
    headers = {"X-User-Email": "alice@example.com"}
    _request(app_auth_required, "GET", "/api/users/me", headers=headers)
    resp = _request(
        app_auth_required, "PATCH", "/api/users/me",
        headers=headers,
        json={"mailto": "not an email"},
    )
    assert resp.status_code == 400


def test_health_unaffected_by_auth(app_auth_required):
    resp = _request(app_auth_required, "GET", "/api/health")
    assert resp.status_code == 200
    resp = _request(app_auth_required, "GET", "/health")
    assert resp.status_code == 200
