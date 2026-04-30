"""Phase 8 — feedback log writer + DB mirror.

Round-trip: triggering a save event from the API
  - writes one JSON line to {vault}/{user_id}/state/feedback.jsonl
  - inserts one row into feedback_events
  - the line + row reflect the same timestamp
  - the profile detail panel surfaces a formatted log line

Also covers ``selector_config_hash`` stability under key reordering.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

import httpx
import pytest

from rag_lib.api import settings as settings_module
from rag_lib.feedback.log import (
    FeedbackEvent,
    log_event,
    selector_config_hash,
    tail_jsonl,
)
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
    app.state.test_tmp_path = tmp_path
    app.state.test_db = db
    yield app
    settings_module.get_settings.cache_clear()


def _request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.request(method, path, **kwargs)
    return asyncio.run(_run())


def _read_db(db_path) -> sqlite3.Connection:
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    return c


# ---------------------------------------------------------------------------
# selector_config_hash
# ---------------------------------------------------------------------------


def test_selector_config_hash_is_16_chars_hex():
    h = selector_config_hash({"type": "centroid", "threshold": 0.85})
    assert isinstance(h, str)
    assert len(h) == 16
    int(h, 16)  # parses as hex


def test_selector_config_hash_is_stable_under_key_order():
    a = selector_config_hash({"type": "centroid", "threshold": 0.85, "embedding_model": "v1"})
    b = selector_config_hash({"embedding_model": "v1", "threshold": 0.85, "type": "centroid"})
    assert a == b


def test_selector_config_hash_changes_with_threshold():
    a = selector_config_hash({"type": "centroid", "threshold": 0.85})
    b = selector_config_hash({"type": "centroid", "threshold": 0.84})
    assert a != b


# ---------------------------------------------------------------------------
# log_event direct
# ---------------------------------------------------------------------------


def test_log_event_writes_jsonl_and_db_row(tmp_path):
    db_path = tmp_path / "radar.db"
    _seed_db(db_path)
    conn = _read_db(db_path)

    event = FeedbackEvent(
        profile_id=1, profile_slug="provenance-fairscape",
        openalex_id="W2000000", action="saved",
        score=0.91, selector="CentroidSelector",
        selector_config_hash="abc123def4567890",
        doi="10.1234/foo.0",
    )
    stamped = log_event(tmp_path / "vault", 1, conn, event)
    assert stamped.timestamp is not None

    jsonl = tmp_path / "vault" / "1" / "state" / "feedback.jsonl"
    assert jsonl.exists()
    lines = jsonl.read_text().strip().splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["action"] == "saved"
    assert obj["openalex_id"] == "W2000000"
    assert obj["score"] == pytest.approx(0.91)
    assert obj["timestamp"] == stamped.timestamp

    row = conn.execute(
        "SELECT * FROM feedback_events WHERE openalex_id = ?",
        ("W2000000",),
    ).fetchone()
    assert row is not None
    assert row["action"] == "saved"
    assert row["score"] == pytest.approx(0.91)
    assert row["selector"] == "CentroidSelector"
    assert row["ts"] == stamped.timestamp
    conn.close()


def test_log_event_skips_disallowed_actions(tmp_path):
    db_path = tmp_path / "radar.db"
    _seed_db(db_path)
    conn = _read_db(db_path)

    event = FeedbackEvent(
        profile_id=1, profile_slug="provenance-fairscape",
        openalex_id="W2000000", action="shown",
        score=0.5,
    )
    log_event(tmp_path / "vault", 1, conn, event)

    jsonl = tmp_path / "vault" / "1" / "state" / "feedback.jsonl"
    assert not jsonl.exists()
    n = conn.execute("SELECT count(*) AS n FROM feedback_events").fetchone()["n"]
    assert n == 0
    conn.close()


def test_tail_jsonl_returns_last_n_lines(tmp_path):
    p = tmp_path / "feedback.jsonl"
    p.write_text("\n".join(f"line-{i}" for i in range(5)) + "\n")
    assert tail_jsonl(p, n=3) == ["line-2", "line-3", "line-4"]
    assert tail_jsonl(p, n=10) == [f"line-{i}" for i in range(5)]
    assert tail_jsonl(tmp_path / "missing.jsonl") == []


# ---------------------------------------------------------------------------
# End-to-end: API save → JSONL line + DB row + profile detail
# ---------------------------------------------------------------------------


def test_save_writes_feedback_event(app):
    daily = _request(app, "GET", "/api/radar/daily").json()
    card = daily["cards"][0]
    card_id = card["id"]
    expected_score = card["score"]

    r = _request(app, "POST", "/api/radar/cards/save", json={"card_id": card_id})
    assert r.status_code == 200, r.text
    assert r.json()["state"] == "saved"

    jsonl = app.state.test_tmp_path / "vault" / "1" / "state" / "feedback.jsonl"
    assert jsonl.exists(), "expected JSONL file to exist after save"
    lines = jsonl.read_text().strip().splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["action"] == "saved"
    assert obj["openalex_id"] == card_id
    assert obj["profile"] == "provenance-fairscape"
    assert obj["score"] == pytest.approx(expected_score)

    conn = _read_db(app.state.test_db)
    row = conn.execute(
        "SELECT action, score, openalex_id FROM feedback_events ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    assert row["action"] == "saved"
    assert row["openalex_id"] == card_id
    assert row["score"] == pytest.approx(expected_score)


def test_dismiss_writes_feedback_event(app):
    daily = _request(app, "GET", "/api/radar/daily").json()
    card_id = daily["cards"][1]["id"]

    r = _request(app, "POST", "/api/radar/cards/dismiss", json={"card_id": card_id})
    assert r.status_code == 200
    assert r.json()["state"] == "dismissed"

    conn = _read_db(app.state.test_db)
    row = conn.execute(
        "SELECT action FROM feedback_events ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    assert row["action"] == "dismissed"


def test_save_toggle_off_does_not_log_extra_event(app):
    daily = _request(app, "GET", "/api/radar/daily").json()
    card_id = daily["cards"][2]["id"]

    _request(app, "POST", "/api/radar/cards/save", json={"card_id": card_id})  # saved → 1 row
    _request(app, "POST", "/api/radar/cards/save", json={"card_id": card_id})  # toggled off → 0 added

    conn = _read_db(app.state.test_db)
    n = conn.execute(
        "SELECT count(*) AS n FROM feedback_events WHERE openalex_id = ?",
        (card_id,),
    ).fetchone()["n"]
    conn.close()
    assert n == 1


def test_profile_detail_includes_recent_feedback(app):
    daily = _request(app, "GET", "/api/radar/daily").json()
    card_id = daily["cards"][0]["id"]
    _request(app, "POST", "/api/radar/cards/save", json={"card_id": card_id})

    detail = _request(app, "GET", "/api/profiles/provenance-fairscape/detail").json()
    assert len(detail["feedbackLog"]) == 1
    line = detail["feedbackLog"][0]
    assert "SAVED" in line
    assert "score=" in line
    assert detail["feedbackMoreCount"] == 0


def test_feedback_endpoint_filters_by_action(app):
    daily = _request(app, "GET", "/api/radar/daily").json()
    saved_id = daily["cards"][0]["id"]
    dismissed_id = daily["cards"][1]["id"]

    _request(app, "POST", "/api/radar/cards/save", json={"card_id": saved_id})
    _request(app, "POST", "/api/radar/cards/dismiss", json={"card_id": dismissed_id})

    all_events = _request(app, "GET", "/api/profiles/provenance-fairscape/feedback").json()
    assert len(all_events) == 2

    only_saved = _request(
        app, "GET", "/api/profiles/provenance-fairscape/feedback?action=saved"
    ).json()
    assert len(only_saved) == 1
    assert only_saved[0]["action"] == "saved"
    assert only_saved[0]["openalex_id"] == saved_id


def test_feedback_endpoint_unknown_profile_404(app):
    r = _request(app, "GET", "/api/profiles/does-not-exist/feedback")
    assert r.status_code == 404
