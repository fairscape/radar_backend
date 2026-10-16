"""Step 3 waits for UMLS extraction over every seed before listing topics."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from rag_lib.api import settings as settings_module
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import papers as papers_repo, profiles as profiles_repo, users as users_repo


@pytest.fixture()
def app(tmp_path, monkeypatch):
    db = tmp_path / "radar.db"
    for k, v in {"RADAR_DB_PATH": str(db), "RADAR_VAULT_DIR": str(tmp_path / "vault"),
                 "RADAR_CHROMA_DIR": str(tmp_path / "chroma"), "RADAR_LOG_JSON": "false",
                 "RADAR_SCHEDULER_ENABLED": "false", "RADAR_UMLS_ENABLED": "true",
                 "RADAR_DEFAULT_EMBEDDING_MODEL": "placeholder-v1"}.items():
        monkeypatch.setenv(k, v)
    settings_module.get_settings.cache_clear()
    conn = connect(db)
    apply_migrations(conn)
    uid = int(users_repo.upsert(conn, "demo@example.com")["id"])
    pid = profiles_repo.upsert(conn, user_id=uid, name="Chromatin", embedding_model="placeholder-v1",
                               n_seed=3, topic_filters={"topics": []}, threshold=0.8)
    for i in range(3):
        papers_repo.upsert(conn, {"openalex_id": f"W{i}", "title": f"Paper {i}", "abstract": "a", "source": "orcid"})
        profiles_repo.attach_seed(conn, pid, f"W{i}")
    conn.close()

    from rag_lib.api.services import vault

    def fake_extract(conn, settings, oa, text):
        if oa == "W2":
            return False                                  # the linker failed on this one
        conn.execute("UPDATE papers SET umls_concepts_json = '[]' WHERE openalex_id = ?", (oa,))
        conn.commit()
        return True

    monkeypatch.setattr(vault, "_try_extract_umls_unlocked", fake_extract)
    monkeypatch.setattr(vault, "_UMLS_FAILED", {})
    from rag_lib.umls import extractor
    monkeypatch.setattr(extractor, "extract_umls_concepts", lambda *a, **k: [])   # skip the 4-min warmup
    from rag_lib.api.app import create_app
    yield create_app()
    settings_module.get_settings.cache_clear()


def _request(app, method, path, **kw):
    async def _run():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
                return await c.request(method, path, **kw)
    return asyncio.run(_run())


def test_status_then_extraction_leaves_nothing_pending_and_counts_the_failure(app):
    before = _request(app, "GET", "/api/profiles/chromatin/umls").json()
    assert (before["total"], before["pending"], before["done"]) == (3, 3, 0)

    after = _request(app, "POST", "/api/profiles/chromatin/umls").json()
    assert (after["done"], after["pending"], after["failed"]) == (2, 0, 1)
    assert after["running"] is False
    # Asking again does not re-run the failing seed during its back-off.
    assert _request(app, "POST", "/api/profiles/chromatin/umls").json()["failed"] == 1


def test_a_merge_that_cannot_run_keeps_the_stored_umls_topics_and_says_so(app, tmp_path, monkeypatch):
    import json
    import sqlite3

    from rag_lib.api.services import wizard

    _request(app, "POST", "/api/profiles/chromatin/umls")
    conn = connect(tmp_path / "radar.db")
    stored = {"topics": [{"id": "https://openalex.org/T9", "display_name": "Kept UMLS topic",
                          "count": 1, "on": False, "source": "umls"}]}
    conn.execute("UPDATE profiles SET topic_filters_json = ?", (json.dumps(stored),))
    conn.commit()
    conn.close()

    def locked(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(wizard, "_merge_umls_topics", locked)
    topics = _request(app, "POST", "/api/profiles/chromatin/recompute-topics").json()
    kept = [t for t in topics if t["id"] == "https://openalex.org/T9"]
    assert kept and kept[0]["on"] is False and kept[0]["source"] == "umls"   # kept, still switched off
    st = _request(app, "GET", "/api/profiles/chromatin/umls").json()
    assert st["topics_available"] is False and st["topics_reason"] == "busy"


def test_an_unreadable_seed_is_sent_back_to_extraction_and_reported(app, tmp_path):
    _request(app, "POST", "/api/profiles/chromatin/umls")
    conn = connect(tmp_path / "radar.db")
    conn.execute("UPDATE papers SET umls_concepts_json = '[]', umls_mapped_topics_json = '{not json' "
                 "WHERE openalex_id = 'W0'")
    conn.commit()
    conn.close()
    _request(app, "POST", "/api/profiles/chromatin/recompute-topics")
    st = _request(app, "GET", "/api/profiles/chromatin/umls").json()
    assert (st["topics_available"], st["topics_reason"], st["topics_n"]) == (False, "bad_data", 1)
    conn = connect(tmp_path / "radar.db")
    assert conn.execute("SELECT umls_concepts_json FROM papers WHERE openalex_id = 'W0'").fetchone()[0] is None
    conn.close()
