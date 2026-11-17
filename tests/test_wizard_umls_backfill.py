"""Building a topic list reads no seed (review U5).

The seeds are read before the list is asked for: step 3 and the edit
page's Topics step POST /umls and wait (that also sends data made by
earlier rules back to extraction), an upload reads its own at once, and
the umls-backfill job picks up any other. The list used to read them
itself, up to 30 s a request: on an interest saved before CTX, the edit
page's list and every switch took ~33 s each while the seeds were read a
batch at a time, and the list grew under the user's clicks. A seed not
read yet is pending, and the stored UMLS topics are kept beside it.

The extractor is a stub: what matters is that nothing calls it here.
"""

from __future__ import annotations

import json

import pytest

from rag_lib.api.services import wizard as wizard_service
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import (
    papers as papers_repo,
    profiles as profiles_repo,
    users as users_repo,
)

_EMPTY = {"topics": [], "subfields": [], "fields": [], "domains": []}


@pytest.fixture
def db(tmp_path, monkeypatch):
    from rag_lib.api import settings as settings_mod

    monkeypatch.setenv("RADAR_UMLS_ENABLED", "true")
    monkeypatch.setenv("RADAR_UMLS_CACHE_DIR", str(tmp_path / "umls_cache"))
    settings_mod.get_settings.cache_clear()
    conn = connect(tmp_path / "radar.db")
    apply_migrations(conn)
    user = users_repo.upsert(conn, "demo@example.com")
    pid = profiles_repo.upsert(
        conn, user_id=int(user["id"]), name="p",
        embedding_model="placeholder-v1", n_seed=0, topic_filters={},
    )
    called: list[str] = []
    monkeypatch.setattr("rag_lib.api.services.vault._try_extract_umls",
                        lambda c, s, oid, text: called.append(oid))
    forgot: list[list[str]] = []
    monkeypatch.setattr(wizard_service, "forget_stale_umls_mappings",
                        lambda c, ids: forgot.append(list(ids)))
    yield conn, int(pid), called, forgot
    conn.close()
    settings_mod.get_settings.cache_clear()


def _seed(conn, profile_id, oid, *, concepts=None, mapped=None):
    papers_repo.upsert(conn, {"openalex_id": oid, "title": "A Title", "abstract": "an abstract",
                              "body_text": "body text", "source": "user_pdf"})
    conn.execute("UPDATE papers SET umls_concepts_json = ?, umls_mapped_topics_json = ? WHERE openalex_id = ?",
                 (None if concepts is None else json.dumps(concepts),
                  None if mapped is None else json.dumps(mapped), oid))
    conn.commit()
    profiles_repo.attach_seed(conn, profile_id, oid)
    conn.commit()


def test_a_seed_not_read_yet_is_pending_not_read_here(db):
    conn, pid, called, forgot = db
    _seed(conn, pid, "W_PENDING")
    wizard_service._try_merge_umls_topics(conn, pid, dict(_EMPTY))
    assert called == [] and forgot == []
    state = wizard_service.umls_topics_state(pid)
    assert state["available"] is False and state["reason"] == "pending" and state["n"] == 1


def test_data_made_by_earlier_rules_is_left_as_it_is_by_a_build(db):
    """Sent back to extraction by POST /umls only: a build used to NULL it
    and then read it inside the request."""
    conn, pid, called, forgot = db
    old = [{"topic_id": "https://openalex.org/T9", "display_name": "Nine", "similarity": 0.9,
            "source_cui": "C1", "source_name": "c", "confidence": 0.95, "mapping": "context-1"}]
    _seed(conn, pid, "W_OLD", concepts=[{"cui": "C1", "name": "c", "confidence": 0.95}], mapped=old)
    wizard_service._try_merge_umls_topics(conn, pid, dict(_EMPTY))
    assert called == [] and forgot == []
    row = conn.execute("SELECT umls_mapped_topics_json FROM papers WHERE openalex_id = 'W_OLD'").fetchone()
    assert json.loads(row[0]) == old


def test_nothing_runs_when_umls_is_disabled(db, monkeypatch):
    conn, pid, called, forgot = db
    _seed(conn, pid, "W1")
    monkeypatch.setenv("RADAR_UMLS_ENABLED", "false")
    from rag_lib.api import settings as settings_mod
    settings_mod.get_settings.cache_clear()
    out = wizard_service._try_merge_umls_topics(conn, pid, dict(_EMPTY))
    assert called == [] and forgot == []
    assert out == _EMPTY
