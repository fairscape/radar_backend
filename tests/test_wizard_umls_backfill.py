"""Step 3 computes UMLS data for seeds whose extraction has not landed.

``vault.upload`` starts extraction on a daemon thread and returns, but
the wizard reads ``papers.umls_mapped_topics_json`` synchronously a few
seconds later — the user uploads in step 1 and clicks through to step 3.
The column was still NULL, so step 3 showed only the OpenAlex topics and
said nothing about the half that was still being computed.

These tests stub the extractor: what matters here is which papers are
handed to it and what text they are handed, not what SciSpacy does with
it.
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


class _Settings:
    RADAR_UMLS_ENABLED = True
    RADAR_UMLS_MAX_TOPIC_ADDITIONS = 10
    RADAR_UMLS_CACHE_DIR = "/nonexistent"


@pytest.fixture
def db(tmp_path):
    conn = connect(tmp_path / "radar.db")
    apply_migrations(conn)
    user = users_repo.upsert(conn, "demo@example.com")
    pid = profiles_repo.upsert(
        conn, user_id=int(user["id"]), name="p",
        embedding_model="placeholder-v1", n_seed=0, topic_filters={},
    )
    yield conn, int(pid)
    conn.close()


def _seed(conn, profile_id, oid, *, concepts=None, title="A Title",
          abstract="an abstract", body="body text"):
    papers_repo.upsert(conn, {
        "openalex_id": oid, "title": title, "abstract": abstract,
        "body_text": body, "source": "user_pdf",
    })
    if concepts is not None:
        conn.execute(
            "UPDATE papers SET umls_concepts_json = ? WHERE openalex_id = ?",
            (json.dumps(concepts), oid),
        )
    conn.commit()
    profiles_repo.attach_seed(conn, profile_id, oid)
    conn.commit()


def test_a_seed_with_no_concepts_is_extracted_on_demand(db, monkeypatch):
    conn, pid = db
    _seed(conn, pid, "W_PENDING")

    called: list[str] = []
    monkeypatch.setattr(
        "rag_lib.api.services.vault._try_extract_umls",
        lambda c, s, oid, text: called.append(oid),
    )
    wizard_service._extract_missing_umls(conn, _Settings(), ["W_PENDING"])

    assert called == ["W_PENDING"]


def test_a_seed_that_already_has_concepts_is_left_alone(db, monkeypatch):
    """Re-running the linker costs seconds a paper and changes nothing."""
    conn, pid = db
    _seed(conn, pid, "W_DONE", concepts=[{"cui": "C0011860"}])

    called: list[str] = []
    monkeypatch.setattr(
        "rag_lib.api.services.vault._try_extract_umls",
        lambda c, s, oid, text: called.append(oid),
    )
    wizard_service._extract_missing_umls(conn, _Settings(), ["W_DONE"])

    assert called == []


def test_only_the_pending_seeds_are_extracted(db, monkeypatch):
    conn, pid = db
    _seed(conn, pid, "W_DONE", concepts=[{"cui": "C1"}])
    _seed(conn, pid, "W_PENDING_A")
    _seed(conn, pid, "W_PENDING_B")

    called: list[str] = []
    monkeypatch.setattr(
        "rag_lib.api.services.vault._try_extract_umls",
        lambda c, s, oid, text: called.append(oid),
    )
    wizard_service._extract_missing_umls(
        conn, _Settings(), ["W_DONE", "W_PENDING_A", "W_PENDING_B"],
    )

    assert called == ["W_PENDING_A", "W_PENDING_B"]


def test_the_text_handed_to_the_extractor_leads_with_the_title(db, monkeypatch):
    """Same selection rule as the upload path — the title first, because
    an abstract that only ever says "T2DM" yields no diabetes concept."""
    conn, pid = db
    _seed(conn, pid, "W1",
          title="Early detection of type 2 diabetes mellitus",
          abstract="T2DM screening with ML")

    seen: dict = {}
    monkeypatch.setattr(
        "rag_lib.api.services.vault._try_extract_umls",
        lambda c, s, oid, text: seen.update(text=text),
    )
    wizard_service._extract_missing_umls(conn, _Settings(), ["W1"])

    assert seen["text"].startswith("Early detection of type 2 diabetes mellitus")
    assert "T2DM screening with ML" in seen["text"]


def test_nothing_runs_when_umls_is_disabled(db, monkeypatch):
    conn, pid = db
    _seed(conn, pid, "W1")

    called: list[str] = []
    monkeypatch.setattr(
        "rag_lib.api.services.vault._try_extract_umls",
        lambda c, s, oid, text: called.append(oid),
    )

    class _Off(_Settings):
        RADAR_UMLS_ENABLED = False

    monkeypatch.setattr(wizard_service, "get_settings", lambda: _Off(),
                        raising=False)
    monkeypatch.setattr("rag_lib.api.settings.get_settings", lambda: _Off())
    out = wizard_service._try_merge_umls_topics(
        conn, pid, {"topics": [], "subfields": [], "fields": [], "domains": []},
    )

    assert called == []
    assert out == {"topics": [], "subfields": [], "fields": [], "domains": []}


def test_an_unknown_seed_id_does_not_raise(db, monkeypatch):
    conn, pid = db
    called: list[str] = []
    monkeypatch.setattr(
        "rag_lib.api.services.vault._try_extract_umls",
        lambda c, s, oid, text: called.append(oid),
    )
    wizard_service._extract_missing_umls(conn, _Settings(), ["W_MISSING"])
    assert called == []
