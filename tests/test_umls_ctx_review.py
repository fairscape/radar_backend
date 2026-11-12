"""Review 2026-11-11 of CTX (FUNCTIONAL_SPEC CTX-6, CTX-R1/R5): without the
rich topic index nothing is lost and nothing loops; a failed extraction
waits out its back-off whoever asked for it; vault maps with the topic
confidence setting."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import papers as papers_repo
from rag_lib.umls import cache


def _write_rich(d, model="mxbai-embed-large"):
    d.mkdir(parents=True, exist_ok=True)
    (d / cache.RICH_TOPICS_FILE).write_text(json.dumps([{"id": "https://openalex.org/T1", "display_name": "One"}]))
    np.savez_compressed(d / cache.RICH_EMBEDDINGS_FILE, embeddings=np.ones((1, 4), dtype=np.float32))
    (d / cache.RICH_META_FILE).write_text(json.dumps({"embedding_model": model, "dim": 4}))


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(cache, "_rich_index", None)


def test_rich_index_problem(tmp_path, fresh, monkeypatch):
    assert "FileNotFoundError" in cache.rich_index_problem(tmp_path, "mxbai-embed-large")
    _write_rich(tmp_path)
    assert cache.rich_index_problem(tmp_path, "mxbai-embed-large") is None
    monkeypatch.setattr(cache, "_rich_index", None)
    assert "TopicIndexMismatch" in cache.rich_index_problem(tmp_path, "other-model")


@pytest.fixture
def db(tmp_path, monkeypatch):
    from rag_lib.api import settings as settings_mod

    monkeypatch.setenv("RADAR_UMLS_ENABLED", "true")
    monkeypatch.setenv("RADAR_UMLS_CACHE_DIR", str(tmp_path / "umls_cache"))
    settings_mod.get_settings.cache_clear()
    conn = connect(tmp_path / "radar.db")
    apply_migrations(conn)
    papers_repo.upsert(conn, {"openalex_id": "W1", "title": "T", "abstract": "A.", "source": "s"})
    old = (json.dumps([{"cui": "C1", "name": "x"}]), json.dumps([{"topic_id": "T9", "similarity": 0.9}]))
    conn.execute("UPDATE papers SET umls_concepts_json = ?, umls_mapped_topics_json = ?", old)
    conn.commit()
    yield conn, old, tmp_path / "umls_cache"
    conn.close()
    settings_mod.get_settings.cache_clear()


def _umls(conn):
    return tuple(conn.execute("SELECT umls_concepts_json, umls_mapped_topics_json FROM papers").fetchone())


def test_old_data_is_kept_while_there_is_no_rich_index(db, fresh, monkeypatch):
    from rag_lib.api.services import wizard

    conn, old, cache_dir = db
    wizard.forget_stale_umls_mappings(conn, ["W1"])
    assert _umls(conn) == old              # made again it could not be mapped
    _write_rich(cache_dir)
    monkeypatch.setattr(cache, "_rich_index", None)
    wizard.forget_stale_umls_mappings(conn, ["W1"])
    assert _umls(conn) == (None, None)     # with the index: sent back once


SETTINGS = SimpleNamespace(
    RADAR_UMLS_ENABLED=True, RADAR_UMLS_MIN_CONFIDENCE=0.7, RADAR_UMLS_TOPIC_MIN_CONFIDENCE=0.93,
    RADAR_UMLS_SPACY_MODEL="x", RADAR_UMLS_CACHE_DIR="/nonexistent", RADAR_UMLS_MAX_CONCEPTS=30,
    RADAR_UMLS_EMBEDDING_MODEL="x",
)


@pytest.fixture
def paper(tmp_path, monkeypatch):
    from rag_lib.api.services import vault
    from rag_lib.umls import extractor, topic_mapper
    from rag_lib.umls.extractor import UmlsConcept

    conn = connect(tmp_path / "r.db")
    apply_migrations(conn)
    papers_repo.upsert(conn, {"openalex_id": "W1", "title": "T", "abstract": "A.", "source": "s"})
    conn.commit()
    monkeypatch.setattr(vault, "_UMLS_FAILED", {})
    concept = UmlsConcept("C1", "x", "T047", "Disease or Syndrome", 0.95, (("x", "x is."),))
    monkeypatch.setattr(extractor, "extract_umls_concepts", lambda text, **kw: [concept])
    state = {"fail": True, "calls": [], "kw": None}

    def mapper(concepts, **kw):
        state["calls"].append(1)
        state["kw"] = kw
        if state["fail"]:
            raise FileNotFoundError("Topic index not found")
        return []
    monkeypatch.setattr(topic_mapper, "map_concepts_to_topics", mapper)
    yield conn, state
    conn.close()


def test_vault_maps_with_the_topic_confidence_setting(paper):
    from rag_lib.api.services import vault

    conn, state = paper
    state["fail"] = False
    assert vault._try_extract_umls(conn, SETTINGS, "W1", "text") is True
    assert state["kw"]["min_confidence"] == SETTINGS.RADAR_UMLS_TOPIC_MIN_CONFIDENCE


def test_a_failed_extraction_waits_out_its_back_off_whoever_asked(paper):
    from rag_lib.api.services import vault, wizard

    conn, state = paper
    assert vault._try_extract_umls(conn, SETTINGS, "W1", "text") is False
    assert _umls(conn) == (None, None)               # nothing half-written
    assert vault._recently_failed("W1")
    # step 3 asks again: not run again while it waits
    wizard._extract_missing_umls(conn, SETTINGS, ["W1"])
    assert len(state["calls"]) == 1
    assert vault.seed_umls_status(conn, SETTINGS, ["W1"])["failed"] == 1
    # a later success clears it
    state["fail"] = False
    assert vault._try_extract_umls(conn, SETTINGS, "W1", "text") is True
    assert not vault._recently_failed("W1")


def test_umls_switched_off_is_no_failure(paper):
    from rag_lib.api.services import vault

    conn, _ = paper
    off = SimpleNamespace(**{**vars(SETTINGS), "RADAR_UMLS_ENABLED": False})
    assert vault._try_extract_umls(conn, off, "W1", "text") is False
    assert not vault._recently_failed("W1")


def test_health_says_whether_the_index_is_usable(tmp_path, fresh, monkeypatch):
    from rag_lib.api.routers import health

    s = SimpleNamespace(RADAR_UMLS_ENABLED=True, RADAR_UMLS_CACHE_DIR=tmp_path,
                        RADAR_UMLS_EMBEDDING_MODEL="mxbai-embed-large")
    assert health._umls_topic_index(s) == "unusable"
    _write_rich(tmp_path)
    assert health._umls_topic_index(s) == "ok"
    assert health._umls_topic_index(SimpleNamespace(**{**vars(s), "RADAR_UMLS_ENABLED": False})) is None
