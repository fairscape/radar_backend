"""Seed UMLS backfill (vault.backfill_seed_umls, scheduler.backfill_seed_umls_job).

Seeds were extracted only when wizard step 3 opened, inside a 30-second
budget; what did not fit was left for "the next time step 3 is opened",
which never comes once the interest is committed. An 82-seed import got 6.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_lib.api.services import vault as vault_service
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import papers as papers_repo
from rag_lib.db.repos import profiles as profiles_repo
from rag_lib.db.repos import users as users_repo


@pytest.fixture()
def conn(tmp_path):
    c = connect(tmp_path / "radar.db")
    apply_migrations(c)
    uid = int(users_repo.upsert(c, "demo@example.com")["id"])
    pid = profiles_repo.upsert(c, user_id=uid, name="i", embedding_model="placeholder-v1", n_seed=0)
    for i in range(5):
        papers_repo.upsert(c, {"openalex_id": f"W{i}", "title": f"Seed {i} about insulin resistance",
                               "abstract": "type 2 diabetes", "source": "prosopia"})
        profiles_repo.attach_seed(c, pid, f"W{i}")
    papers_repo.upsert(c, {"openalex_id": "W_not_a_seed", "title": "not a seed", "source": "openalex_gatherer"})
    yield c
    c.close()


SETTINGS = SimpleNamespace(
    RADAR_UMLS_ENABLED=True, RADAR_UMLS_MIN_CONFIDENCE=0.7, RADAR_UMLS_SPACY_MODEL="x",
    RADAR_UMLS_CACHE_DIR="/nonexistent", RADAR_UMLS_MAX_CONCEPTS=30,
    RADAR_UMLS_MIN_TOPIC_SIMILARITY=0.4, RADAR_UMLS_EMBEDDING_MODEL="x",
)


@pytest.fixture()
def no_concepts(monkeypatch):
    """The real linker takes minutes to load; every paper yields nothing."""
    calls: list[str] = []
    import rag_lib.umls.extractor as extractor

    def fake(text, **kw):
        calls.append(text)
        return []
    monkeypatch.setattr(extractor, "extract_umls_concepts", fake)
    return calls


def _null(conn):
    return {r[0] for r in conn.execute("SELECT openalex_id FROM papers WHERE umls_concepts_json IS NULL")}


def test_backfill_reaches_seeds_only_and_respects_the_limit(conn, no_concepts):
    assert vault_service.backfill_seed_umls(conn, SETTINGS, limit=3) == 3
    assert len(no_concepts) == 3
    assert "W_not_a_seed" in _null(conn)


def test_a_paper_with_no_concepts_is_not_retried_forever(conn, no_concepts):
    """'Looked, found nothing' is stored as [], not left NULL -- NULL is
    what every caller reads as 'not extracted yet'."""
    vault_service.backfill_seed_umls(conn, SETTINGS, limit=40)
    assert len(no_concepts) == 5
    assert vault_service.backfill_seed_umls(conn, SETTINGS, limit=40) == 0
    assert len(no_concepts) == 5
    assert _null(conn) == {"W_not_a_seed"}


def test_a_scoped_backfill_touches_only_the_papers_named(conn, no_concepts):
    assert vault_service.backfill_seed_umls(conn, SETTINGS, openalex_ids=["W1", "W_not_a_seed"]) == 2
    assert "W0" in _null(conn) and "W1" not in _null(conn)


def test_disabled_umls_does_nothing(conn, no_concepts):
    off = SimpleNamespace(**{**vars(SETTINGS), "RADAR_UMLS_ENABLED": False})
    assert vault_service.backfill_seed_umls(conn, off) == 0
    assert no_concepts == []


@pytest.fixture()
def failing_for(monkeypatch):
    """Extraction raises for the ids given; succeeds (no concepts) otherwise."""
    import rag_lib.umls.extractor as extractor

    seen: list[str] = []
    bad: set[str] = set()

    def fake(text, **kw):
        seen.append(text)
        if any(b in text for b in bad):
            raise RuntimeError("topic embedder down")
        return []
    monkeypatch.setattr(extractor, "extract_umls_concepts", fake)
    monkeypatch.setattr(vault_service, "_UMLS_FAILED", {})
    return bad, seen


def test_papers_that_keep_failing_do_not_block_the_rest(conn, failing_for):
    """The sweep takes the first N NULL rows; a paper that keeps failing stays
    NULL, so the same ones were retried forever and later seeds never reached."""
    bad, _ = failing_for
    bad.update({"Seed 0", "Seed 1"})
    vault_service.backfill_seed_umls(conn, SETTINGS, limit=2)      # W0, W1 fail
    vault_service.backfill_seed_umls(conn, SETTINGS, limit=2)      # must move on
    assert {"W2", "W3"}.isdisjoint(_null(conn))


def test_a_paper_with_no_text_is_marked_done(conn, failing_for):
    conn.execute("UPDATE papers SET title = '', abstract = NULL, body_text = NULL WHERE openalex_id = 'W4'")
    conn.commit()
    vault_service.backfill_seed_umls(conn, SETTINGS, openalex_ids=["W4"])
    assert conn.execute("SELECT umls_concepts_json FROM papers WHERE openalex_id = 'W4'").fetchone()[0] == "[]"


def test_a_paper_another_caller_already_filled_is_not_extracted_again(conn, failing_for):
    _, seen = failing_for
    conn.execute("UPDATE papers SET umls_concepts_json = '[]' WHERE openalex_id = 'W0'")
    conn.commit()
    assert vault_service._try_extract_umls(conn, SETTINGS, "W0", "text") is True
    assert seen == []
    assert vault_service._try_extract_umls(conn, SETTINGS, "W0", "text", force=True) is True
    assert len(seen) == 1


def test_a_failed_paper_is_tried_again_after_a_while(conn, failing_for, monkeypatch):
    """Skipped until a restart, an hour's outage left ~160 seeds without UMLS
    for as long as the backend ran -- weeks."""
    bad, _ = failing_for
    bad.add("Seed 0")
    vault_service.backfill_seed_umls(conn, SETTINGS, openalex_ids=["W0"])
    assert vault_service.backfill_seed_umls(conn, SETTINGS, openalex_ids=["W0"]) == 0   # skipped for now
    bad.clear()
    monkeypatch.setattr(vault_service, "_UMLS_RETRY_AFTER_S", 0)
    assert vault_service.backfill_seed_umls(conn, SETTINGS, openalex_ids=["W0"]) == 1
    assert "W0" not in _null(conn)
