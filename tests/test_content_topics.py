"""No topics are guessed from an uploaded PDF's text (user 11-04).

An unpublished paper's topics come only from its UMLS concepts and from
Claude's research topics. Rows written while they were guessed
(``source: pdf_content``) stay in the database and are read as no topics,
by every reader of ``papers.topics_json``.
"""

from __future__ import annotations

import json

import pytest

from rag_lib import content_topics
from rag_lib.api import mappers, settings as settings_module
from rag_lib.api.services import suggestions, wizard
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import papers as papers_repo, profiles as profiles_repo, users as users_repo


def _t(n: int, name: str | None = None) -> dict:
    return {"id": f"https://openalex.org/T{n}", "display_name": name or f"Topic {n}",
            "subfield": None, "field": None, "domain": None, "score": 0.9}


def _legacy(*ns: int) -> str:
    """``topics_json`` as it was written when topics were read from a PDF's text."""
    return json.dumps({"primary_topic": None, "topics": [_t(n) for n in ns], "source": content_topics.SOURCE})


def _openalex(*ns: int) -> str:
    return json.dumps({"primary_topic": _t(ns[0]), "topics": [_t(n) for n in ns]})


@pytest.fixture()
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("RADAR_UMLS_ENABLED", "false")   # no UMLS models in a unit test
    settings_module.get_settings.cache_clear()
    c = connect(tmp_path / "r.db")
    apply_migrations(c)
    yield c
    c.close()
    settings_module.get_settings.cache_clear()


def _interest_with(conn, seeds: dict[str, str | None]) -> int:
    """A draft whose seeds are ``{openalex_id: topics_json}``."""
    uid = int(users_repo.upsert(conn, "a@example.com")["id"])
    pid = profiles_repo.create_draft(conn, user_id=uid, name="mixed", slug="mixed")
    for oa, tj in seeds.items():
        papers_repo.upsert(conn, {"openalex_id": oa, "title": f"Paper {oa[-6:]}",
                                  "body_text": "x " * 50, "source": "user_pdf", "topics_json": tj})
        profiles_repo.attach_seed(conn, pid, oa)
    return pid


def test_what_counts_as_read_from_a_pdf():
    assert content_topics.is_from_content(_legacy(1))
    assert content_topics.seed_topics_raw(_legacy(1)) is None
    for raw in (None, "", "not json", "[]", _openalex(1), json.dumps({"source": "umls", "topics": []})):
        assert not content_topics.is_from_content(raw)
        assert content_topics.seed_topics_raw(raw) == raw


def test_a_legacy_pdf_topic_is_not_in_the_interest_s_topics(conn):
    pid = _interest_with(conn, {
        "https://openalex.org/W1": _openalex(1, 2),
        "local:abc": _legacy(500, 2),
    })
    topics = wizard._aggregate_topics(conn, pid, new_on=True)["topics"]
    by_id = {t["id"]: t for t in topics}
    assert "https://openalex.org/T500" not in by_id
    assert all(t.get("source") != content_topics.SOURCE for t in topics)
    # T2 is W1's alone: counted once, not twice
    assert by_id["https://openalex.org/T2"]["count"] == 1


def test_an_unmatched_seed_without_topics_is_left_without_them(conn):
    """It used to be read on aggregation and its guess stored on the paper."""
    pid = _interest_with(conn, {"local:old": None})
    assert wizard._aggregate_topics(conn, pid, new_on=True)["topics"] == []
    assert conn.execute("SELECT topics_json FROM papers WHERE openalex_id = 'local:old'").fetchone()[0] is None


def test_a_legacy_pdf_topic_is_no_seed_s(conn):
    pid = _interest_with(conn, {"https://openalex.org/W1": _openalex(1), "local:abc": _legacy(500)})
    codes = wizard._seed_topic_codes(conn, pid)
    assert any(c.endswith("T1") for c in codes) and not any(c.endswith("T500") for c in codes)


def test_every_reader_of_a_paper_s_topics_reads_a_legacy_row_as_none(conn):
    _interest_with(conn, {"local:abc": _legacy(500)})
    row = papers_repo.get_by_openalex_id(conn, "local:abc")
    assert papers_repo.decode_topics(row) == {"primary_topic": None, "topics": []}
    assert mappers._decode_topics(row["topics_json"]) == {"primary_topic": None, "topics": []}
    assert suggestions._topics_of(row) == (None, [])


def test_the_overlap_fallback_leaves_a_legacy_pdf_topic_out(conn, monkeypatch):
    """Review L2: when the rebuild fails the stored list stands in -- without
    the topics no rebuild has now."""
    pid = _interest_with(conn, {"https://openalex.org/W1": _openalex(1)})
    conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE id = ?", (json.dumps({"topics": [
        {"id": _t(1)["id"], "display_name": "Topic 1", "on": True},
        {"id": _t(500)["id"], "display_name": "Old", "on": True, "source": content_topics.SOURCE},
    ]}), pid))
    conn.commit()

    def boom(*a, **k):
        raise RuntimeError("rebuild failed")
    monkeypatch.setattr(wizard, "_aggregate_topics", boom)
    ids = [t["id"] for t in wizard._overlap_basis(conn, pid)["topics"]]
    assert ids == [_t(1)["id"]]


def test_a_legacy_pdf_topic_left_on_does_not_decide_a_forgotten_researcher_s_topics():
    """Review L3: the researcher is gone and the only other topic on is one
    once guessed from a PDF -- not in the rebuilt list, so it cannot be what
    stays on: the profile's topics are kept (adopted), not all switched off."""
    from rag_lib import researcher_topics as rt

    R, P = "https://openalex.org/T7", "https://openalex.org/T500"
    stored = {"topics": [
        {"id": R, "display_name": "From the profile", "on": True, "source": rt.SOURCE, rt.BEFORE: False},
        {"id": P, "display_name": "Old", "on": True, "source": content_topics.SOURCE},
    ]}
    topic_filters = {"topics": []}
    prior_on: dict = {}
    wizard._restore_or_adopt(stored, topic_filters, prior_on)
    assert [(t["id"], t["source"]) for t in topic_filters["topics"]] == [(R, rt.ADOPTED)]
