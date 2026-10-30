"""Topics for an uploaded PDF OpenAlex does not have, read from its text."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from rag_lib import content_topics
from rag_lib.api import settings as settings_module
from rag_lib.api.services import wizard
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import papers as papers_repo, profiles as profiles_repo, users as users_repo


def _t(n: int, name: str | None = None) -> dict:
    return {"id": f"https://openalex.org/T{n}", "display_name": name or f"Topic {n}",
            "subfield": None, "field": None, "domain": None, "score": 0.9}


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


def _openalex(*ns: int) -> str:
    return json.dumps({"primary_topic": _t(ns[0]), "topics": [_t(n) for n in ns]})


# --- reading topics from text ------------------------------------------------


class _Index:
    def __init__(self, sims):
        self.sims = sims

    def check_compatible(self, *a, **k):
        pass

    def search(self, vec, *, top_k, min_similarity):
        hits = [({"id": f"https://openalex.org/T{i}", "display_name": f"Topic {i}",
                  "subfield": "Sub", "field": "Field", "domain": "Dom"}, s)
                for i, s in enumerate(self.sims)]
        return [h for h in hits if h[1] >= min_similarity][:top_k]


def test_text_maps_to_the_nearest_topics_above_the_floor(monkeypatch):
    seen = {}
    monkeypatch.setattr("rag_lib.umls.cache.get_topic_index", lambda d: _Index([0.81, 0.7, 0.6, 0.5]))
    monkeypatch.setattr("rag_lib.embedders.get_embedder",
                        lambda m: (lambda text: seen.setdefault("text", text) and np.ones(4)))
    settings = SimpleNamespace(RADAR_UMLS_EMBEDDING_MODEL="mxbai-embed-large", RADAR_UMLS_CACHE_DIR="x")
    out = content_topics.topics_from_text("Buffer sizing", "These are lab notes. " * 400, settings=settings)
    assert [t["display_name"] for t in out] == ["Topic 0", "Topic 1", "Topic 2"]   # 0.5 is below 0.55
    assert out[0]["score"] == 0.81 and out[0]["subfield"] == {"id": None, "display_name": "Sub"}
    assert seen["text"].startswith("Buffer sizing. These are lab notes.")
    assert len(seen["text"]) <= len("Buffer sizing. ") + content_topics.BODY_CHARS


def test_topics_that_share_a_name_count_once(monkeypatch):
    class Dup(_Index):
        def search(self, vec, *, top_k, min_similarity):
            return [({"id": f"https://openalex.org/T{i}", "display_name": n}, s) for i, (n, s) in
                    enumerate([("Marine ecosystems", 0.75), ("Marine Ecosystems", 0.75), ("Fisheries", 0.7), ("Corals", 0.69)])][:top_k]
    monkeypatch.setattr("rag_lib.umls.cache.get_topic_index", lambda d: Dup([]))
    monkeypatch.setattr("rag_lib.embedders.get_embedder", lambda m: (lambda text: np.ones(4)))
    settings = SimpleNamespace(RADAR_UMLS_EMBEDDING_MODEL="m", RADAR_UMLS_CACHE_DIR="x")
    out = content_topics.topics_from_text("Reef survey notes", "fish " * 50, settings=settings)
    assert [t["display_name"] for t in out] == ["Marine ecosystems", "Fisheries", "Corals"]


def test_too_little_text_reads_nothing():
    assert content_topics.topics_from_text("", "  ", settings=None) == []


def test_a_failing_embedder_leaves_it_for_later(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("ollama down")
    monkeypatch.setattr(content_topics, "topics_from_text", boom)
    assert content_topics.try_topics_json("t", None, "b", settings=SimpleNamespace()) is None


# --- into the interest's topics ---------------------------------------------


def test_an_unmatched_seeds_topics_survive_the_top_eight_cut(conn):
    """Matched seeds bring 10 topics; the PDF's one topic has a count of 1 and
    used to be cut. It stays, marked as read from the PDF."""
    pid = _interest_with(conn, {
        "https://openalex.org/W1": _openalex(*range(1, 11)),
        "https://openalex.org/W2": _openalex(*range(1, 11)),
        "local:abc": content_topics.topics_json([_t(500, "Buffer Management")]),
    })
    topics = wizard._aggregate_topics(conn, pid, new_on=True)["topics"]
    by_id = {t["id"]: t for t in topics}
    assert by_id["https://openalex.org/T500"]["source"] == content_topics.SOURCE
    assert by_id["https://openalex.org/T500"]["on"] is True
    assert "source" not in by_id["https://openalex.org/T1"]


def test_a_content_topic_a_matched_seed_also_has_is_not_marked(conn):
    pid = _interest_with(conn, {
        "https://openalex.org/W1": _openalex(7),
        "local:abc": content_topics.topics_json([_t(7), _t(8)]),
    })
    by_id = {t["id"]: t for t in wizard._aggregate_topics(conn, pid, new_on=True)["topics"]}
    assert "source" not in by_id["https://openalex.org/T7"]
    assert by_id["https://openalex.org/T8"]["source"] == content_topics.SOURCE


def test_an_older_unmatched_seed_is_read_when_topics_are_aggregated(conn, monkeypatch):
    """Uploaded before this existed: no topics_json. Aggregation reads it once
    and stores the result, empty or not, on the paper."""
    calls = []
    monkeypatch.setattr(content_topics, "topics_from_text",
                        lambda title, body, settings: calls.append(title) or [_t(42, "Lab Notes")])
    pid = _interest_with(conn, {"local:old": None})
    topics = wizard._aggregate_topics(conn, pid, new_on=True)["topics"]
    assert [t["display_name"] for t in topics] == ["Lab Notes"]
    stored = conn.execute("SELECT topics_json FROM papers WHERE openalex_id = 'local:old'").fetchone()[0]
    assert json.loads(stored)["source"] == content_topics.SOURCE
    wizard._aggregate_topics(conn, pid, new_on=True)
    assert len(calls) == 1
