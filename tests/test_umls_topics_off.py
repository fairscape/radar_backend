"""UMLS topics are added switched off and say where they came from
(FUNCTIONAL_SPEC U.2-8, U-6; the rules themselves: test_umls_topic_mapper).

Through the wizard's aggregation, on stored mappings: what the topic list
holds, what a scan would query, and a mapping stored by the earlier rules
going back to extraction.
"""

from __future__ import annotations

import json

import pytest

from rag_lib import openalex_tiers
from rag_lib.api.mappers import topic_filters_to_topics
from rag_lib.api.services import wizard as wizard_service
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import papers as papers_repo, profiles as profiles_repo, users as users_repo
from rag_lib.umls.topic_mapper import MAPPING, MappedTopic

OWN = {"id": "https://openalex.org/T1", "display_name": "Alzheimer's disease research", "score": 0.9}


@pytest.fixture
def db(tmp_path, monkeypatch):
    from rag_lib.api import settings as settings_mod

    monkeypatch.setenv("RADAR_UMLS_ENABLED", "true")
    monkeypatch.setenv("RADAR_UMLS_CACHE_DIR", str(tmp_path / "no-index"))
    # The rich index is there (without it old data is kept: test_umls_ctx_review).
    monkeypatch.setattr("rag_lib.umls.cache.rich_index_problem", lambda *a: None)
    settings_mod.get_settings.cache_clear()
    conn = connect(tmp_path / "radar.db")
    apply_migrations(conn)
    user = users_repo.upsert(conn, "demo@example.com")
    pid = profiles_repo.upsert(conn, user_id=int(user["id"]), name="p", embedding_model="placeholder-v1",
                               n_seed=0, topic_filters={})
    yield conn, int(pid)
    conn.close()
    settings_mod.get_settings.cache_clear()


# Concepts as extracted since CTX: with where the text says them.
CONCEPTS = [{"cui": "C1", "mentions": [{"text": "immune", "sentence": "The immune response was measured."}]}]


def _seed(conn, pid, oid, mapped, concepts=CONCEPTS):
    papers_repo.upsert(conn, {"openalex_id": oid, "title": "A title", "abstract": "an abstract", "source": "s",
                              "topics_json": json.dumps({"primary_topic": OWN, "topics": [OWN]})})
    conn.execute("UPDATE papers SET umls_concepts_json = ?, umls_mapped_topics_json = ? "
                 "WHERE openalex_id = ?", (json.dumps(concepts), json.dumps(mapped), oid))
    profiles_repo.attach_seed(conn, pid, oid)
    conn.commit()


def _mt(tid, name, conf, concept, sim=0.7, mention=None, rank=1):
    # One concept per name: a concept gives one topic (CTX-3b).
    tid = tid if tid.startswith("https://") else f"https://openalex.org/{tid}"
    return MappedTopic(topic_id=tid, display_name=name, similarity=sim,
                       source_cui=f"C-{concept}", source_name=concept, confidence=conf, mention=mention,
                       mapping=MAPPING, rank=rank).to_dict()


def test_u6_umls_topics_start_off_and_a_scan_does_not_query_them(db):
    conn, pid = db
    _seed(conn, pid, "W1", [_mt("T9", "Immune Response and Inflammation", 0.98, "Immune response",
                                mention="immune"),
                            _mt("T8", "Dental materials", 0.8, "Baseline", sim=0.9)])
    filters = wizard_service._aggregate_topics(conn, pid, new_on=True)
    by_id = {t["id"]: t for t in filters["topics"]}
    assert by_id[OWN["id"]]["on"] is True
    umls = by_id["https://openalex.org/T9"]
    assert umls["source"] == "umls" and umls["on"] is False
    assert "https://openalex.org/T8" not in by_id               # under 0.9
    assert openalex_tiers.enabled_topic_ids(filters) == [OWN["id"]]
    # the API says which concept it came from
    out = {t.id: t for t in topic_filters_to_topics(filters)}
    assert out["https://openalex.org/T9"].umls_concept == "Immune response"
    assert out["https://openalex.org/T9"].umls_confidence == 0.98
    assert out["https://openalex.org/T9"].umls_mention == "immune"
    assert out[OWN["id"]].umls_concept is None


def test_u7_a_umls_topic_switched_on_stays_on(db):
    conn, pid = db
    _seed(conn, pid, "W1", [_mt("T9", "Immune Response and Inflammation", 0.98, "Immune response")])
    filters = wizard_service._aggregate_topics(conn, pid, new_on=True)
    for t in filters["topics"]:
        t["on"] = True                                          # "Turn on all UMLS topics"
    conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE id = ?", (json.dumps(filters), pid))
    conn.commit()
    again = wizard_service._aggregate_topics(conn, pid, new_on=True)
    assert "https://openalex.org/T9" in openalex_tiers.enabled_topic_ids(again)


def test_a_mapping_stored_by_the_earlier_rules_goes_back_to_extraction(db, monkeypatch):
    conn, pid = db
    old = [{"topic_id": "https://openalex.org/T9", "display_name": "X", "similarity": 0.7,
            "source_cui": "C1", "source_name": "x"}]
    _seed(conn, pid, "W1", old)
    _seed(conn, pid, "W2", [_mt("T9", "X", 0.95, "x")])
    # concepts stored before their sentences were (CTX): nothing to map them by
    _seed(conn, pid, "W3", [_mt("T9", "X", 0.95, "x")], concepts=[{"cui": "C1"}])
    # nothing found: a result, not stale
    _seed(conn, pid, "W4", [], concepts=[])
    extracted = []
    monkeypatch.setattr("rag_lib.api.services.vault._try_extract_umls",
                        lambda c, s, oid, text: extracted.append(oid))
    wizard_service._aggregate_topics(conn, pid, new_on=True)
    assert sorted(extracted) == ["W1", "W3"]
    for oid in ("W1", "W3"):
        assert conn.execute("SELECT umls_concepts_json FROM papers WHERE openalex_id = ?", (oid,)).fetchone()[0] is None


def test_step_3_entry_sends_an_old_mapping_back_before_it_waits(db):
    """POST /umls (step 3's entry) reads the old mapping as pending, so the
    wait it starts covers it; the topic list is built after."""
    conn, pid = db
    old = [{"topic_id": "https://openalex.org/T9", "display_name": "X", "similarity": 0.7,
            "source_cui": "C1", "source_name": "x"}]
    _seed(conn, pid, "W1", old)
    wizard_service.forget_stale_umls_mappings(conn, ["W1"])
    assert conn.execute("SELECT umls_mapped_topics_json FROM papers WHERE openalex_id = 'W1'").fetchone()[0] is None
    _seed(conn, pid, "W2", [_mt("T9", "X", 0.95, "x")])
    wizard_service.forget_stale_umls_mappings(conn, ["W2"])
    assert conn.execute("SELECT umls_mapped_topics_json FROM papers WHERE openalex_id = 'W2'").fetchone()[0]
    _seed(conn, pid, "W3", [_mt("T9", "X", 0.95, "x")], concepts=[{"cui": "C1"}])
    wizard_service.forget_stale_umls_mappings(conn, ["W3"])
    assert tuple(conn.execute("SELECT umls_concepts_json, umls_mapped_topics_json FROM papers "
                              "WHERE openalex_id = 'W3'").fetchone()) == (None, None)


def test_a_umls_topic_the_profile_marks_goes_back_off_when_the_profile_is_removed(db, monkeypatch):
    """Its state before the profile is recorded as off, not as a new topic's
    "on": taking the profile off must not switch it on."""
    from rag_lib import researcher_topics

    conn, pid = db
    _seed(conn, pid, "W1", [_mt("T9", "Immune Response and Inflammation", 0.98, "Immune response")])
    tid = "https://openalex.org/T9"

    def mark(filters, profile, *, basis=None):
        for t in filters.get("topics") or []:
            if t.get("id") == tid:
                t["profile"] = researcher_topics.INTEREST
        return filters
    monkeypatch.setattr(wizard_service, "_researcher_topics_for", lambda c, p: {"topics": []})
    monkeypatch.setattr(wizard_service.researcher_topics, "apply", mark)
    filters = wizard_service._aggregate_topics(conn, pid, new_on=True, profile_topics=True)
    t = next(t for t in filters["topics"] if t["id"] == tid)
    assert t[researcher_topics.BEFORE] is False


def test_concepts_with_no_mapping_at_all_were_mapped_by_the_earlier_rules(db):
    conn, pid = db
    _seed(conn, pid, "W1", [])
    conn.execute("UPDATE papers SET umls_mapped_topics_json = NULL WHERE openalex_id = 'W1'")
    _seed(conn, pid, "W2", [])
    conn.execute("UPDATE papers SET umls_concepts_json = '[]', umls_mapped_topics_json = NULL WHERE openalex_id = 'W2'")
    conn.commit()
    wizard_service.forget_stale_umls_mappings(conn, ["W1", "W2"])
    got = dict(conn.execute("SELECT openalex_id, umls_concepts_json FROM papers").fetchall())
    assert got["W1"] is None          # sent back
    assert got["W2"] == "[]"          # nothing found is a result


def test_ctx_through_the_wizard_the_nearer_topic_comes_first_and_none_is_barred(db):
    """CTX-3, through the real aggregation: no similarity bar by default (a
    far topic is still offered) and the nearer topic first, whatever the
    concepts' confidence."""
    conn, pid = db
    _seed(conn, pid, "W1", [_mt("T9", "Dental materials and restorations", 0.99, "BaseLine dental cement", sim=0.31),
                            _mt("T7", "Immune Response and Inflammation", 0.95, "Immune response", sim=0.74)])
    umls = [t["id"] for t in wizard_service._aggregate_topics(conn, pid, new_on=True)["topics"]
            if t.get("source") == "umls"]
    assert umls == ["https://openalex.org/T7", "https://openalex.org/T9"]


def test_ctx3b_through_the_wizard_a_concept_on_the_seeds_own_topic_gives_its_next(db):
    """User 11-11 (option B), through the stored mapping and the real
    aggregation: the concept's nearest is the seed's own topic, its 2nd is
    offered, its 3rd is not (a concept gives one)."""
    conn, pid = db
    _seed(conn, pid, "W1", [_mt(OWN["id"], OWN["display_name"], 0.95, "Adaptive Immunity", sim=0.75),
                            _mt("T901", "Immune responses and vaccinations", 0.95, "Adaptive Immunity",
                                sim=0.69, rank=2),
                            _mt("T902", "T-cell and B-cell Immunology", 0.95, "Adaptive Immunity",
                                sim=0.68, rank=3)])
    umls = [t for t in wizard_service._aggregate_topics(conn, pid, new_on=True)["topics"]
            if t.get("source") == "umls"]
    assert [(t["id"], t["concept"]) for t in umls] == [("https://openalex.org/T901", "Adaptive Immunity")]


def test_ut_a_the_wizard_follows_the_setting(db, monkeypatch):
    from rag_lib.api import settings as settings_mod

    conn, pid = db
    _seed(conn, pid, "W1", [_mt("T9", "Dental materials and restorations", 0.99, "BaseLine dental cement", sim=0.31)])
    monkeypatch.setenv("RADAR_UMLS_TOPIC_MIN_SIMILARITY", "0.5")
    settings_mod.get_settings.cache_clear()
    ids = {t["id"] for t in wizard_service._aggregate_topics(conn, pid, new_on=True)["topics"]}
    assert "https://openalex.org/T9" not in ids
