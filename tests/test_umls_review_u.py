"""Review 2026-11-12 of the UMLS topics as live (U2, U4, U6; U5 is in
test_wizard_umls_backfill.py)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from rag_lib.api import mappers
from rag_lib.api.services import wizard
from rag_lib.umls import cache
from rag_lib.umls.topic_mapper import MappedTopic, merge_umls_topics
from tests.edit_helpers import edit_saved_db
from tests.test_seed_removal_topics import (  # noqa: F401 -- fixture
    T3, T4, W1, W2, _T5, _T7, _T8, _by_id, _e, _remove, _save_with_seven_on, _stored, _umls_on, live)


# --- U2 / UF-R1: why no seed brings a saved topic ---

def _copy_id(conn, copy):
    return conn.execute("SELECT id FROM profiles WHERE slug = ?", (copy,)).fetchone()[0]


def test_u2_a_umls_topic_its_seed_now_maps_elsewhere_is_no_seed(live, monkeypatch, tmp_path):
    """W1 is still a seed; read again it no longer gives Five: not "its
    seed paper was removed"."""
    conn, uid, pid, slug = live
    _umls_on(conn, monkeypatch, tmp_path, {W1: [_e("C1", _T5, "Five", 0.9)], W2: []})
    wizard.aggregate_draft_topics(conn, user_id=uid, slug=slug)
    on = [t for t, v in _by_id(_stored(conn, pid)).items() if v["on"]] + [_T5]
    wizard.save_topics(conn, user_id=uid, slug=slug, selected_topic_ids=on)
    _umls_on(conn, monkeypatch, tmp_path, {W1: [_e("C1", _T8, "Eight", 0.9)], W2: []})
    with edit_saved_db(conn, uid, slug, save=False) as copy:
        got = _by_id(wizard.preview_topics(conn, user_id=uid, slug=copy)[0])
    assert got[_T5].get("unmatched") is True and got[_T5][wizard.UNMATCHED_WHY] == "no_seed"
    assert wizard.UNMATCHED_WHY not in got[_T8]
    api = {t.id: t for t in mappers.topic_filters_to_topics({"topics": list(got.values())})}
    assert api[_T5].unmatched is True and api[_T5].unmatched_why == "no_seed"
    assert api[_T8].unmatched is None and api[_T8].unmatched_why is None


def test_u2_the_topics_of_a_seed_removed_in_this_edit_are_seed_removed(live, monkeypatch, tmp_path):
    """Seven chosen by W2 only, Three W2's OpenAlex topic: W2 removed on the copy."""
    conn, uid, pid, slug = live
    _umls_on(conn, monkeypatch, tmp_path, {W1: [_e("C1", _T5, "Five", 0.9)], W2: [_e("C2", _T7, "Seven", 0.85)]})
    _save_with_seven_on(conn, uid, pid, slug)
    with edit_saved_db(conn, uid, slug, save=False) as copy:
        _remove(conn, _copy_id(conn, copy), W2)
        got = _by_id(wizard.preview_topics(conn, user_id=uid, slug=copy)[0])
        assert got[_T7][wizard.UNMATCHED_WHY] == "seed_removed"
        assert got[T3][wizard.UNMATCHED_WHY] == "seed_removed"
        # neither mark is stored
        wizard.save_topics(conn, user_id=uid, slug=copy, selected_topic_ids=[t for t, v in got.items() if v["on"]])
        assert all(wizard.UNMATCHED not in t and wizard.UNMATCHED_WHY not in t
                   for t in _stored(conn, _copy_id(conn, copy))["topics"])


def test_uf_r1_an_openalex_topic_its_seed_lost_when_classed_again_is_no_seed(live):
    """Review UF-R1: W2 fetched again, OpenAlex now classes it Four, not
    Three; no seed was removed."""
    conn, uid, pid, slug = live
    conn.execute("UPDATE papers SET topics_json = ? WHERE openalex_id = ?",
                 (json.dumps({"primary_topic": {"id": T4, "display_name": "Four"},
                              "topics": [{"id": T4, "display_name": "Four"}]}), W2))
    conn.commit()
    with edit_saved_db(conn, uid, slug, save=False) as copy:
        got = _by_id(wizard.preview_topics(conn, user_id=uid, slug=copy)[0])
    assert got[T3].get("unmatched") is True and got[T3][wizard.UNMATCHED_WHY] == "no_seed"


def test_u2_without_a_reason_the_api_does_not_say_removed():
    t = mappers.topic_filters_to_topics({"topics": [{"id": "T1", "display_name": "One", "unmatched": True}]})[0]
    assert t.unmatched_why == "no_seed"


# --- U6: the seeds' topics are one per topic id, whatever their names ---

def _mt(tid, name, cui, sim=0.8):
    return MappedTopic(topic_id=tid, display_name=name, similarity=sim, source_cui=cui,
                       source_name=cui, confidence=0.95, mention=cui, rank=1)


def test_u6_one_topic_under_an_old_and_a_new_name_is_one_entry_of_two_seeds():
    url = "https://openalex.org/T13434"
    merged = merge_umls_topics({"topics": []}, [[_mt(url, "Cancer Research and Treatment", "C1", 0.7)],
                                                 [_mt(url, "Cancer Nanomedicine", "C2", 0.9)],
                                                 [_mt("T13434", "Cancer Nanomedicine", "C3", 0.6)]])
    umls = [t for t in merged["topics"] if t["source"] == "umls"]
    assert len(umls) == 1
    assert umls[0]["count"] == 3 and umls[0]["papers"] == [0, 1, 2]
    assert umls[0]["display_name"] == "Cancer Nanomedicine"        # its nearest concept's


def test_u6_within_a_seed_one_topic_under_two_names_is_given_once():
    url = "https://openalex.org/T13434"
    merged = merge_umls_topics({"topics": []}, [[_mt(url, "Old name", "C1", 0.9), _mt(url, "New name", "C2", 0.8),
                                                  _mt("https://openalex.org/T2", "Two", "C2", 0.7)]])
    umls = {t["id"]: t for t in merged["topics"] if t["source"] == "umls"}
    assert set(umls) == {url, "https://openalex.org/T2"}            # C2 gives its next one


def test_u6_the_same_name_under_another_id_is_still_one():
    merged = merge_umls_topics({"topics": []}, [[_mt("https://openalex.org/T1", "One", "C1")],
                                                 [_mt("https://openalex.org/T9", "One", "C2")]])
    umls = [t for t in merged["topics"] if t["source"] == "umls"]
    assert len(umls) == 1 and umls[0]["count"] == 2


def test_u6_a_bare_code_the_interest_has_is_had():
    merged = merge_umls_topics({"topics": [{"id": "T5", "display_name": "Zebra migration"}]},
                               [[_mt("https://openalex.org/T5", "Five", "C1")]])
    assert [t for t in merged["topics"] if t.get("source") == "umls"] == []
    merged = merge_umls_topics({"topics": [{"id": "https://openalex.org/T5", "display_name": "Zebra migration"}]},
                               [[_mt("T5", "Five", "C1")]])
    assert [t for t in merged["topics"] if t.get("source") == "umls"] == []


# --- U4: the two topic indexes name every topic alike ---

def _write(d, names, rich):
    d.mkdir(parents=True, exist_ok=True)
    for (tf, ef, mf), topics in (((cache.TOPICS_FILE, cache.EMBEDDINGS_FILE, cache.META_FILE), names),
                                 ((cache.RICH_TOPICS_FILE, cache.RICH_EMBEDDINGS_FILE, cache.RICH_META_FILE), rich)):
        (d / tf).write_text(json.dumps([{"id": i, "display_name": n} for i, n in topics]))
        np.savez_compressed(d / ef, embeddings=np.ones((len(topics), 4), dtype=np.float32))
        (d / mf).write_text(json.dumps({"embedding_model": "mxbai-embed-large", "dim": 4}))


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(cache, "_index", None)
    monkeypatch.setattr(cache, "_rich_index", None)


def test_u4_a_renamed_topic_is_said(tmp_path, fresh):
    _write(tmp_path, [("T1", "Cancer Research and Treatment"), ("T2", "Two")],
           [("T1", "Cancer Nanomedicine"), ("T2", "Two")])
    p = cache.topic_names_problem(tmp_path)
    assert p.startswith("1 topics named differently and 0 in only one")
    assert "'Cancer Research and Treatment' is now 'Cancer Nanomedicine'" in p


def test_u4_a_topic_in_one_index_only_is_said(tmp_path, fresh):
    _write(tmp_path, [("T1", "One")], [("T1", "One"), ("T2", "Two")])
    assert cache.topic_names_problem(tmp_path).startswith("0 topics named differently and 1 in only one")


def test_u4_alike_or_unreadable_is_no_problem(tmp_path, fresh, monkeypatch):
    assert cache.topic_names_problem(tmp_path) is None                # nothing built: said elsewhere
    _write(tmp_path, [("T1", "One")], [("T1", "One")])
    assert cache.topic_names_problem(tmp_path) is None


def test_u4_health_says_names_differ(tmp_path, fresh, monkeypatch):
    from rag_lib.api.routers import health

    s = SimpleNamespace(RADAR_UMLS_ENABLED=True, RADAR_UMLS_CACHE_DIR=tmp_path,
                        RADAR_UMLS_EMBEDDING_MODEL="mxbai-embed-large")
    _write(tmp_path, [("T1", "Old")], [("T1", "New")])
    assert health._umls_topic_index(s) == "names_differ"
    monkeypatch.setattr(cache, "_index", None)
    monkeypatch.setattr(cache, "_rich_index", None)
    _write(tmp_path, [("T1", "New")], [("T1", "New")])
    assert health._umls_topic_index(s) == "ok"


def test_uf_r6_a_topic_mapped_under_its_old_name_is_listed_under_its_name_now(tmp_path, fresh):
    _write(tmp_path, [("https://openalex.org/T1", "Cancer Nanomedicine")], [("https://openalex.org/T1", "Cancer Nanomedicine")])
    merged = merge_umls_topics({"topics": []}, [[_mt("https://openalex.org/T1", "Cancer Research and Treatment", "C1")]],
                               cache_dir=str(tmp_path))
    assert [t["display_name"] for t in merged["topics"]] == ["Cancer Nanomedicine"]
