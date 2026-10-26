"""OpenAlex topics mapped from the research topics Claude generated
(``rag_lib.llm_topics``, ``POST /api/profiles/{key}/llm-topics``).

They are listed as ``source: llm``, start switched off, survive the list
being rebuilt from the seeds, and are searched by once switched on.
"""

from __future__ import annotations

import json

import numpy as np

from rag_lib import llm_topics
from rag_lib.llm_topics import LabelMatch
from rag_lib.openalex_tiers import enabled_topic_ids
from tests.test_review_round12_journeys import (  # noqa: F401 -- fixtures
    _go_live, _ok, _q, _request, app, env,
)

T = "https://openalex.org/T"


# --- mapping ----------------------------------------------------------------


class _Index:
    """Topic vectors on axes; a label's vector says how close it is to each."""

    def __init__(self, names):
        self.names = names

    def check_compatible(self, *a, **k):
        pass

    def search(self, vec, *, top_k, min_similarity):
        sims = [(i, float(v)) for i, v in enumerate(vec) if v >= min_similarity]
        sims.sort(key=lambda x: -x[1])
        return [({"id": f"{T}{i}", "display_name": self.names[i]}, s) for i, s in sims[:top_k]]


def test_a_labels_best_match_from_the_lower_bar_more_only_from_the_higher(monkeypatch):
    vecs = {"speech": [0.83, 0.82, 0.70, 0], "records": [0, 0, 0.69, 0.87], "vague": [0.6, 0, 0, 0]}
    monkeypatch.setattr("rag_lib.embedders.get_embedder", lambda m: (lambda s: np.array(vecs[s])))
    monkeypatch.setattr("rag_lib.umls.cache.get_topic_index", lambda d: _Index(["A", "B", "C", "D"]))
    got = llm_topics.map_labels(["speech", "records", "vague"], embedding_model="m", cache_dir="x",
                                min_similarity=0.65, extra_similarity=0.80)
    # speech: A (best) and B (0.82 >= 0.80), not C (0.70); records: D only
    # (C at 0.69 is a second match under 0.80); vague: nothing reaches 0.65.
    assert [(m.display_name, m.label) for m in got] == [("D", "records"), ("A", "speech"), ("B", "speech")]


# --- merging ----------------------------------------------------------------


def _m(i, name, label="x", sim=0.9):
    return LabelMatch(f"{T}{i}", name, sim, label)


def test_new_topics_join_off_and_duplicates_are_not_added():
    stored = {"topics": [{"id": f"{T}1", "display_name": "Speech", "on": True}]}
    out, added = llm_topics.merge(stored, [_m(1, "Speech"), _m(2, "speech"), _m(3, "Dementia", "Dementia & CI")])
    assert [t["display_name"] for t in added] == ["Dementia"]
    new = out["topics"][-1]
    assert new["source"] == "llm" and new["on"] is False and new["llm_label"] == "Dementia & CI"
    assert stored["topics"] == [{"id": f"{T}1", "display_name": "Speech", "on": True}]  # not modified


def test_mapping_again_replaces_the_ones_left_off_and_keeps_the_ones_switched_on():
    first, _ = llm_topics.merge({"topics": []}, [_m(1, "A"), _m(2, "B")])
    first["topics"][0]["on"] = True            # the user took A
    again, added = llm_topics.merge(first, [_m(3, "C")])
    assert [(t["display_name"], t["on"]) for t in again["topics"]] == [("A", True), ("C", False)]
    assert [t["display_name"] for t in added] == ["C"]
    # Mapped once more to the same: listed, but nothing is new.
    same, added = llm_topics.merge(again, [_m(3, "C")])
    assert [t["display_name"] for t in same["topics"]] == ["A", "C"] and added == []


def test_at_most_max_additions():
    _, added = llm_topics.merge({"topics": []}, [_m(i, f"N{i}") for i in range(20)], max_additions=4)
    assert len(added) == 4


# --- through the API --------------------------------------------------------


def _set_terms(env, slug, source="rp-sdk-llm", labels=("Dementia & Cognitive Impairment", "Speech Analysis")):
    from rag_lib.db import connect
    c = connect(env)
    try:
        c.execute("UPDATE profiles SET interest_terms_json = ? WHERE slug = ?",
                  (json.dumps({"source": source, "terms": [{"label": s} for s in labels], "saved": True}), slug))
        c.commit()
    finally:
        c.close()


def _stored(env, slug):
    return json.loads(_q(env, "SELECT topic_filters_json FROM profiles WHERE slug = ?", slug)[0][0])


def test_the_route_adds_llm_topics_off_and_a_rebuild_keeps_them(app, env, monkeypatch):
    slug, _ = _go_live(app, env, "LLM One")
    calls = []

    def fake_map(labels, **kw):
        calls.append(labels)
        return [_m(9001, "Dementia and Cognitive Impairment Research", labels[0], 0.82),
                _m(9002, "Speech and Audio Processing", labels[1], 0.81)]
    monkeypatch.setattr(llm_topics, "map_labels", fake_map)

    # Nothing generated yet, then a list the cluster method made: refused.
    r = _request(app, "POST", f"/api/profiles/{slug}/llm-topics")
    assert r.status_code == 409, r.text
    _set_terms(env, slug, source="rp-sdk-cluster")
    assert _request(app, "POST", f"/api/profiles/{slug}/llm-topics").status_code == 409
    assert calls == []

    _set_terms(env, slug)
    before = enabled_topic_ids(_stored(env, slug))
    res = _ok(_request(app, "POST", f"/api/profiles/{slug}/llm-topics"))
    assert calls == [["Dementia & Cognitive Impairment", "Speech Analysis"]]
    assert [(t["name"], t["source"], t["on"], t["llm_label"]) for t in res["added"]] == [
        ("Dementia and Cognitive Impairment Research", "llm", False, "Dementia & Cognitive Impairment"),
        ("Speech and Audio Processing", "llm", False, "Speech Analysis"),
    ]
    # Off: the next scan searches exactly what it did before.
    assert enabled_topic_ids(_stored(env, slug)) == before

    # The list rebuilt from the seeds (Recompute) keeps them, still off.
    _ok(_request(app, "POST", f"/api/profiles/{slug}/recompute-topics"))
    llm = [t for t in _stored(env, slug)["topics"] if t.get("source") == "llm"]
    assert [(t["id"], t["on"]) for t in llm] == [(f"{T}9001", False), (f"{T}9002", False)]

    # Switched on with the rest, it is searched by -- and stays on through a rebuild.
    _ok(_request(app, "PUT", f"/api/profiles/{slug}/topics", json={"selected_topic_ids": before + [f"{T}9001"]}))
    _ok(_request(app, "POST", f"/api/profiles/{slug}/recompute-topics"))
    assert f"{T}9001" in enabled_topic_ids(_stored(env, slug))
    assert f"{T}9002" not in enabled_topic_ids(_stored(env, slug))

    # The interest page lists them with their source and label.
    detail = _ok(_request(app, "GET", f"/api/profiles/{slug}/detail"))
    shown = {t["id"]: t for t in detail["topics"]}
    assert shown[f"{T}9001"]["source"] == "llm" and shown[f"{T}9001"]["llm_label"] == "Dementia & Cognitive Impairment"


def test_an_unavailable_topic_index_is_a_503_and_changes_nothing(app, env, monkeypatch):
    slug, _ = _go_live(app, env, "LLM Two")
    _set_terms(env, slug)
    before = _stored(env, slug)

    def broken(*a, **k):
        raise FileNotFoundError("no topic index in data/umls_cache")
    monkeypatch.setattr(llm_topics, "map_labels", broken)
    r = _request(app, "POST", f"/api/profiles/{slug}/llm-topics")
    assert r.status_code == 503, r.text
    assert _stored(env, slug) == before
