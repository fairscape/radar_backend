"""A researcher's declared interests as topics of an interest built from them.

A Prosopia profile says, in ``research_interests``, which OpenAlex topics
its owner wants (declared, or inferred from their papers) and which they do
not (a negative weight), plus text concepts for what maps to no topic. The
researcher row stores that document; an interest built from the researcher
can take the topics from it: the wanted ones join its list switched on, the
not-wanted ones are switched off. Offline: the Prosopia and OpenAlex stubs
of the import suites.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from rag_lib import researcher_topics as rt
from rag_lib.api.mappers import topic_filters_to_topics
from rag_lib.db import connect
from tests.test_orcid_import import ORCID, WORKS, StubOrcidOpenAlex
from tests.test_prosopia_import import StubProsopia, _stub_openalex, _stub_profile
from tests.test_researchers import _app, _request, env  # noqa: F401 -- fixture


def _topic(code: str, name: str, *, weight=None, method="inferred"):
    entry = {"@type": "ResearchInterest", "method": method,
             "concept": {"@id": f"https://openalex.org/{code}", "system": "https://openalex.org/topics",
                         "code": code, "display": name, "version": "2026-09"}}
    if weight is not None:
        entry["weight"] = weight
    return entry


def _text(label: str, *, weight: float, method="declared"):
    return {"@type": "ResearchInterest", "method": method, "weight": weight,
            "concept": {"label": label, "unmapped": True}}


DOCUMENT = {
    "name": "Nathan C. Sheffield",
    "interests": ["Genomics and Chromatin Dynamics", "Lab notebooks"],
    "not_interests": ["phylogenomics", "wet-lab-protocol-development"],
    "research_interests": [
        _topic("T10222", "Genomics and Chromatin Dynamics"),                  # inferred
        _topic("T10222", "Genomics and Chromatin Dynamics", weight=0.5, method="declared"),
        _topic("T10515", "Cancer-related molecular mechanisms", weight=0.2, method="declared"),
        _topic("T10001", "Test Topic"),                                       # inferred, carried by a seed below
        _topic("T10001", "Test Topic", weight=-0.5, method="declared"),       # ...but the owner said no
        _topic("T11986", "Scientific Computing", weight=-0.3, method="declared"),
        _text("phylogenomics", weight=-0.5),
        _text("wet-lab-protocol-development", weight=-0.5),
    ],
}


# --- reading the document ------------------------------------------------------


def test_read_settles_each_topic_once_and_keeps_the_labels():
    view = rt.read(DOCUMENT)
    assert [(t["id"], t["method"], t["weight"]) for t in view["wanted"]] == [
        ("https://openalex.org/T10222", "declared", 0.5),
        ("https://openalex.org/T10515", "declared", 0.2),
    ]
    # Declared beats inferred: T10001 is not wanted, whatever the papers say.
    assert [t["id"] for t in view["unwanted"]] == ["https://openalex.org/T10001", "https://openalex.org/T11986"]
    # A label a topic already covers is not repeated in words.
    assert view["wanted_labels"] == ["Lab notebooks"]
    assert view["unwanted_labels"] == ["phylogenomics", "wet-lab-protocol-development"]


def test_read_accepts_bare_codes_and_ignores_junk():
    doc = {"research_interests": [
        {"concept": {"id": "T10222", "display": "Genomics"}},
        {"concept": {"@id": "https://openalex.org/T10222", "display": "Genomics"}, "weight": "x"},
        {"concept": "not a dict"}, "not an entry", {"concept": {"label": ""}},
        {"concept": {"@id": "https://openalex.org/W123", "display": "a work, not a topic"}},
    ]}
    view = rt.read(doc)
    assert [t["id"] for t in view["wanted"]] == ["https://openalex.org/T10222"]
    assert view["unwanted"] == [] and view["wanted_labels"] == []
    assert rt.read(None) == rt.read({}) == rt.read({"research_interests": "nope"})
    assert rt.available(rt.read(None)) is False and rt.available(view) is True


def test_from_row_reads_the_stored_document_and_survives_bad_json():
    assert rt.from_row({"document_json": json.dumps(DOCUMENT)})["wanted"]
    assert rt.from_row({"document_json": "{not json"}) == rt.read(None)
    assert rt.from_row({"document_json": None}) == rt.read(None)
    assert rt.from_row(None) == rt.read(None)


def test_apply_adds_wanted_marks_both_and_strip_undoes_it():
    seeds = {"topics": [
        {"id": "T10001", "display_name": "Test Topic", "count": 3, "on": True},
        {"id": "https://openalex.org/T10222", "display_name": "Genomics and Chromatin Dynamics", "count": 1, "on": False},
    ], "fields": [{"id": "F1", "display_name": "Test Field", "count": 4}]}
    out = rt.apply(seeds, rt.read(DOCUMENT))
    by = {t["id"]: t for t in out["topics"]}
    assert by["T10001"]["profile"] == rt.NOT_INTEREST and "source" not in by["T10001"]   # matched by bare code
    assert by["https://openalex.org/T10222"]["profile"] == rt.INTEREST and by["https://openalex.org/T10222"]["count"] == 1
    added = by["https://openalex.org/T10515"]
    assert added["source"] == rt.SOURCE and added["count"] == 0 and added["profile"] == rt.INTEREST
    assert "https://openalex.org/T11986" not in by                     # a not-wanted topic no seed has is nothing to filter
    assert out[rt.FLAG] is True and out["fields"] == seeds["fields"]
    assert seeds["topics"][0].get("profile") is None                   # the input was not touched
    back = rt.strip(out)
    assert [t["id"] for t in back["topics"]] == ["T10001", "https://openalex.org/T10222"]
    assert all("profile" not in t for t in back["topics"]) and back[rt.FLAG] is False


class _Index:
    def __init__(self, hits):
        self.hits = hits      # label -> (id, name, sim)

    def check_compatible(self, *a, **k):
        pass

    def search(self, vec, *, top_k, min_similarity):
        label = "".join(chr(int(x)) for x in vec)      # the stub embedder's codepoints
        if label not in self.hits:
            return []
        tid, name, sim = self.hits[label]
        return [({"id": tid, "display_name": name}, sim)] if sim >= min_similarity else []


def _label_settings(monkeypatch, hits):
    monkeypatch.setattr("rag_lib.umls.cache.get_topic_index", lambda d: _Index(hits))
    monkeypatch.setattr("rag_lib.embedders.get_embedder",
                        lambda m: (lambda text: np.array([ord(c) for c in text], dtype=np.float32)))
    rt.reset_label_cache()
    return SimpleNamespace(RADAR_UMLS_ENABLED=True, RADAR_UMLS_EMBEDDING_MODEL="m", RADAR_UMLS_CACHE_DIR="x")


def test_labels_map_to_their_nearest_topic_on_their_own_side(monkeypatch):
    settings = _label_settings(monkeypatch, {
        "phylogenomics": ("https://openalex.org/T10434", "Genomics and Phylogenetic Studies", 0.74),
        "wet lab protocol development": ("https://openalex.org/T19999", "Laboratory Methods", 0.41),   # too far
        "Lab notebooks": ("https://openalex.org/T10222", "Genomics and Chromatin Dynamics", 0.9),       # already a topic
    })
    view = rt.with_labels(rt.read(DOCUMENT), settings=settings)
    mapped = [t for t in view["unwanted"] if t["method"] == "label"]
    assert [(t["id"], t["label"]) for t in mapped] == [("https://openalex.org/T10434", "phylogenomics")]
    assert len(view["wanted"]) == 2       # nothing added for a label the topics already cover
    assert view["unwanted_labels"] == ["phylogenomics", "wet-lab-protocol-development"]   # the words stay listed


def test_a_label_mapping_to_both_sides_is_left_out_and_a_dead_embedder_changes_nothing(monkeypatch):
    doc = {"interests": ["chromatin"], "not_interests": ["chromatin biology"]}
    settings = _label_settings(monkeypatch, {
        "chromatin": ("https://openalex.org/T1", "Chromatin", 0.8),
        "chromatin biology": ("https://openalex.org/T1", "Chromatin", 0.8),
    })
    view = rt.with_labels(rt.read(doc), settings=settings)
    assert view["wanted"] == [] and view["unwanted"] == []

    def boom(*a, **k):
        raise RuntimeError("ollama down")
    monkeypatch.setattr("rag_lib.embedders.get_embedder", boom)
    rt.reset_label_cache()
    # The view comes back as read, saying its labels are not mapped; with
    # UMLS off there is nothing to map, which is not an outage.
    assert rt.with_labels(rt.read(DOCUMENT), settings=settings) == {**rt.read(DOCUMENT), "labels_unmapped": True}
    assert rt.with_labels(rt.read(DOCUMENT), settings=SimpleNamespace(RADAR_UMLS_ENABLED=False)) == {**rt.read(DOCUMENT), "labels_unmapped": False}


def test_the_mapper_carries_the_profile_mark():
    topics = topic_filters_to_topics({"topics": [
        {"id": "T1", "display_name": "A", "count": 1, "on": False, "profile": rt.NOT_INTEREST},
        {"id": "T2", "display_name": "B", "count": 0, "source": rt.SOURCE, "profile": rt.INTEREST},
        {"id": "T3", "display_name": "C", "count": 2},
    ]})
    assert [(t.id, t.on, t.source, t.profile) for t in topics] == [
        ("T1", False, None, rt.NOT_INTEREST), ("T2", True, rt.SOURCE, rt.INTEREST), ("T3", True, None, None)]


# --- through the API ------------------------------------------------------------


def _profile_with_document():
    profile = _stub_profile()
    profile.metadata = DOCUMENT       # what ``record_prosopia_researcher`` stores as document_json
    profile.rid = "0000-0001-5643-4068"
    return profile


def _draft_from(app, researcher_id, name="From Nathan"):
    resp = _request(app, "POST", f"/api/researchers/{researcher_id}/interests", json={"name": name})
    assert resp.status_code == 200, resp.text
    return resp.json()["draft"]["slug"]


def _topics(app, slug):
    resp = _request(app, "GET", f"/api/profiles/draft/{slug}/topics")
    assert resp.status_code == 200, resp.text
    return {t["id"]: t for t in resp.json()}


def test_an_interest_takes_the_profiles_topics_and_keeps_them_across_aggregation(env):
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    rid = _import_prosopia(app)
    slug = _draft_from(app, rid)

    # Before: the seeds' own topics (the canned works all carry T10001).
    view = _request(app, "GET", f"/api/profiles/{slug}/researcher-topics").json()
    assert view["researcher_id"] == rid and view["researcher_name"] == "Nathan C. Sheffield"
    assert view["applied"] is False and view["topics"]["available"] is True
    assert [t["name"] for t in view["topics"]["wanted"]] == ["Genomics and Chromatin Dynamics", "Cancer-related molecular mechanisms"]
    assert [t["id"] for t in view["topics"]["unwanted"]] == ["https://openalex.org/T10001", "https://openalex.org/T11986"]
    before = _topics(app, slug)
    assert before["T10001"]["on"] is True and before["T10001"]["profile"] is None

    resp = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True})
    assert resp.status_code == 200, resp.text
    after = {t["id"]: t for t in resp.json()}
    assert after["T10001"]["on"] is False and after["T10001"]["profile"] == "not_interest"
    assert after["T10001"]["source"] is None and after["T10001"]["count"] == before["T10001"]["count"] > 0
    for tid in ("https://openalex.org/T10222", "https://openalex.org/T10515"):
        assert after[tid]["on"] is True and after[tid]["source"] == "researcher"
        assert after[tid]["profile"] == "interest" and after[tid]["count"] == 0
    assert "https://openalex.org/T11986" not in after
    assert _request(app, "GET", f"/api/profiles/{slug}/researcher-topics").json()["applied"] is True

    # Step 3 re-aggregates from the seeds every time it is opened: the
    # profile's topics and the switched-off one stay as they were.
    again = _topics(app, slug)
    assert {k: (v["on"], v["profile"]) for k, v in again.items()} == {k: (v["on"], v["profile"]) for k, v in after.items()}

    # The user may still want the topic the profile does not: their switch wins from then on.
    conn = connect(env)
    try:
        stored = json.loads(conn.execute("SELECT topic_filters_json FROM profiles WHERE slug = ?", (slug,)).fetchone()[0])
        for t in stored["topics"]:
            if t["id"] == "T10001":
                t["on"] = True
        conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE slug = ?", (json.dumps(stored), slug))
        conn.commit()
    finally:
        conn.close()
    assert _topics(app, slug)["T10001"]["on"] is True

    # Dropping them: the added topics go, the marks go, the seeds' topic stays as the user left it.
    resp = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": False})
    assert resp.status_code == 200, resp.text
    dropped = {t["id"]: t for t in resp.json()}
    assert set(dropped) == {"T10001"} and dropped["T10001"]["profile"] is None and dropped["T10001"]["on"] is True
    assert set(_topics(app, slug)) == {"T10001"}
    assert _request(app, "GET", f"/api/profiles/{slug}/researcher-topics").json()["applied"] is False


def test_a_not_wanted_topic_a_new_seed_brings_in_later_starts_off(env):
    """Applied once, the profile keeps filtering: a topic the owner does not
    want that first shows up on a later aggregation (a new seed carries it)
    starts switched off, where any other new topic on a draft starts on."""
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    assert _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).status_code == 200
    conn = connect(env)
    try:
        # One seed now also carries T11986 (not wanted) and T12345 (nothing said about it).
        oa_id = conn.execute("SELECT openalex_id FROM profile_seeds ps JOIN profiles p ON p.id = ps.profile_id "
                             "WHERE p.slug = ? AND openalex_id LIKE 'https://openalex.org/W%' LIMIT 1", (slug,)).fetchone()[0]
        raw = json.loads(conn.execute("SELECT topics_json FROM papers WHERE openalex_id = ?", (oa_id,)).fetchone()[0])
        for code, name in (("T11986", "Scientific Computing"), ("T12345", "Something New")):
            raw["topics"].append({"id": f"https://openalex.org/{code}", "display_name": name, "score": 0.5,
                                  "subfield": None, "field": None, "domain": None})
        conn.execute("UPDATE papers SET topics_json = ? WHERE openalex_id = ?", (json.dumps(raw), oa_id))
        conn.commit()
    finally:
        conn.close()
    topics = _topics(app, slug)
    assert topics["https://openalex.org/T11986"]["on"] is False
    assert topics["https://openalex.org/T11986"]["profile"] == "not_interest"
    assert topics["https://openalex.org/T12345"]["on"] is True and topics["https://openalex.org/T12345"]["profile"] is None


def test_applying_is_an_explicit_choice_that_overrides_earlier_switches(env):
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    _topics(app, slug)
    # The user switched the seeds' topic off before (as the trial scan saves it).
    conn = connect(env)
    try:
        stored = json.loads(conn.execute("SELECT topic_filters_json FROM profiles WHERE slug = ?", (slug,)).fetchone()[0])
        stored["topics"][0]["on"] = False
        stored["topics"].append({"id": "https://openalex.org/T10222", "display_name": "Genomics and Chromatin Dynamics", "count": 0, "on": False})
        conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE slug = ?", (json.dumps(stored), slug))
        conn.commit()
    finally:
        conn.close()
    after = {t["id"]: t for t in _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).json()}
    assert after["https://openalex.org/T10222"]["on"] is True       # wanted by the profile: on, whatever it was
    assert after["T10001"]["on"] is False


def test_profile_topics_are_refused_where_there_are_none(env):
    # An ORCID researcher has no profile document.
    app = _app(StubProsopia(_profile_with_document()), StubOrcidOpenAlex(WORKS))
    resp = _request(app, "POST", "/api/researchers/import", json={"source": "orcid", "ref": ORCID})
    assert resp.status_code == 200, resp.text
    slug = _draft_from(app, resp.json()["researcher_id"], name="From an ORCID")
    view = _request(app, "GET", f"/api/profiles/{slug}/researcher-topics").json()
    assert view["researcher_id"] is not None and view["topics"]["available"] is False and view["applied"] is False
    resp = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True})
    assert resp.status_code == 400 and "profile lists topics" in resp.json()["detail"]
    # Dropping what was never applied is harmless.
    assert _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": False}).status_code == 200

    # A draft with no researcher at all.
    resp = _request(app, "POST", "/api/profiles/draft", json={"name": "By hand"})
    assert resp.status_code == 200, resp.text
    bare = resp.json()["slug"]
    view = _request(app, "GET", f"/api/profiles/{bare}/researcher-topics").json()
    assert view["researcher_id"] is None and view["topics"]["available"] is False
    assert _request(app, "PUT", f"/api/profiles/{bare}/researcher-topics", json={"enabled": True}).status_code == 400
    assert _request(app, "GET", "/api/profiles/no-such-interest/researcher-topics").status_code == 404
    assert _request(app, "PUT", "/api/profiles/no-such-interest/researcher-topics", json={"enabled": True}).status_code == 404


def test_a_forgotten_researcher_takes_their_topics_with_them(env):
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    rid = _import_prosopia(app)
    slug = _draft_from(app, rid)
    assert _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).status_code == 200
    assert _request(app, "DELETE", f"/api/researchers/{rid}").status_code == 200
    topics = _topics(app, slug)
    assert set(topics) == {"T10001"} and topics["T10001"]["profile"] is None
    view = _request(app, "GET", f"/api/profiles/{slug}/researcher-topics").json()
    assert view["researcher_id"] is None and view["applied"] is False


def test_the_researcher_page_lists_the_profiles_topics(env):
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    rid = _import_prosopia(app)
    detail = _request(app, "GET", f"/api/researchers/{rid}").json()
    topics = detail["topics"]
    assert topics["available"] is True
    assert [t["name"] for t in topics["wanted"]] == ["Genomics and Chromatin Dynamics", "Cancer-related molecular mechanisms"]
    assert [t["method"] for t in topics["unwanted"]] == ["declared", "declared"]
    assert topics["wanted_labels"] == ["Lab notebooks"]
    assert topics["unwanted_labels"] == ["phylogenomics", "wet-lab-protocol-development"]

    # Another user's view of the same slug is their own import, not this one's.
    other = {"X-User-Email": "someone-else@example.com"}
    assert _request(app, "GET", f"/api/researchers/{rid}", headers=other).status_code == 404


def test_a_profile_without_interests_reads_as_unavailable(env):
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    rid = _import_prosopia(app)
    assert _request(app, "GET", f"/api/researchers/{rid}").json()["topics"] == {
        "available": False, "wanted": [], "unwanted": [], "wanted_labels": [], "unwanted_labels": [],
        "labels_unmapped": False}


def _import_prosopia(app) -> int:
    resp = _request(app, "POST", "/api/researchers/import", json={"source": "prosopia", "ref": "sheffield-nathan"})
    assert resp.status_code == 200, resp.text
    return resp.json()["researcher_id"]


# --- after the first review -------------------------------------------------------


def _commit(app, slug, selected):
    resp = _request(app, "POST", "/api/profiles", json={"slug": slug, "threshold": 0.5, "selected_topic_ids": selected})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _stored(env, slug):
    conn = connect(env)
    try:
        return json.loads(conn.execute("SELECT topic_filters_json FROM profiles WHERE slug = ?", (slug,)).fetchone()[0])
    finally:
        conn.close()


def test_applying_that_would_switch_every_topic_off_is_refused(env):
    """A profile that wants nothing the seeds have and rejects what they do:
    applied, a scan would search by nothing. Refused like save_topics."""
    profile = _stub_profile()
    profile.metadata = {"research_interests": [_topic("T10001", "Test Topic", weight=-0.5, method="declared")]}
    profile.rid = "0000-0001-5643-4068"
    app = _app(StubProsopia(profile), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    before = _topics(app, slug)
    resp = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True})
    assert resp.status_code == 409 and "every topic off" in resp.json()["detail"]
    assert _topics(app, slug) == before                      # nothing was written
    assert _request(app, "GET", f"/api/profiles/{slug}/researcher-topics").json()["applied"] is False


def test_removing_puts_the_seeds_topics_back_as_they_were(env):
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    assert _topics(app, slug)["T10001"]["on"] is True
    on = {t["id"]: t for t in _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).json()}
    assert on["T10001"]["on"] is False
    # Re-aggregation (step 3 reopened) keeps the pre-apply state around.
    _topics(app, slug)
    assert _stored(env, slug)["topics"][0]["on_before_profile"] is True
    off = {t["id"]: t for t in _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": False}).json()}
    assert set(off) == {"T10001"} and off["T10001"]["on"] is True and off["T10001"]["profile"] is None
    assert "on_before_profile" not in _stored(env, slug)["topics"][0]
    assert _stored(env, slug)["researcher_topics"] is False

    # Switched off by the user before the profile was applied: off again after it is removed.
    conn = connect(env)
    try:
        stored = _stored(env, slug)
        stored["topics"][0]["on"] = False
        conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE slug = ?", (json.dumps(stored), slug))
        conn.commit()
    finally:
        conn.close()
    _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True})
    assert _topics(app, slug)["T10001"]["on"] is False
    back = {t["id"]: t for t in _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": False}).json()}
    assert back["T10001"]["on"] is False


def test_a_saved_interest_keeps_the_profile_and_its_new_wanted_topics_start_on(env):
    """Commit carries the flag and the profile-only topics; on the saved
    interest, where new topics otherwise start off, a topic the profile
    gains later (a re-import) still starts on, and a not-wanted one off."""
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    rid = _import_prosopia(app)
    slug = _draft_from(app, rid)
    applied = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True})
    assert applied.status_code == 200
    # Committed with step 3's ticks as the profile set them (T10001 off).
    _commit(app, slug, [t["id"] for t in applied.json() if t["on"]])
    stored = _stored(env, slug)
    assert stored["researcher_topics"] is True
    assert {t["id"] for t in stored["topics"] if t.get("source") == "researcher"} == {"https://openalex.org/T10222", "https://openalex.org/T10515"}
    assert _request(app, "GET", f"/api/profiles/{slug}/researcher-topics").json()["applied"] is True

    # The profile gains a declared wanted topic and a declared unwanted one the seeds carry.
    conn = connect(env)
    try:
        doc = dict(DOCUMENT)
        doc["research_interests"] = DOCUMENT["research_interests"] + [_topic("T77777", "Newly Declared", weight=0.9, method="declared")]
        conn.execute("UPDATE researchers SET document_json = ? WHERE id = ?", (json.dumps(doc), rid))
        conn.commit()
    finally:
        conn.close()
    preview = {t["id"]: t for t in _request(app, "GET", f"/api/profiles/{slug}/topics/preview").json()}
    assert preview["https://openalex.org/T77777"]["on"] is True and preview["https://openalex.org/T77777"]["new"] is True
    assert preview["T10001"]["on"] is False
    # Saving the preview's ticks keeps the flag and the profile-only topics.
    saved = _request(app, "PUT", f"/api/profiles/{slug}/topics",
                     json={"selected_topic_ids": [i for i, t in preview.items() if t["on"]]})
    assert saved.status_code == 200, saved.text
    assert _stored(env, slug)["researcher_topics"] is True
    assert "https://openalex.org/T77777" in {t["id"] for t in saved.json()}


def test_the_profile_being_unreadable_for_a_moment_does_not_drop_it(env, monkeypatch):
    """The label mapper down, say: the aggregation cannot read the profile, so
    it folds nothing in this time, but the flag stays and the next read puts
    everything back. Only a forgotten researcher drops it."""
    from rag_lib.api.services import wizard
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    assert _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).status_code == 200
    applied = {t["id"]: t for t in _topics(app, slug).values()}
    monkeypatch.setattr(wizard, "_researcher_topics_for", lambda conn, pid: None)
    during = _topics(app, slug)
    # Kept as last stored: the profile-only topics, the marks, the states.
    assert {k: (v["on"], v["source"], v["profile"]) for k, v in during.items()} == \
        {k: (v["on"], v["source"], v["profile"]) for k, v in applied.items()}
    assert _stored(env, slug)["topics"][0]["on_before_profile"] is True
    view = _request(app, "GET", f"/api/profiles/{slug}/researcher-topics").json()
    assert view["applied"] is True and view["topics"]["available"] is False
    monkeypatch.undo()
    after = _topics(app, slug)
    assert "https://openalex.org/T10222" in after and after["T10001"]["profile"] == "not_interest"
    # And removing after the outage still puts the seeds' topic back.
    off = {t["id"]: t for t in _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": False}).json()}
    assert off["T10001"]["on"] is True


def test_kept_umls_topics_do_not_carry_old_marks():
    from rag_lib.api.services.wizard import _keep_stored_umls
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE profiles (id INTEGER PRIMARY KEY, topic_filters_json TEXT)")
    conn.execute("INSERT INTO profiles VALUES (1, ?)", (json.dumps({"topics": [
        {"id": "T9", "display_name": "Kept", "count": 1, "on": False, "source": "umls",
         "profile": "not_interest", "on_before_profile": True}]}),))
    out = _keep_stored_umls(conn, 1, {"topics": []})
    assert out["topics"] == [{"id": "T9", "display_name": "Kept", "count": 1, "on": False, "source": "umls"}]


def test_label_mapping_remembers_a_miss_and_backs_off_after_a_failure(monkeypatch):
    calls = []
    settings = _label_settings(monkeypatch, {"phylogenomics": ("https://openalex.org/T10434", "Genomics and Phylogenetic Studies", 0.74)})
    monkeypatch.setattr("rag_lib.embedders.get_embedder",
                        lambda m: (lambda text: calls.append(text) or np.array([ord(c) for c in text], dtype=np.float32)))
    doc = {"not_interests": ["phylogenomics", "nothing-like-a-topic"]}
    rt.with_labels(rt.read(doc), settings=settings)
    rt.with_labels(rt.read(doc), settings=settings)
    assert len(calls) == 2            # each label embedded once; the miss is remembered too

    def boom(*a, **k):
        raise RuntimeError("ollama down")
    monkeypatch.setattr("rag_lib.embedders.get_embedder", boom)
    rt.reset_label_cache()
    assert rt.with_labels(rt.read(doc), settings=settings) == {**rt.read(doc), "labels_unmapped": True}
    # Backing off: nothing is asked for a while, so the next call does not raise and maps nothing.
    monkeypatch.setattr("rag_lib.embedders.get_embedder", lambda m: (lambda text: calls.append(text) or np.array([ord(c) for c in text], dtype=np.float32)))
    n = len(calls)
    assert rt.with_labels(rt.read(doc), settings=settings) == {**rt.read(doc), "labels_unmapped": True}
    assert len(calls) == n
    # No labels at all: nothing to map, not an outage.
    assert rt.with_labels(rt.read({"research_interests": []}), settings=settings)["labels_unmapped"] is False
    rt.reset_label_cache()
    assert [t["label"] for t in rt.with_labels(rt.read(doc), settings=settings)["unwanted"]] == ["phylogenomics"]


# --- after the second review ------------------------------------------------------


def test_a_topic_marked_later_is_put_back_too(env):
    """A not-wanted topic a new seed brings in after the apply is marked and
    switched off by the next aggregation; removing the profile puts it back
    to what it would have started as (on, on a draft)."""
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    assert _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).status_code == 200
    conn = connect(env)
    try:
        oa_id = conn.execute("SELECT openalex_id FROM profile_seeds ps JOIN profiles p ON p.id = ps.profile_id "
                             "WHERE p.slug = ? AND openalex_id LIKE 'https://openalex.org/W%' LIMIT 1", (slug,)).fetchone()[0]
        raw = json.loads(conn.execute("SELECT topics_json FROM papers WHERE openalex_id = ?", (oa_id,)).fetchone()[0])
        raw["topics"].append({"id": "https://openalex.org/T11986", "display_name": "Scientific Computing", "score": 0.5,
                              "subfield": None, "field": None, "domain": None})
        conn.execute("UPDATE papers SET topics_json = ? WHERE openalex_id = ?", (json.dumps(raw), oa_id))
        conn.commit()
    finally:
        conn.close()
    later = _topics(app, slug)["https://openalex.org/T11986"]
    assert later["on"] is False and later["profile"] == "not_interest"
    stored = {t["id"]: t for t in _stored(env, slug)["topics"]}
    assert stored["https://openalex.org/T11986"]["on_before_profile"] is True
    off = {t["id"]: t for t in _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": False}).json()}
    assert off["https://openalex.org/T11986"]["on"] is True and off["https://openalex.org/T11986"]["profile"] is None


def test_re_applying_keeps_the_original_pre_apply_state(env):
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    for _ in range(2):
        assert _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).status_code == 200
    stored = {t["id"]: t for t in _stored(env, slug)["topics"]}
    assert stored["T10001"]["on"] is False and stored["T10001"]["on_before_profile"] is True
    off = {t["id"]: t for t in _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": False}).json()}
    assert off["T10001"]["on"] is True


def test_forgetting_the_researcher_takes_the_profile_off_a_saved_interest(env):
    """A saved interest's list is what every scan reads and nothing rebuilds
    it on its own, so forgetting the researcher cleans it there and then."""
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    rid = _import_prosopia(app)
    slug = _draft_from(app, rid)
    applied = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).json()
    _commit(app, slug, [t["id"] for t in applied if t["on"]])
    assert _request(app, "DELETE", f"/api/researchers/{rid}").status_code == 200
    stored = _stored(env, slug)
    assert stored["researcher_topics"] is False
    assert [t["id"] for t in stored["topics"]] == ["T10001"]
    assert stored["topics"][0]["on"] is True and "profile" not in stored["topics"][0]
    detail = _request(app, "GET", f"/api/profiles/{slug}/detail").json()
    assert [(t["id"], t["on"], t["profile"]) for t in detail["topics"]] == [("T10001", True, None)]


def test_a_list_read_that_overlaps_an_apply_does_not_undo_it(env, monkeypatch):
    """Step 3's GET rebuilds the list and writes it. One that started before
    "Use profile topics" and finished after used to write its older list over
    the apply. Now it notices the flag changed and rebuilds."""
    from rag_lib.api.services import wizard
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    _topics(app, slug)
    real = wizard._aggregate_topics
    hits = []

    def racing(conn, profile_id, **kw):
        out = real(conn, profile_id, **kw)
        if not hits:
            hits.append(1)
            # The apply lands on another connection while the read is in flight.
            other = connect(env)
            try:
                wizard.set_researcher_topics(other, user_id=1, slug=slug, enabled=True)
            finally:
                other.close()
        return out

    monkeypatch.setattr(wizard, "_aggregate_topics", racing)
    topics = _topics(app, slug)
    stored = _stored(env, slug)
    assert stored["researcher_topics"] is True
    assert "https://openalex.org/T10222" in topics and topics["T10001"]["on"] is False
    assert len(hits) == 1 and wizard._aggregate_topics is racing


def test_a_researcher_removed_underneath_restores_the_seeds_topics_on_the_next_read(env):
    """Forgetting through the API cleans the list at once; a row that went
    some other way (the repo, a hand edit) is met by the next aggregation:
    no profile to read and nobody to read it from, so the marked topics go
    back as they were and the flag goes."""
    from rag_lib.db.repos import researchers as repo
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    rid = _import_prosopia(app)
    slug = _draft_from(app, rid)
    assert _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).status_code == 200
    assert _topics(app, slug)["T10001"]["on"] is False
    conn = connect(env)
    try:
        repo.delete(conn, rid)
    finally:
        conn.close()
    topics = _topics(app, slug)
    assert set(topics) == {"T10001"} and topics["T10001"]["on"] is True and topics["T10001"]["profile"] is None
    assert _stored(env, slug).get("researcher_topics") is not True


# --- after the third review -------------------------------------------------------


def test_label_topics_stay_while_the_mapper_is_down(env, monkeypatch):
    """The profile's text labels map to topics only while the mapper is up.
    A read made while it is down must not drop those topics from the list,
    nor forget the user's switch on them."""
    from rag_lib.api.services import wizard
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    mapped = {"id": "https://openalex.org/T10434", "display_name": "Genomics and Phylogenetic Studies",
              "weight": None, "method": "label", "label": "Lab notebooks"}
    state = {"up": True}

    def with_labels(view, *, settings):
        if state["up"]:
            return {**view, "wanted": [*view["wanted"], dict(mapped)], "labels_unmapped": False}
        return {**view, "labels_unmapped": True}

    monkeypatch.setattr(wizard.researcher_topics, "with_labels", with_labels)
    on = {t["id"]: t for t in _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).json()}
    assert on[mapped["id"]]["source"] == "researcher" and on[mapped["id"]]["on"] is True
    # The user switches the label topic off.
    conn = connect(env)
    try:
        stored = _stored(env, slug)
        for t in stored["topics"]:
            if t["id"] == mapped["id"]:
                t["on"] = False
        conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE slug = ?", (json.dumps(stored), slug))
        conn.commit()
    finally:
        conn.close()
    state["up"] = False
    during = _topics(app, slug)
    assert during[mapped["id"]]["on"] is False and during[mapped["id"]]["source"] == "researcher"
    view = _request(app, "GET", f"/api/profiles/{slug}/researcher-topics").json()
    assert view["applied"] is True and view["topics"]["available"] is True and view["topics"]["labels_unmapped"] is True
    state["up"] = True
    assert _topics(app, slug)[mapped["id"]]["on"] is False


def test_removing_that_would_leave_nothing_on_is_refused_on_a_saved_interest(env):
    """Profile wants only a topic no seed has; the user switched the seeds'
    topic off and saved. Removing the profile would leave nothing on, and
    the saved list is what the next scan searches by."""
    profile = _stub_profile()
    profile.metadata = {"research_interests": [_topic("T10515", "Cancer-related", weight=0.5, method="declared")]}
    profile.rid = "0000-0001-5643-4068"
    app = _app(StubProsopia(profile), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    applied = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).json()
    assert {t["id"] for t in applied if t["on"]} == {"T10001", "https://openalex.org/T10515"}
    _commit(app, slug, ["https://openalex.org/T10515"])            # T10001 off
    resp = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": False})
    assert resp.status_code == 409 and "switch one on first" in resp.json()["detail"]
    stored = _stored(env, slug)
    assert stored["researcher_topics"] is True and any(t["on"] for t in stored["topics"])


def test_removing_what_was_never_applied_writes_nothing(env):
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    _topics(app, slug)
    before = _stored(env, slug)
    # A new seed topic arrives; a remove with nothing applied must not pull it in unseen.
    conn = connect(env)
    try:
        oa_id = conn.execute("SELECT openalex_id FROM profile_seeds ps JOIN profiles p ON p.id = ps.profile_id "
                             "WHERE p.slug = ? AND openalex_id LIKE 'https://openalex.org/W%' LIMIT 1", (slug,)).fetchone()[0]
        raw = json.loads(conn.execute("SELECT topics_json FROM papers WHERE openalex_id = ?", (oa_id,)).fetchone()[0])
        raw["topics"].append({"id": "https://openalex.org/T12345", "display_name": "Something New", "score": 0.5,
                              "subfield": None, "field": None, "domain": None})
        conn.execute("UPDATE papers SET topics_json = ? WHERE openalex_id = ?", (json.dumps(raw), oa_id))
        conn.commit()
    finally:
        conn.close()
    resp = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": False})
    assert resp.status_code == 200 and [t["id"] for t in resp.json()] == ["T10001"]
    assert _stored(env, slug) == before


def test_a_read_after_the_researcher_vanished_rebuilds_once(env, monkeypatch):
    from rag_lib.api.services import wizard
    from rag_lib.db.repos import researchers as repo
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    rid = _import_prosopia(app)
    slug = _draft_from(app, rid)
    assert _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).status_code == 200
    conn = connect(env)
    try:
        repo.delete(conn, rid)
    finally:
        conn.close()
    calls = []
    real = wizard._aggregate_topics
    monkeypatch.setattr(wizard, "_aggregate_topics", lambda *a, **k: calls.append(1) or real(*a, **k))
    _topics(app, slug)
    assert len(calls) == 1
    assert _stored(env, slug)["researcher_topics"] is False


# --- after the fourth review ------------------------------------------------------


def _switch(env, slug, states):
    """Set stored on/off states by id, as a trial scan's saved ticks would."""
    conn = connect(env)
    try:
        stored = json.loads(conn.execute("SELECT topic_filters_json FROM profiles WHERE slug = ?", (slug,)).fetchone()[0])
        for t in stored["topics"]:
            if t["id"] in states:
                t["on"] = states[t["id"]]
        conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE slug = ?", (json.dumps(stored), slug))
        conn.commit()
    finally:
        conn.close()


def test_forgetting_the_researcher_keeps_the_profiles_topics_where_nothing_else_is_on(env):
    """The user kept only the profile's topics on. Putting the seeds' topics
    back as they were (off) would leave the saved interest searching by
    nothing, so the profile's stay -- as ordinary topics, no marks."""
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    rid = _import_prosopia(app)
    slug = _draft_from(app, rid)
    # The seed's topic is switched off before the profile is applied (a trial
    # scan saves the ticks), so it was off before and goes back to off.
    _topics(app, slug)
    _switch(env, slug, {"T10001": False})
    assert _topics(app, slug)["T10001"]["on"] is False
    applied = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).json()
    only_profile = [t["id"] for t in applied if t["on"]]
    assert only_profile and "T10001" not in only_profile
    _commit(app, slug, only_profile)
    assert _request(app, "DELETE", f"/api/researchers/{rid}").status_code == 200
    stored = _stored(env, slug)
    assert stored["researcher_topics"] is False
    assert {t["id"] for t in stored["topics"] if t["on"]} == set(only_profile)
    assert all("profile" not in t and "on_before_profile" not in t and t.get("source") != "researcher"
               for t in stored["topics"])
    detail = _request(app, "GET", f"/api/profiles/{slug}/detail").json()
    assert {t["id"] for t in detail["topics"] if t["on"]} == set(only_profile)
    assert all(t["profile"] is None for t in detail["topics"])


def test_a_topic_the_profile_stops_rejecting_goes_back_to_what_it_was(env):
    """The owner's document is re-read without the rejection: the topic the
    profile had switched off is on again, with no mark and nothing left
    behind for "Remove" to restore; a switch the user made since stays."""
    profile = _stub_profile()
    profile.metadata = {"research_interests": [
        _topic("T10001", "Test Topic", weight=-0.5, method="declared"),
        _topic("T10222", "Genomics and Chromatin Dynamics", weight=0.5, method="declared"),
    ]}
    profile.rid = "0000-0001-5643-4068"
    app = _app(StubProsopia(profile), _stub_openalex())
    rid = _import_prosopia(app)
    slug = _draft_from(app, rid)
    applied = {t["id"]: t for t in _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).json()}
    assert applied["T10001"]["on"] is False and applied["T10001"]["profile"] == "not_interest"

    profile.metadata = {"research_interests": [_topic("T10222", "Genomics and Chromatin Dynamics", weight=0.5, method="declared")]}
    assert _request(app, "POST", "/api/researchers/import", json={"source": "prosopia", "ref": "sheffield-nathan"}).status_code == 200
    topics = _topics(app, slug)
    assert topics["T10001"]["on"] is True and topics["T10001"]["profile"] is None
    assert "on_before_profile" not in {t["id"]: t for t in _stored(env, slug)["topics"]}["T10001"]
    removed = {t["id"]: t for t in _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": False}).json()}
    assert removed["T10001"]["on"] is True


def test_a_topic_the_profile_stops_rejecting_keeps_the_users_own_switch(env):
    profile = _stub_profile()
    profile.metadata = {"research_interests": [
        _topic("T10001", "Test Topic", weight=-0.5, method="declared"),
        _topic("T10222", "Genomics and Chromatin Dynamics", weight=0.5, method="declared"),
    ]}
    profile.rid = "0000-0001-5643-4068"
    app = _app(StubProsopia(profile), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True})
    # The user switches the rejected topic back on themselves (a trial scan saves the ticks).
    _switch(env, slug, {"T10001": True})
    profile.metadata = {"research_interests": [_topic("T10222", "Genomics and Chromatin Dynamics", weight=0.5, method="declared")]}
    _request(app, "POST", "/api/researchers/import", json={"source": "prosopia", "ref": "sheffield-nathan"})
    assert _topics(app, slug)["T10001"]["on"] is True


def test_a_rebuild_overtaken_every_time_is_refused_rather_than_written(env, monkeypatch):
    """Three rebuilds, each overtaken by an apply or remove: the last one is
    not written anyway (it could undo one); the client gets a 409."""
    from rag_lib.api.services import wizard as wizard_service

    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    before = _stored(env, slug)
    real = wizard_service._stored_topic_filters
    state = {"flag": False}

    def flipping(conn, profile_id):
        out = dict(real(conn, profile_id))
        if conn.in_transaction:                 # the read under the write lock
            state["flag"] = not state["flag"]
            out["researcher_topics"] = state["flag"]
        return out

    monkeypatch.setattr(wizard_service, "_stored_topic_filters", flipping)
    resp = _request(app, "GET", f"/api/profiles/draft/{slug}/topics")
    assert resp.status_code == 409 and "try again" in resp.json()["detail"]
    monkeypatch.undo()
    assert _stored(env, slug) == before


def test_adopted_topics_survive_the_next_rebuild_and_a_save(env):
    """After the forget, a rebuild from the seeds dropped the adopted topics
    on the first save; the preview showed only the seed topics, all off."""
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    rid = _import_prosopia(app)
    slug = _draft_from(app, rid)
    _topics(app, slug)
    _switch(env, slug, {"T10001": False})
    applied = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).json()
    only_profile = [t["id"] for t in applied if t["on"]]
    _commit(app, slug, only_profile)
    assert _request(app, "DELETE", f"/api/researchers/{rid}").status_code == 200
    preview = {t["id"]: t for t in _request(app, "GET", f"/api/profiles/{slug}/topics/preview").json()}
    assert set(only_profile) <= set(preview) and all(preview[i]["on"] for i in only_profile)
    saved = _request(app, "PUT", f"/api/profiles/{slug}/topics", json={"selected_topic_ids": only_profile})
    assert saved.status_code == 200, saved.text
    stored = _stored(env, slug)
    assert {t["id"] for t in stored["topics"] if t["on"]} == set(only_profile)
    assert all(t.get("source") == "adopted" for t in stored["topics"] if t["id"] in only_profile)
