"""A researcher's declared interests as topics of an interest built from them.

A Prosopia profile says, in ``research_interests``, which OpenAlex topics
its owner wants (declared, or inferred from their papers) and which they do
not (a negative weight), plus text concepts for what maps to no topic. The
researcher row stores that document; an interest built from the researcher
can take the topics from it: the wanted ones join its list switched on, and
each not-wanted one is filtered out (switched off), kept as the other
sources have it, or added, as the user chooses -- kept by default when
another source lists it, filtered otherwise. Offline: the Prosopia and
OpenAlex stubs of the import suites.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from rag_lib import researcher_topics as rt
from rag_lib.api.mappers import topic_filters_to_topics
from rag_lib.db import connect
from tests.edit_helpers import edit_saved
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
    # By default a not-wanted topic the seeds carry is kept (the user decides);
    # one nothing else lists is filtered.
    kept = rt.apply(seeds, rt.read(DOCUMENT))
    assert kept[rt.CHOICES] == {"T10001": rt.KEEP, "T11986": rt.FILTER}
    assert {t["id"]: t.get("profile") for t in kept["topics"]}["T10001"] is None
    out = rt.apply({**seeds, rt.CHOICES: {"T10001": rt.FILTER}}, rt.read(DOCUMENT))
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
    assert back[rt.CHOICES] == out[rt.CHOICES]                        # kept for the next apply
    assert rt.CHOICES not in rt.adopt(out)
    # "add" treats a not-wanted topic as a wanted one, listed or not.
    added = {t["id"]: t for t in rt.apply({**seeds, rt.CHOICES: {"T11986": rt.ADD}}, rt.read(DOCUMENT))["topics"]}
    assert added["https://openalex.org/T11986"]["profile"] == rt.INTEREST
    assert added["https://openalex.org/T11986"]["source"] == rt.SOURCE


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
    assert view["unmatched_labels"] == ["wet-lab-protocol-development"]                    # only the one too far


def test_a_label_mapping_to_both_sides_is_kept_on_both_and_a_dead_embedder_changes_nothing(monkeypatch):
    """The two sides disagree on the topic: it stays on both, so the topic
    step shows the not-wanted one as overlapping and the user settles it
    (it used to be dropped without a word)."""
    doc = {"interests": ["chromatin"], "not_interests": ["chromatin biology"],
           "research_interests": [_topic("T2", "Genomics")]}
    settings = _label_settings(monkeypatch, {
        "chromatin": ("https://openalex.org/T1", "Chromatin", 0.8),
        "chromatin biology": ("https://openalex.org/T1", "Chromatin", 0.8),
        "genomics data": ("https://openalex.org/T2", "Genomics", 0.9),
    })
    view = rt.with_labels(rt.read(doc), settings=settings)
    assert [t["id"] for t in view["wanted"]] == ["https://openalex.org/T2", "https://openalex.org/T1"]
    assert [(t["id"], t["label"], t["similarity"]) for t in view["unwanted"]] == [
        ("https://openalex.org/T1", "chromatin biology", 0.8)]
    rows = rt.not_interest_rows(view, {"topics": []})
    assert [(r["id"], r["overlap"], r["choice"]) for r in rows] == [
        ("https://openalex.org/T1", [rt.PROFILE_INTEREST], rt.KEEP)]
    assert view["unmatched_labels"] == []
    # A not-wanted label on a topic the profile wants outright is kept too.
    view = rt.with_labels(rt.read({**doc, "not_interests": ["genomics data"]}), settings=settings)
    assert [t["id"] for t in view["unwanted"]] == ["https://openalex.org/T2"]
    # A wanted label on a topic the profile rejects outright is not added to the wanted side.
    rejecting = {"interests": ["genomics data"], "research_interests": [_topic("T2", "Genomics", weight=-0.5)]}
    view = rt.with_labels(rt.read(rejecting), settings=settings)
    assert view["wanted"] == [] and [t["id"] for t in view["unwanted"]] == ["https://openalex.org/T2"]

    def boom(*a, **k):
        raise RuntimeError("ollama down")
    monkeypatch.setattr("rag_lib.embedders.get_embedder", boom)
    rt.reset_label_cache()
    # The view comes back as read, saying its labels are not mapped; with
    # UMLS off there is nothing to map, which is not an outage.
    assert rt.with_labels(rt.read(DOCUMENT), settings=settings) == {**rt.read(DOCUMENT), "labels_unmapped": True, "label_matching": True}
    off = rt.with_labels(rt.read(DOCUMENT), settings=SimpleNamespace(RADAR_UMLS_ENABLED=False))
    assert off == {**rt.read(DOCUMENT), "labels_unmapped": False, "label_matching": False,
                   "unmatched_labels": rt.read(DOCUMENT)["unwanted_labels"]}


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


def _choose(app, slug, topic_id, choice="filter"):
    """What one not-wanted topic does on the interest; the list as returned."""
    resp = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics/not-interest",
                    json={"topic_id": topic_id, "choice": choice})
    assert resp.status_code == 200, resp.text
    return {t["id"]: t for t in resp.json()}


def _edit(app, slug, *, save=True):
    """Edit interest on saved interest ``slug``: yields the copy's slug
    (FUNCTIONAL_SPEC E -- a saved interest changes only through a copy)."""
    return edit_saved(lambda method, path, **kw: _request(app, method, path, **kw), slug, save=save)


def _apply_filtering(app, slug, topic_id="T10001"):
    """Apply the profile and filter out the not-wanted topic the seeds carry
    (kept by default, as another source lists it)."""
    resp = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True})
    assert resp.status_code == 200, resp.text
    return _choose(app, slug, topic_id, "filter")


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

    rows = {r["id"]: r for r in view["topics"]["not_interest"]}
    # The seeds carry T10001 too: kept unless the user filters it. Nothing else has T11986: filtered.
    assert (rows["https://openalex.org/T10001"]["overlap"], rows["https://openalex.org/T10001"]["choice"]) == (["openalex"], "keep")
    assert (rows["https://openalex.org/T11986"]["overlap"], rows["https://openalex.org/T11986"]["choice"]) == ([], "filter")
    assert rows["https://openalex.org/T10001"]["in_list"] is True and rows["https://openalex.org/T11986"]["in_list"] is False

    resp = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True})
    assert resp.status_code == 200, resp.text
    kept = {t["id"]: t for t in resp.json()}
    assert kept["T10001"]["on"] is True and kept["T10001"]["profile"] is None
    after = _choose(app, slug, "https://openalex.org/T10001", "filter")
    assert after["T10001"]["on"] is False and after["T10001"]["profile"] == "not_interest"
    assert {r["id"]: r["choice"] for r in _request(app, "GET", f"/api/profiles/{slug}/researcher-topics").json()["topics"]["not_interest"]} == {
        "https://openalex.org/T10001": "filter", "https://openalex.org/T11986": "filter"}
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
        "labels_unmapped": False, "unmatched_labels": [], "label_matching": False, "not_interest": []}


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
    # Applied, the seeds' topic is kept (they list it); filtering it out is what is refused.
    assert _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).status_code == 200
    before = _stored(env, slug)
    resp = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics/not-interest",
                    json={"topic_id": "T10001", "choice": "filter"})
    assert resp.status_code == 409 and "every topic off" in resp.json()["detail"]
    assert _stored(env, slug) == before                      # nothing was written


def test_removing_puts_the_seeds_topics_back_as_they_were(env):
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    assert _topics(app, slug)["T10001"]["on"] is True
    on = _apply_filtering(app, slug)
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


def test_a_saved_interest_keeps_the_profile_it_applied_when_the_researcher_changes(env):
    """Commit carries the flag and the profile-only topics. The researcher's
    document changing afterwards (a re-import gains a wanted topic) does not
    reach the saved interest: it reads the copy of the profile it took when
    it applied the topics (FUNCTIONAL_SPEC E.7, X-14). It used to fold the
    new topic in, switched on. A topics save (on an Edit interest copy,
    spec E) keeps the flag and the profile-only topics."""
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    rid = _import_prosopia(app)
    slug = _draft_from(app, rid)
    applied = _apply_filtering(app, slug)
    # Committed with step 3's ticks as the profile set them (T10001 off).
    _commit(app, slug, [i for i, t in applied.items() if t["on"]])
    stored = _stored(env, slug)
    assert stored["researcher_topics"] is True
    assert {t["id"] for t in stored["topics"] if t.get("source") == "researcher"} == {"https://openalex.org/T10222", "https://openalex.org/T10515"}
    assert _request(app, "GET", f"/api/profiles/{slug}/researcher-topics").json()["applied"] is True

    # The researcher's profile gains a declared wanted topic.
    conn = connect(env)
    try:
        doc = dict(DOCUMENT)
        doc["research_interests"] = DOCUMENT["research_interests"] + [_topic("T77777", "Newly Declared", weight=0.9, method="declared")]
        conn.execute("UPDATE researchers SET document_json = ? WHERE id = ?", (json.dumps(doc), rid))
        conn.commit()
    finally:
        conn.close()
    preview = {t["id"]: t for t in _request(app, "GET", f"/api/profiles/{slug}/topics/preview").json()}
    assert "https://openalex.org/T77777" not in preview
    assert preview["T10001"]["on"] is False
    view = _request(app, "GET", f"/api/profiles/{slug}/researcher-topics").json()
    assert view["applied"] is True
    assert "https://openalex.org/T77777" not in {t["id"] for t in view["topics"]["wanted"]}
    assert _stored(env, slug) == stored
    # Saving the preview's ticks keeps the flag and the profile-only topics.
    with _edit(app, slug) as copy:
        saved = _request(app, "PUT", f"/api/profiles/{copy}/topics",
                         json={"selected_topic_ids": [i for i, t in preview.items() if t["on"]]})
        assert saved.status_code == 200, saved.text
    after = _stored(env, slug)
    assert after["researcher_topics"] is True
    assert {t["id"] for t in after["topics"] if t.get("source") == "researcher"} == {"https://openalex.org/T10222", "https://openalex.org/T10515"}
    assert "https://openalex.org/T77777" not in {t["id"] for t in after["topics"]}


def test_the_profile_being_unreadable_for_a_moment_does_not_drop_it(env, monkeypatch):
    """The label mapper down, say: the aggregation cannot read the profile, so
    it folds nothing in this time, but the flag stays and the next read puts
    everything back. Only a forgotten researcher drops it."""
    from rag_lib.api.services import wizard
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    _apply_filtering(app, slug)
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
    assert rt.with_labels(rt.read(doc), settings=settings) == {**rt.read(doc), "labels_unmapped": True, "label_matching": True}
    # Backing off: nothing is asked for a while, so the next call does not raise and maps nothing.
    monkeypatch.setattr("rag_lib.embedders.get_embedder", lambda m: (lambda text: calls.append(text) or np.array([ord(c) for c in text], dtype=np.float32)))
    n = len(calls)
    assert rt.with_labels(rt.read(doc), settings=settings) == {**rt.read(doc), "labels_unmapped": True, "label_matching": True}
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
    _apply_filtering(app, slug)
    assert _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).status_code == 200
    stored = {t["id"]: t for t in _stored(env, slug)["topics"]}
    assert stored["T10001"]["on"] is False and stored["T10001"]["on_before_profile"] is True
    off = {t["id"]: t for t in _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": False}).json()}
    assert off["T10001"]["on"] is True


def test_forgetting_the_researcher_leaves_a_saved_interests_topics_as_they_were(env):
    """A saved interest's list is what every scan reads. Forgetting the
    researcher leaves it exactly as the user saved it, marks and all: the
    interest reads the copy of the profile it took when it applied the
    topics (FUNCTIONAL_SPEC E.7, X-15). Before spec E the forget took the
    marks off and made the profile's topics the interest's own
    (``adopted``); round 12 before that put the rejected one back on."""
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    rid = _import_prosopia(app)
    slug = _draft_from(app, rid)
    applied = _apply_filtering(app, slug)
    on_at_commit = [i for i, t in applied.items() if t["on"]]
    assert "T10001" not in on_at_commit and len(on_at_commit) == 2
    _commit(app, slug, on_at_commit)
    before = _stored(env, slug)
    detail_before = _request(app, "GET", f"/api/profiles/{slug}/detail").json()["topics"]
    assert _request(app, "DELETE", f"/api/researchers/{rid}").status_code == 200
    stored = _stored(env, slug)
    assert stored == before and stored["researcher_topics"] is True
    assert [t["id"] for t in stored["topics"] if t["on"]] == on_at_commit
    assert {t["source"] for t in stored["topics"] if t["id"] in on_at_commit} == {"researcher"}
    detail = _request(app, "GET", f"/api/profiles/{slug}/detail").json()
    assert detail["topics"] == detail_before
    assert [t["id"] for t in detail["topics"] if t["on"]] == on_at_commit
    # Still applied, read from the interest's copy of the profile.
    view = _request(app, "GET", f"/api/profiles/{slug}/researcher-topics").json()
    assert view["researcher_id"] is None and view["applied"] is True
    assert view["researcher_name"] == "Nathan C. Sheffield" and view["topics"]["available"] is True
    # And a rebuild (the edit page's preview) does not take them off either.
    preview = {t["id"]: t for t in _request(app, "GET", f"/api/profiles/{slug}/topics/preview").json()}
    assert {i for i, t in preview.items() if t["on"]} == set(on_at_commit)
    assert preview["T10001"]["profile"] == "not_interest"


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
    assert "https://openalex.org/T10222" in topics and topics["T10001"]["on"] is True
    assert stored["not_interest_choices"] == {"T10001": "keep", "T11986": "filter"}
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
    _apply_filtering(app, slug)
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
    before = _stored(env, slug)
    # A saved interest changes on its Edit interest copy, which keeps the
    # saved interest's rules (FUNCTIONAL_SPEC E).
    with _edit(app, slug, save=False) as copy:
        copy_before = _stored(env, copy)
        resp = _request(app, "PUT", f"/api/profiles/{copy}/researcher-topics", json={"enabled": False})
        assert resp.status_code == 409 and "switch one on first" in resp.json()["detail"]
        assert _stored(env, copy) == copy_before
    stored = _stored(env, slug)
    assert stored == before
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
    nothing, so the profile's stay. Since FUNCTIONAL_SPEC E.7 (X-15) the
    forget does not touch the saved interest at all: the profile's topics
    stay the profile's, read from the interest's copy of it (they were
    turned into ordinary, unmarked topics before)."""
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
    before = _stored(env, slug)
    assert _request(app, "DELETE", f"/api/researchers/{rid}").status_code == 200
    stored = _stored(env, slug)
    assert stored == before and stored["researcher_topics"] is True
    assert {t["id"] for t in stored["topics"] if t["on"]} == set(only_profile)
    assert all(t.get("source") == "researcher" and t.get("profile") == "interest"
               for t in stored["topics"] if t["id"] in only_profile)
    detail = _request(app, "GET", f"/api/profiles/{slug}/detail").json()
    assert {t["id"] for t in detail["topics"] if t["on"]} == set(only_profile)
    assert {t["id"] for t in detail["topics"] if t["profile"] == "interest"} == set(only_profile)


def test_a_topic_the_profile_stops_rejecting_goes_back_to_what_it_was(env):
    """The owner's document is re-imported without the rejection. The
    interest took a copy of the profile when it applied it, and a re-import
    does not reach that copy (FUNCTIONAL_SPEC E.7): the topic stays off,
    marked. Removing the profile and applying it again takes the profile
    as it is now: the topic is on again, with no mark and nothing left
    behind for "Remove" to restore. (Before E.7 the re-import alone did
    this, on the next read.)"""
    profile = _stub_profile()
    profile.metadata = {"research_interests": [
        _topic("T10001", "Test Topic", weight=-0.5, method="declared"),
        _topic("T10222", "Genomics and Chromatin Dynamics", weight=0.5, method="declared"),
    ]}
    profile.rid = "0000-0001-5643-4068"
    app = _app(StubProsopia(profile), _stub_openalex())
    rid = _import_prosopia(app)
    slug = _draft_from(app, rid)
    applied = _apply_filtering(app, slug)
    assert applied["T10001"]["on"] is False and applied["T10001"]["profile"] == "not_interest"

    profile.metadata = {"research_interests": [_topic("T10222", "Genomics and Chromatin Dynamics", weight=0.5, method="declared")]}
    assert _request(app, "POST", "/api/researchers/import", json={"source": "prosopia", "ref": "sheffield-nathan"}).status_code == 200
    topics = _topics(app, slug)
    assert topics["T10001"]["on"] is False and topics["T10001"]["profile"] == "not_interest"
    assert _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": False}).status_code == 200
    again = {t["id"]: t for t in _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).json()}
    assert again["T10001"]["on"] is True and again["T10001"]["profile"] is None
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
    _apply_filtering(app, slug)
    # The user switches the rejected topic back on themselves (a trial scan saves the ticks).
    _switch(env, slug, {"T10001": True})
    profile.metadata = {"research_interests": [_topic("T10222", "Genomics and Chromatin Dynamics", weight=0.5, method="declared")]}
    _request(app, "POST", "/api/researchers/import", json={"source": "prosopia", "ref": "sheffield-nathan"})
    assert _topics(app, slug)["T10001"]["on"] is True


def test_a_rebuild_overtaken_every_time_is_refused_rather_than_written(env, monkeypatch):
    """Three rebuilds, each overtaken by another write: the last one is not
    written anyway (it could undo one); the client gets a 409."""
    from rag_lib.api.services import wizard as wizard_service

    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    before = _stored(env, slug)
    real = wizard_service._raw_topic_filters
    state = {"n": 0}

    def changing(conn, profile_id):
        out = real(conn, profile_id)
        if conn.in_transaction:                 # the read under the write lock
            state["n"] += 1
            return f"{out} {state['n']}"
        return out

    monkeypatch.setattr(wizard_service, "_raw_topic_filters", changing)
    resp = _request(app, "GET", f"/api/profiles/draft/{slug}/topics")
    assert resp.status_code == 409 and "try again" in resp.json()["detail"]
    monkeypatch.undo()
    assert _stored(env, slug) == before


def test_adopted_topics_survive_the_next_rebuild_and_a_save(env):
    """After the forget, a rebuild from the seeds dropped the adopted topics
    on the first save; the preview showed only the seed topics, all off.

    Since FUNCTIONAL_SPEC E.7 a forget no longer adopts on a saved interest
    (it keeps its copy of the profile); an interest forgotten before that
    still has its adopted list, written here as that forget wrote it. The
    save goes through an Edit interest copy (spec E)."""
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    rid = _import_prosopia(app)
    slug = _draft_from(app, rid)
    _topics(app, slug)
    _switch(env, slug, {"T10001": False})
    applied = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).json()
    only_profile = [t["id"] for t in applied if t["on"]]
    _commit(app, slug, only_profile)
    # The forget as it was before E.7: the profile's topics the interest's own.
    _write_stored(env, slug, rt.adopt(_stored(env, slug)))
    conn = connect(env)
    try:
        conn.execute("UPDATE profiles SET researcher_id = NULL, researcher_doc_json = NULL WHERE slug = ?", (slug,))
        conn.commit()
    finally:
        conn.close()
    assert _request(app, "DELETE", f"/api/researchers/{rid}").status_code == 200
    assert all(t.get("source") == "adopted" for t in _stored(env, slug)["topics"] if t["id"] in only_profile)
    preview = {t["id"]: t for t in _request(app, "GET", f"/api/profiles/{slug}/topics/preview").json()}
    assert set(only_profile) <= set(preview) and all(preview[i]["on"] for i in only_profile)
    with _edit(app, slug) as copy:
        saved = _request(app, "PUT", f"/api/profiles/{copy}/topics", json={"selected_topic_ids": only_profile})
        assert saved.status_code == 200, saved.text
    stored = _stored(env, slug)
    assert {t["id"] for t in stored["topics"] if t["on"]} == set(only_profile)
    assert all(t.get("source") == "adopted" for t in stored["topics"] if t["id"] in only_profile)


# --- not-wanted topics: the user's choice --------------------------------------------


def _rows(app, slug):
    view = _request(app, "GET", f"/api/profiles/{slug}/researcher-topics").json()
    return {r["id"]: r for r in view["topics"]["not_interest"]}


def test_a_not_wanted_topic_can_be_added_and_taken_back(env):
    """A not-wanted topic nothing else lists (filtered by default) can be
    added as a topic of the interest's own, switched on; back to filtered,
    it leaves the list again."""
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    assert _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).status_code == 200
    t11986 = "https://openalex.org/T11986"
    added = _choose(app, slug, t11986, "add")
    assert added[t11986]["on"] is True and added[t11986]["source"] == "researcher" and added[t11986]["profile"] == "interest"
    assert _rows(app, slug)[t11986]["choice"] == "add" and _rows(app, slug)[t11986]["in_list"] is True
    assert _topics(app, slug)[t11986]["on"] is True                # a re-read keeps it
    gone = _choose(app, slug, t11986, "filter")
    assert t11986 not in gone and _rows(app, slug)[t11986]["choice"] == "filter"


def test_choosing_moves_only_that_topic_on_a_saved_interest(env):
    """On a saved interest the stored list is what scans search by: setting
    one not-wanted topic's choice changes that topic and nothing else --
    a switch the user made on another stays."""
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    applied = {t["id"]: t for t in _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).json()}
    on = [i for i, t in applied.items() if t["on"]]
    _commit(app, slug, on)
    # The user switches a wanted topic off on the edit page -- on its copy
    # of the saved interest, which keeps the saved interest's rules (spec E).
    with _edit(app, slug) as copy:
        saved = _request(app, "PUT", f"/api/profiles/{copy}/topics",
                         json={"selected_topic_ids": [i for i in on if i != "https://openalex.org/T10515"]})
        assert saved.status_code == 200, saved.text
        before = {t["id"]: t["on"] for t in _stored(env, copy)["topics"]}
        after = _choose(app, copy, "T10001", "filter")
        assert after["T10001"]["on"] is False and after["T10001"]["profile"] == "not_interest"
        assert {i: t["on"] for i, t in after.items() if i != "T10001"} == {i: v for i, v in before.items() if i != "T10001"}
        back = _choose(app, copy, "T10001", "keep")
        assert back["T10001"]["on"] is True and back["T10001"]["profile"] is None     # back to its state before the filter
        assert back["https://openalex.org/T10515"]["on"] is False
        on_copy = {t["id"]: t["on"] for t in _stored(env, copy)["topics"]}
    # Saved: the interest's list is the copy's.
    assert {t["id"]: t["on"] for t in _stored(env, slug)["topics"]} == on_copy


def test_choices_survive_removing_and_applying_again(env):
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    _apply_filtering(app, slug)
    assert _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": False}).status_code == 200
    assert _topics(app, slug)["T10001"]["on"] is True
    assert _rows(app, slug)["https://openalex.org/T10001"]["choice"] == "filter"   # remembered, shown before applying
    again = {t["id"]: t for t in _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).json()}
    assert again["T10001"]["on"] is False and again["T10001"]["profile"] == "not_interest"


def test_a_list_applied_before_choices_keeps_filtering(env):
    """An interest the profile was applied to before choices were recorded
    had every not-wanted topic it listed switched off. Without a record its
    marks read as "filter": a rebuild must not switch them back on."""
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    _apply_filtering(app, slug)
    conn = connect(env)
    try:
        stored = _stored(env, slug)
        del stored["not_interest_choices"]
        conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE slug = ?", (json.dumps(stored), slug))
        conn.commit()
    finally:
        conn.close()
    assert _rows(app, slug)["https://openalex.org/T10001"]["choice"] == "filter"
    topics = _topics(app, slug)
    assert topics["T10001"]["on"] is False and topics["T10001"]["profile"] == "not_interest"
    assert _stored(env, slug)["not_interest_choices"]["T10001"] == "filter"


def test_a_choice_is_refused_where_it_means_nothing(env):
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    url = f"/api/profiles/{slug}/researcher-topics/not-interest"
    # Before the profile is applied.
    resp = _request(app, "PUT", url, json={"topic_id": "T10001", "choice": "filter"})
    assert resp.status_code == 400 and "apply the profile" in resp.json()["detail"]
    assert _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).status_code == 200
    # A topic the profile wants, and one it says nothing about.
    for tid in ("https://openalex.org/T10222", "T99999"):
        resp = _request(app, "PUT", url, json={"topic_id": tid, "choice": "filter"})
        assert resp.status_code == 400 and "does not want" in resp.json()["detail"]
    assert _request(app, "PUT", url, json={"topic_id": "T10001", "choice": "drop"}).status_code == 422
    assert _request(app, "PUT", "/api/profiles/no-such/researcher-topics/not-interest",
                    json={"topic_id": "T10001", "choice": "filter"}).status_code == 404
    other = {"X-User-Email": "someone-else@example.com"}
    assert _request(app, "PUT", url, json={"topic_id": "T10001", "choice": "filter"}, headers=other).status_code == 404


# --- not-wanted choices: after the first review ------------------------------------


def _add_seed_topics(env, slug, topics):
    """One seed of the interest now also carries these ``(code, name)`` topics."""
    conn = connect(env)
    try:
        oa_id = conn.execute("SELECT openalex_id FROM profile_seeds ps JOIN profiles p ON p.id = ps.profile_id "
                             "WHERE p.slug = ? AND openalex_id LIKE 'https://openalex.org/W%' LIMIT 1", (slug,)).fetchone()[0]
        raw = json.loads(conn.execute("SELECT topics_json FROM papers WHERE openalex_id = ?", (oa_id,)).fetchone()[0])
        for code, name in topics:
            raw["topics"].append({"id": f"https://openalex.org/{code}", "display_name": name, "score": 0.5,
                                  "subfield": None, "field": None, "domain": None})
        conn.execute("UPDATE papers SET topics_json = ? WHERE openalex_id = ?", (json.dumps(raw), oa_id))
        conn.commit()
    finally:
        conn.close()


def _write_stored(env, slug, stored):
    conn = connect(env)
    try:
        conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE slug = ?", (json.dumps(stored), slug))
        conn.commit()
    finally:
        conn.close()


def test_a_list_read_that_overlaps_a_choice_does_not_undo_it(env, monkeypatch):
    """The step-3 rebuild compared only the applied flag before writing: a
    choice made while it ran was written over, with the 200 saying it was
    set. The whole stored list is compared now."""
    from rag_lib.api.services import wizard
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    assert _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).status_code == 200
    real = wizard._aggregate_topics
    hits = []

    def racing(conn, profile_id, **kw):
        out = real(conn, profile_id, **kw)
        if not hits:
            hits.append(1)
            other = connect(env)
            try:
                wizard.set_not_interest_choice(other, user_id=1, slug=slug, topic_id="T10001", choice="filter")
            finally:
                other.close()
        return out

    monkeypatch.setattr(wizard, "_aggregate_topics", racing)
    topics = _topics(app, slug)
    assert hits and _stored(env, slug)["not_interest_choices"]["T10001"] == "filter"
    assert topics["T10001"]["on"] is False and topics["T10001"]["profile"] == "not_interest"


def test_a_choice_on_a_saved_interest_is_made_while_the_mapper_is_down(env, monkeypatch):
    """The stored list used to come back unchanged while the profile's labels
    could not be mapped -- the choice dropped, with a 200."""
    from rag_lib.api.services import wizard
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    applied = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).json()
    _commit(app, slug, [t["id"] for t in applied if t["on"]])
    monkeypatch.setattr(wizard.researcher_topics, "with_labels",
                        lambda view, *, settings: {**view, "labels_unmapped": True})
    # Made on the Edit interest copy and saved (FUNCTIONAL_SPEC E).
    with _edit(app, slug) as copy:
        after = _choose(app, copy, "T10001", "filter")
        assert after["T10001"]["on"] is False and after["T10001"]["profile"] == "not_interest"
        assert _stored(env, copy)["not_interest_choices"]["T10001"] == "filter"
    stored = _stored(env, slug)
    assert stored["not_interest_choices"]["T10001"] == "filter"
    by = {t["id"]: t for t in stored["topics"]}
    assert by["T10001"]["on"] is False and by["T10001"]["profile"] == "not_interest"


def test_a_list_applied_before_choices_filters_a_not_wanted_topic_a_seed_brings_later(env):
    """Such a list switched every not-wanted topic off, whichever source
    listed it. One that turns up afterwards (a new seed carries it) read as
    unrecorded and got the default, kept on."""
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    _apply_filtering(app, slug)
    stored = _stored(env, slug)
    del stored["not_interest_choices"]
    _write_stored(env, slug, stored)
    _add_seed_topics(env, slug, [("T11986", "Scientific Computing")])
    assert _rows(app, slug)["https://openalex.org/T11986"]["choice"] == "filter"
    topics = _topics(app, slug)
    assert topics["https://openalex.org/T11986"]["on"] is False
    assert topics["https://openalex.org/T11986"]["profile"] == "not_interest"


def test_forgetting_the_researcher_forgets_the_choices(env):
    """Removed before the forget, or applied on a draft: the choices stayed
    and came back with the next researcher's profile."""
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    rid = _import_prosopia(app)
    removed = _draft_from(app, rid, name="Removed first")
    _apply_filtering(app, removed)
    assert _request(app, "PUT", f"/api/profiles/{removed}/researcher-topics", json={"enabled": False}).status_code == 200
    applied = _draft_from(app, rid, name="Still applied")
    _apply_filtering(app, applied)
    assert "not_interest_choices" in _stored(env, removed) and "not_interest_choices" in _stored(env, applied)
    assert _request(app, "DELETE", f"/api/researchers/{rid}").status_code == 200
    for slug in (removed, applied):
        assert "not_interest_choices" not in _stored(env, slug)
        _topics(app, slug)                                   # and a rebuild does not bring them back
        assert "not_interest_choices" not in _stored(env, slug)


def test_a_choice_on_a_saved_interest_leaves_marks_a_re_import_changed(env):
    """Setting one choice re-applied the whole profile to the stored list:
    a topic the re-imported profile now rejects was switched off too.

    Since FUNCTIONAL_SPEC E.7 the interest reads its own copy of the
    profile, so the re-import does not reach it at all: the choice (made on
    the Edit interest copy, spec E) moves that one topic and the profile
    the interest applied still wants T10515."""
    profile = _profile_with_document()
    app = _app(StubProsopia(profile), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    applied = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).json()
    _commit(app, slug, [t["id"] for t in applied if t["on"]])
    t10515 = "https://openalex.org/T10515"
    before = {t["id"]: t for t in _stored(env, slug)["topics"]}
    assert before[t10515]["on"] is True and before[t10515]["profile"] == "interest"
    doc = json.loads(json.dumps(DOCUMENT))
    for entry in doc["research_interests"]:
        if entry["concept"].get("code") == "T10515":
            entry["weight"] = -0.2                           # now rejected
    profile.metadata = doc
    assert _request(app, "POST", "/api/researchers/import", json={"source": "prosopia", "ref": "sheffield-nathan"}).status_code == 200
    with _edit(app, slug) as copy:
        after = _choose(app, copy, "T10001", "filter")
        assert after["T10001"]["on"] is False
        assert after[t10515]["on"] is True and after[t10515]["profile"] == "interest"
        others = {i: (t["on"], t.get("profile")) for i, t in after.items() if i != "T10001"}
        assert others == {i: (t["on"], t.get("profile")) for i, t in before.items() if i != "T10001"}
        view = _request(app, "GET", f"/api/profiles/{copy}/researcher-topics").json()
        assert t10515 in {t["id"] for t in view["topics"]["wanted"]}
    saved = {t["id"]: t for t in _stored(env, slug)["topics"]}
    assert saved["T10001"]["on"] is False and saved[t10515]["on"] is True and saved[t10515]["profile"] == "interest"


def test_a_saved_interest_shows_the_default_a_save_records(env):
    """A topic the profile newly rejects, which a new seed also lists: the
    view counted its default against the stored list (not there: filter),
    the save against the rebuilt one (a seed lists it: keep).

    Since FUNCTIONAL_SPEC E.7 a re-import does not reach the saved
    interest (it reads its copy of the profile): the new rejection is not
    listed until the user takes the profile again, on the Edit interest
    copy (remove, then apply), which reads the researcher as now."""
    profile = _profile_with_document()
    app = _app(StubProsopia(profile), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    applied = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).json()
    on = [t["id"] for t in applied if t["on"]]
    _commit(app, slug, on)
    profile.metadata = {**DOCUMENT, "research_interests": [*DOCUMENT["research_interests"],
                                                           _topic("T12345", "Something New", weight=-0.4, method="declared")]}
    assert _request(app, "POST", "/api/researchers/import", json={"source": "prosopia", "ref": "sheffield-nathan"}).status_code == 200
    _add_seed_topics(env, slug, [("T12345", "Something New")])
    assert "https://openalex.org/T12345" not in _rows(app, slug)
    with _edit(app, slug) as copy:
        assert _request(app, "PUT", f"/api/profiles/{copy}/researcher-topics", json={"enabled": False}).status_code == 200
        # Not applied: the view reads the researcher as stored now.
        shown = _rows(app, copy)["https://openalex.org/T12345"]
        assert shown["overlap"] == ["openalex"] and shown["choice"] == "keep"
        assert _request(app, "PUT", f"/api/profiles/{copy}/researcher-topics", json={"enabled": True}).status_code == 200
        assert _stored(env, copy)["not_interest_choices"]["T12345"] == "keep"
        assert _request(app, "PUT", f"/api/profiles/{copy}/topics", json={"selected_topic_ids": on}).status_code == 200
        assert _stored(env, copy)["not_interest_choices"]["T12345"] == "keep"
    assert _stored(env, slug)["not_interest_choices"]["T12345"] == "keep"


def test_an_apply_refused_for_leaving_nothing_on_stays_unapplied(env):
    """A recorded "filter" on the seeds' only topic: applying would switch
    everything off. Refused, and nothing written -- the flag stays off."""
    profile = _stub_profile()
    profile.metadata = {"research_interests": [_topic("T10001", "Test Topic", weight=-0.5, method="declared")]}
    profile.rid = "0000-0001-5643-4068"
    app = _app(StubProsopia(profile), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    _topics(app, slug)
    stored = _stored(env, slug)
    _write_stored(env, slug, {**stored, "not_interest_choices": {"T10001": "filter"}})
    before = _stored(env, slug)
    resp = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True})
    assert resp.status_code == 409 and "every topic off" in resp.json()["detail"]
    assert _stored(env, slug) == before and before.get("researcher_topics") is not True
    view = _request(app, "GET", f"/api/profiles/{slug}/researcher-topics").json()
    assert view["applied"] is False


@pytest.mark.parametrize("saved", [False, True])
def test_applying_twice_changes_nothing_the_second_time(env, saved):
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    first = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).json()
    _choose(app, slug, "https://openalex.org/T11986", "add")
    if saved:
        _commit(app, slug, [t["id"] for t in first if t["on"]] + ["https://openalex.org/T11986"])
    def states(key=slug):
        stored = _stored(env, key)
        return ({t["id"]: (t["on"], t.get("profile"), t.get("source")) for t in stored["topics"]},
                stored["not_interest_choices"], stored["researcher_topics"])

    once = states()
    if not saved:
        assert _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).status_code == 200
        assert states() == once
        return
    # A saved interest: the second apply on its Edit interest copy (spec E).
    with _edit(app, slug) as copy:
        assert states(copy) == once
        assert _request(app, "PUT", f"/api/profiles/{copy}/researcher-topics", json={"enabled": True}).status_code == 200
        assert states(copy) == once
    assert states() == once


def test_the_view_lists_the_not_wanted_labels_that_map_to_nothing(env):
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    topics = _request(app, "GET", f"/api/profiles/{slug}/researcher-topics").json()["topics"]
    # UMLS is off in the suite: there is nothing to map against.
    assert topics["unmatched_labels"] == ["phylogenomics", "wet-lab-protocol-development"]


def test_applying_on_a_saved_interest_records_the_default_shown(env):
    """Before the apply the view counts a default against the rebuilt list;
    the apply on a saved interest counted it against the stored one, which
    a topic a new seed brings is not in yet."""
    profile = _profile_with_document()
    profile.metadata = {**DOCUMENT, "research_interests": [*DOCUMENT["research_interests"],
                                                           _topic("T12345", "Something New", weight=-0.4, method="declared")]}
    app = _app(StubProsopia(profile), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    _commit(app, slug, list(_topics(app, slug)))
    _add_seed_topics(env, slug, [("T12345", "Something New")])
    assert _rows(app, slug)["https://openalex.org/T12345"]["choice"] == "keep"
    # Applied on the Edit interest copy and saved (FUNCTIONAL_SPEC E).
    with _edit(app, slug) as copy:
        assert _rows(app, copy)["https://openalex.org/T12345"]["choice"] == "keep"
        assert _request(app, "PUT", f"/api/profiles/{copy}/researcher-topics", json={"enabled": True}).status_code == 200
        assert _stored(env, copy)["not_interest_choices"]["T12345"] == "keep"
    assert _stored(env, slug)["not_interest_choices"]["T12345"] == "keep"


def test_choices_are_not_carried_once_the_researcher_is_gone(env):
    """The forget clears them; should that write not have happened (the
    researcher deleted underneath), a rebuild does not carry them either."""
    from rag_lib.db.repos import researchers as repo
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    rid = _import_prosopia(app)
    slug = _draft_from(app, rid)
    _apply_filtering(app, slug)
    assert _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": False}).status_code == 200
    assert "not_interest_choices" in _stored(env, slug)
    conn = connect(env)
    try:
        repo.delete(conn, rid)
    finally:
        conn.close()
    _topics(app, slug)
    assert "not_interest_choices" not in _stored(env, slug)


# --- not-wanted choices: after the second review -----------------------------------


@pytest.mark.parametrize("first,then,expect", [
    ("filter", "keep", (True, None)),           # back on, unmarked
    ("add", "filter", None),                    # only the profile had brought it: gone
])
def test_a_choice_on_a_draft_is_made_while_the_mapper_is_down(env, monkeypatch, first, then, expect):
    """The draft's choice went through a rebuild, which keeps the stored
    marks while the labels cannot be mapped: recorded, not applied."""
    from rag_lib.api.services import wizard
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    assert _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).status_code == 200
    tid = "T10001" if first == "filter" else "https://openalex.org/T11986"
    chosen = _choose(app, slug, tid, first)
    assert (chosen[tid]["on"], chosen[tid]["profile"]) == ((False, "not_interest") if first == "filter" else (True, "interest"))
    monkeypatch.setattr(wizard.researcher_topics, "with_labels",
                        lambda view, *, settings: {**view, "labels_unmapped": True})
    after = _choose(app, slug, tid, then)
    stored = {t["id"]: t for t in _stored(env, slug)["topics"]}
    if expect is None:
        assert tid not in after and tid not in stored
    else:
        assert (after[tid]["on"], after[tid]["profile"]) == expect
        assert stored[tid]["on"] is True and "profile" not in stored[tid]


def test_a_commit_does_not_write_over_a_choice_made_while_it_fitted(env, monkeypatch):
    """The commit pruned the list it read and wrote it after the fit,
    seconds later: a choice from another tab in between was lost."""
    from rag_lib.api.services import wizard
    from rag_lib.db.repos import profiles as profiles_repo
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    applied = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).json()
    real = profiles_repo.commit_draft
    calls = []

    def racing(conn, profile_id, **kw):
        calls.append(1)
        if len(calls) == 1:
            other = connect(env)
            try:
                wizard.set_not_interest_choice(other, user_id=1, slug=slug, topic_id="https://openalex.org/T11986", choice="add")
                wizard.set_not_interest_choice(other, user_id=1, slug=slug, topic_id="T10001", choice="filter")
            finally:
                other.close()
        return real(conn, profile_id, **kw)

    monkeypatch.setattr(profiles_repo, "commit_draft", racing)
    _commit(app, slug, [t["id"] for t in applied if t["on"]])
    stored = _stored(env, slug)
    assert len(calls) == 2
    assert stored["not_interest_choices"]["T11986"] == "add"
    # And its effect: the commit's ticks were for the list before the choice.
    by = {t["id"]: t for t in stored["topics"]}
    assert by["https://openalex.org/T11986"]["on"] is True and by["https://openalex.org/T11986"]["profile"] == "interest"
    assert by["T10001"]["on"] is False and by["T10001"]["profile"] == "not_interest"


def test_an_apply_retried_after_a_commit_works_on_the_saved_list(env, monkeypatch):
    """The row was read once, before the retries: a draft committed in
    between was rebuilt from its seeds as a draft, over the saved list."""
    from rag_lib.api.services import wizard
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    _topics(app, slug)
    real_rebuilt, real_stored = wizard._researcher_topics_rebuilt, wizard._researcher_topics_on_stored
    used = []

    def rebuilt(conn, profile_id, **kw):
        used.append("draft")
        out = real_rebuilt(conn, profile_id, **kw)
        other = connect(env)
        try:                                     # the commit lands in another tab
            other.execute("UPDATE profiles SET is_draft = 0 WHERE id = ?", (profile_id,))
            other.commit()
        finally:
            other.close()
        return out

    def on_stored(*a, **kw):
        used.append("saved")
        return real_stored(*a, **kw)

    monkeypatch.setattr(wizard, "_researcher_topics_rebuilt", rebuilt)
    monkeypatch.setattr(wizard, "_researcher_topics_on_stored", on_stored)
    assert _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).status_code == 200
    assert used == ["draft", "saved"]


def test_the_view_survives_a_rebuild_that_fails(env, monkeypatch):
    from rag_lib.api.services import wizard
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    _topics(app, slug)

    def boom(*a, **k):
        raise RuntimeError("umls index gone")

    monkeypatch.setattr(wizard, "_aggregate_topics", boom)
    rows = _rows(app, slug)
    assert rows["https://openalex.org/T10001"]["overlap"] == ["openalex"]   # from the stored list


def test_a_choice_records_only_itself(env):
    """A choice also wrote the defaults of the other unrecorded not-wanted
    topics, without applying them: one then read "filter" while still on.

    The unrecorded one came from a re-import; since FUNCTIONAL_SPEC E.7 a
    re-import does not reach the interest's copy of the profile, so it is
    in the profile from the start and its record is taken out by hand (a
    list from before choices were recorded, as above). The choice is made
    on the Edit interest copy (spec E)."""
    profile = _profile_with_document()
    profile.metadata = {**DOCUMENT, "research_interests": [*DOCUMENT["research_interests"],
                                                           _topic("T12345", "Something New", weight=-0.4, method="declared")]}
    app = _app(StubProsopia(profile), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    applied = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).json()
    _commit(app, slug, [t["id"] for t in applied if t["on"]])
    stored = _stored(env, slug)
    del stored["not_interest_choices"]["T12345"]
    _write_stored(env, slug, stored)
    recorded = dict(_stored(env, slug)["not_interest_choices"])
    assert "T12345" not in recorded
    with _edit(app, slug) as copy:
        _choose(app, copy, "T10001", "filter")
        assert _stored(env, copy)["not_interest_choices"] == {**recorded, "T10001": "filter"}
        assert _rows(app, copy)["https://openalex.org/T10001"]["marked"] == "not_interest"
    assert _stored(env, slug)["not_interest_choices"] == {**recorded, "T10001": "filter"}
    assert _rows(app, slug)["https://openalex.org/T10001"]["marked"] == "not_interest"



# --- not-wanted choices: after the third review ------------------------------------


def test_keep_on_a_saved_interest_keeps_a_topic_a_seed_has_brought_since(env):
    """Added (only the profile had it), saved; a seed carries it since. Keep
    read the stored ``source: researcher`` and took it out of the list --
    scans stopped searching it, and the next save brought it back off."""
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    applied = _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).json()
    t11986 = "https://openalex.org/T11986"
    _choose(app, slug, t11986, "add")
    _commit(app, slug, [t["id"] for t in applied if t["on"]] + [t11986])
    _add_seed_topics(env, slug, [("T11986", "Scientific Computing")])
    assert _rows(app, slug)[t11986]["overlap"] == ["openalex"]
    # Chosen on the Edit interest copy and saved (FUNCTIONAL_SPEC E).
    with _edit(app, slug) as copy:
        kept = _choose(app, copy, t11986, "keep")
    # Still listed, as the seed's; back to its state before the add -- off,
    # as any topic a seed brings to a saved interest starts -- and the next
    # rebuild agrees rather than bringing it back as new.
    assert t11986 in kept and kept[t11986]["profile"] is None and kept[t11986]["source"] != "researcher"
    preview = {t["id"]: t for t in _request(app, "GET", f"/api/profiles/{slug}/topics/preview").json()}
    assert preview[t11986]["on"] is kept[t11986]["on"] is False
    assert preview[t11986].get("new") is not True


def test_filtering_a_topic_the_profile_also_wants_keeps_it_listed_off(env, monkeypatch):
    """A not-wanted label that maps onto a wanted topic only the profile
    brings: filtered, apply keeps it (marked, off); the choice took it out,
    and the next rebuild put it back."""
    from rag_lib.api.services import wizard
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    t10222 = "https://openalex.org/T10222"
    real = wizard.researcher_topics.with_labels

    def with_labels(view, *, settings):
        out = real(view, settings=settings)
        return {**out, "unwanted": [*out["unwanted"], {"id": t10222, "display_name": "Genomics and Chromatin Dynamics",
                                                        "weight": None, "method": "label", "label": "chromatin stuff",
                                                        "similarity": 0.8}]}

    monkeypatch.setattr(wizard.researcher_topics, "with_labels", with_labels)
    assert _request(app, "PUT", f"/api/profiles/{slug}/researcher-topics", json={"enabled": True}).status_code == 200
    chosen = _choose(app, slug, t10222, "filter")
    assert chosen[t10222]["on"] is False and chosen[t10222]["profile"] == "not_interest"
    again = _topics(app, slug)
    assert again[t10222]["on"] is False and again[t10222]["profile"] == "not_interest"


def test_a_step_3_read_overtaken_by_the_commit_does_not_rebuild_the_saved_list(env, monkeypatch):
    """The draft's rebuild was redone on the committed interest and written
    over it: a topic the seeds had gained came in on, unseen."""
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
            other = connect(env)
            try:
                stored = json.loads(other.execute("SELECT topic_filters_json FROM profiles WHERE id = ?", (profile_id,)).fetchone()[0])
                stored["marker"] = "committed"
                other.execute("UPDATE profiles SET is_draft = 0, topic_filters_json = ? WHERE id = ?", (json.dumps(stored), profile_id))
                other.commit()
            finally:
                other.close()
        return out

    monkeypatch.setattr(wizard, "_aggregate_topics", racing)
    resp = _request(app, "GET", f"/api/profiles/draft/{slug}/topics")
    assert resp.status_code == 409
    assert _stored(env, slug)["marker"] == "committed"


def test_forgetting_after_a_commit_treats_the_interest_as_saved(env, monkeypatch):
    """Draft or saved was read before the delete, outside the lock: a commit
    in between had the saved list stripped as a draft's. Saved by then, the
    interest keeps the topics it was saved with, exactly (E.7: an interest
    does not follow its researcher; it used to have them "adopted")."""
    from rag_lib.api.services import researchers as researchers_service
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    rid = _import_prosopia(app)
    slug = _draft_from(app, rid)
    applied = _apply_filtering(app, slug)
    only_profile = [i for i, t in applied.items() if t["on"] and t.get("source") == "researcher"]
    real = researchers_service.researchers_repo.delete

    def committing(conn, researcher_id):
        conn.execute("UPDATE profiles SET is_draft = 0 WHERE slug = ?", (slug,))
        conn.commit()
        return real(conn, researcher_id)

    monkeypatch.setattr(researchers_service.researchers_repo, "delete", committing)
    before = _stored(env, slug)
    assert _request(app, "DELETE", f"/api/researchers/{rid}").status_code == 200
    assert _stored(env, slug) == before
    stored = {t["id"]: t for t in before["topics"]}
    assert only_profile and all(stored[i]["source"] == "researcher" and stored[i]["on"] for i in only_profile)


def test_the_view_leaves_step_3s_umls_notice_as_it_was(env, monkeypatch):
    from rag_lib.api.services import wizard
    app = _app(StubProsopia(_profile_with_document()), _stub_openalex())
    slug = _draft_from(app, _import_prosopia(app))
    _topics(app, slug)
    conn = connect(env)
    try:
        pid = conn.execute("SELECT id FROM profiles WHERE slug = ?", (slug,)).fetchone()[0]
    finally:
        conn.close()
    wizard._UMLS_MERGE_STATE[pid] = {"available": False, "reason": "busy", "n": 0}
    _rows(app, slug)
    assert wizard._UMLS_MERGE_STATE[pid] == {"available": False, "reason": "busy", "n": 0}


def test_keep_takes_the_source_another_source_now_lists_it_under():
    from rag_lib.api.services import wizard
    tid = "https://openalex.org/T11986"
    stored = {"researcher_topics": True, "topics": [
        {"id": "T10001", "display_name": "Test Topic", "on": True},
        {"id": tid, "display_name": "Scientific Computing", "count": 0, "source": rt.SOURCE,
         "profile": rt.INTEREST, "on": True, rt.BEFORE: True}]}
    view = {"wanted": [], "unwanted": [{"id": tid, "display_name": "Scientific Computing"}]}
    basis = {"topics": [{"id": tid, "display_name": "Scientific Computing", "count": 2, "source": "umls"}]}
    new = rt.chosen_topic(stored, view, "T11986", rt.KEEP, basis=basis)
    out = {t["id"]: t for t in wizard._with_one_topic_from(stored, new, "T11986", {"T11986": rt.KEEP})["topics"]}
    assert out[tid]["source"] == "umls"
    assert out[tid]["on"] is True and "profile" not in out[tid] and rt.BEFORE not in out[tid]
    # No other source: only the profile had it, so it goes.
    assert rt.chosen_topic(stored, view, "T11986", rt.KEEP, basis={"topics": []}) is None
