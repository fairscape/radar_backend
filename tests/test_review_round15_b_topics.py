"""Review 6 (b): probes of the profile-topics lifecycle."""

from __future__ import annotations

import json

from rag_lib import researcher_topics as rt
from rag_lib.api.services import wizard
from rag_lib.db.repos import papers as papers_repo, profiles as profiles_repo, researchers as researchers_repo
from tests.test_researcher_topics import _topic
from tests.test_review_round4 import _db

T1, T2, T3 = ("https://openalex.org/T10001", "https://openalex.org/T10002", "https://openalex.org/T10003")


def _setup(tmp_path, monkeypatch, doc, *, draft):
    monkeypatch.setenv("RADAR_UMLS_ENABLED", "false")
    monkeypatch.setenv("RADAR_CONTENT_TOPICS_ENABLED", "false")
    from rag_lib.api import settings as s
    s.get_settings.cache_clear()
    conn, uid = _db(tmp_path)
    pid = profiles_repo.upsert(conn, user_id=uid, name="Live", embedding_model="placeholder-v1",
                               n_seed=1, topic_filters={"topics": []}, threshold=0.5)
    conn.execute("UPDATE profiles SET is_draft = ? WHERE id = ?", (1 if draft else 0, pid))
    conn.commit()
    slug = profiles_repo.get(conn, pid)["slug"]
    papers_repo.upsert(conn, {"openalex_id": "https://openalex.org/W1", "title": "Seed", "source": "openalex",
                              "primary_topic": {"id": T1, "display_name": "One"},
                              "topics": [{"id": T1, "display_name": "One"}, {"id": T2, "display_name": "Two"}]})
    profiles_repo.attach_seed(conn, pid, "https://openalex.org/W1")
    rid = researchers_repo.upsert(conn, user_id=uid, source="prosopia", key="p", name="P", document_json=json.dumps(doc))
    researchers_repo.set_profile_researcher(conn, pid, rid)
    return conn, uid, pid, slug, rid


def _states(f):
    return {t["id"]: bool(t.get("on", True)) for t in f["topics"]}


def _stored(conn, pid):
    return json.loads(conn.execute("SELECT topic_filters_json FROM profiles WHERE id = ?", (pid,)).fetchone()[0])


DOC = {"research_interests": [_topic("T10001", "One", weight=0.5, method="declared"),
                              _topic("T10002", "Two", weight=-0.5, method="declared"),
                              _topic("T10003", "Three", weight=0.5, method="declared")]}


def test_draft_on_off_on_restores_manual_choices(tmp_path, monkeypatch):
    conn, uid, pid, slug, _ = _setup(tmp_path, monkeypatch, DOC, draft=True)
    wizard.aggregate_draft_topics(conn, user_id=uid, slug=slug)
    wizard.save_topics(conn, user_id=uid, slug=slug, selected_topic_ids=[T2])   # user: T1 off, T2 on
    wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=True)
    # T2 is a seed topic as well: kept until the user filters it out.
    a = wizard.set_not_interest_choice(conn, user_id=uid, slug=slug, topic_id=T2, choice="filter")
    assert _states(a) == {T1: True, T2: False, T3: True}
    r = wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=False)
    assert _states(r) == {T1: False, T2: True}, _states(r)
    a2 = wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=True)
    assert _states(a2) == {T1: True, T2: False, T3: True}
    # step 3 re-read in between
    wizard.aggregate_draft_topics(conn, user_id=uid, slug=slug)
    r2 = wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=False)
    assert _states(r2) == {T1: False, T2: True}, _states(r2)


def test_live_on_off_on_restores_manual_choices(tmp_path, monkeypatch):
    conn, uid, pid, slug, _ = _setup(tmp_path, monkeypatch, DOC, draft=False)
    stored = {"topics": [{"id": T1, "display_name": "One", "count": 1, "on": False},
                         {"id": T2, "display_name": "Two", "count": 1, "on": True}]}
    conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE id = ?", (json.dumps(stored), pid))
    conn.commit()
    wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=True)
    a = wizard.set_not_interest_choice(conn, user_id=uid, slug=slug, topic_id=T2, choice="filter")
    assert _states(a) == {T1: True, T2: False, T3: True}
    # edit page: preview then save without changes in between (user saves another tick)
    prev, _ = wizard.preview_topics(conn, user_id=uid, slug=slug)
    assert _states(prev) == {T1: True, T2: False, T3: True}, _states(prev)
    wizard.save_topics(conn, user_id=uid, slug=slug, selected_topic_ids=[T1, T3])
    r = wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=False)
    assert _states(r) == {T1: False, T2: True}, _states(r)
    a2 = wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=True)
    assert _states(a2) == {T1: True, T2: False, T3: True}
    r2 = wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=False)
    assert _states(r2) == {T1: False, T2: True}, _states(r2)


def test_live_profile_only_topic_survives_edit_save(tmp_path, monkeypatch):
    conn, uid, pid, slug, _ = _setup(tmp_path, monkeypatch, DOC, draft=False)
    stored = {"topics": [{"id": T1, "display_name": "One", "count": 1, "on": True},
                         {"id": T2, "display_name": "Two", "count": 1, "on": True}]}
    conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE id = ?", (json.dumps(stored), pid))
    conn.commit()
    wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=True)
    # user switches the profile-only T3 off on the edit page
    wizard.save_topics(conn, user_id=uid, slug=slug, selected_topic_ids=[T1])
    prev, new = wizard.preview_topics(conn, user_id=uid, slug=slug)
    assert _states(prev)[T3] is False, _states(prev)
    assert not new


def test_prosopia_terms_reads_the_same_fields_researcher_topics_reads(monkeypatch):
    """interest_terms.prosopia_terms and researcher_topics.read both read a
    Prosopia profile's ``fields``; fed the document researcher_topics is
    tested against, the terms path should take its wanted interests and
    leave the not-wanted ones out."""
    from rag_lib import interest_terms
    from tests.test_researcher_topics import DOCUMENT

    class Resp:
        status_code = 200

        def json(self):
            return {"fields": {**DOCUMENT, "expertise": [], "subfields": []}}

    monkeypatch.setattr(interest_terms.requests, "get", lambda url, timeout=0: Resp())
    found = interest_terms.prosopia_terms("https://prosopia.example", "x")
    labels = [t["label"] for t in (found or {}).get("terms") or []]
    assert "Genomics and Chromatin Dynamics" in labels, found
    assert "Test Topic" not in labels and "Scientific Computing" not in labels, found


def test_prosopia_terms_old_shape_still_read_and_not_interests_never_queried(monkeypatch):
    from rag_lib import interest_terms

    class Resp:
        status_code = 200

        def json(self):
            return {"fields": {"research_interests": {"top": [{"label": "Gene regulation"}, {"label": "Phylogenomics"}]},
                               "expertise": ["phylogenomics", "Statistics"], "not_interests": ["phylogenomics"]}}

    monkeypatch.setattr(interest_terms.requests, "get", lambda url, timeout=0: Resp())
    found = interest_terms.prosopia_terms("https://prosopia.example", "x")
    assert [t["label"] for t in found["terms"]] == ["Gene regulation", "Statistics"]
    assert found["not_interests"] == ["phylogenomics"]


def test_a_topic_list_that_keeps_changing_is_a_409_not_a_500(monkeypatch):
    from rag_lib.api.services import wizard

    monkeypatch.setattr(wizard, "_set_researcher_topics_once", lambda *a, **k: None)
    monkeypatch.setattr(wizard, "_require_profile", lambda conn, uid, slug: {"id": 1, "researcher_id": None})
    monkeypatch.setattr(wizard.gather_runs_repo, "has_unfinished_import", lambda conn, pid: False)

    class Conn:
        def execute(self, *a):
            class R:
                def fetchone(self):
                    return ["[]"]
            return R()

    import pytest
    with pytest.raises(wizard.TopicsChanging):
        wizard.set_researcher_topics(Conn(), user_id=1, slug="x", enabled=False)
