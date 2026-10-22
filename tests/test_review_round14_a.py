"""Review 5 (reviewer a): the life of an interest's topic list."""

from __future__ import annotations

import json

from rag_lib import researcher_topics as rt
from rag_lib.api.services import wizard
from rag_lib.db.repos import papers as papers_repo, profiles as profiles_repo, researchers as researchers_repo
from tests.test_researcher_topics import _topic
from tests.test_review_round10 import _live_interest
from tests.test_review_round4 import _db

T1, T2, T3 = ("https://openalex.org/T10001", "https://openalex.org/T10002", "https://openalex.org/T10003")


def _saved(tmp_path, monkeypatch, doc):
    monkeypatch.setenv("RADAR_UMLS_ENABLED", "false")
    conn, uid = _db(tmp_path)
    pid, slug = _live_interest(conn, uid)
    papers_repo.upsert(conn, {"openalex_id": "https://openalex.org/W1", "title": "Seed", "source": "openalex",
                              "primary_topic": {"id": T1, "display_name": "One"},
                              "topics": [{"id": T1, "display_name": "One"}, {"id": T2, "display_name": "Two"}]})
    profiles_repo.attach_seed(conn, pid, "https://openalex.org/W1")
    rid = researchers_repo.upsert(conn, user_id=uid, source="prosopia", key="p", name="P", document_json=json.dumps(doc))
    researchers_repo.set_profile_researcher(conn, pid, rid)
    stored = {"topics": [{"id": T1, "display_name": "One", "count": 1, "on": True},
                         {"id": T2, "display_name": "Two", "count": 1, "on": True}]}
    conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE id = ?", (json.dumps(stored), pid))
    conn.commit()
    return conn, uid, pid, slug, rid


def _states(f):
    return {t["id"]: bool(t.get("on", True)) for t in f["topics"]}


def test_remove_overrides_a_switch_the_user_made_after_the_apply(tmp_path, monkeypatch):
    doc = {"research_interests": [_topic("T10001", "One", weight=0.5, method="declared"),
                                  _topic("T10003", "Three", weight=0.5, method="declared")]}
    conn, uid, pid, slug, _ = _saved(tmp_path, monkeypatch, doc)
    applied = wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=True)
    assert _states(applied) == {T1: True, T2: True, T3: True}
    # Edit page: the user switches T1 off and saves.
    saved = wizard.save_topics(conn, user_id=uid, slug=slug, selected_topic_ids=[T2, T3])
    assert _states(saved) == {T1: False, T2: True, T3: True}
    removed = wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=False)
    # The user's own switch (made after the profile had its say) should survive.
    assert _states(removed)[T1] is False, _states(removed)
    conn.close()


def test_a_topic_the_profile_newly_rejects_stays_on(tmp_path, monkeypatch):
    doc = {"research_interests": [_topic("T10003", "Three", weight=0.5, method="declared")]}
    conn, uid, pid, slug, rid = _saved(tmp_path, monkeypatch, doc)
    wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=True)
    # The researcher is re-imported: the owner now says no to T2.
    doc2 = {"research_interests": doc["research_interests"] + [_topic("T10002", "Two", weight=-0.5, method="declared")]}
    researchers_repo.upsert(conn, user_id=uid, source="prosopia", key="p", name="P", document_json=json.dumps(doc2))
    view = wizard.researcher_topics_view(conn, user_id=uid, slug=slug)
    assert view["applied"] and [t["id"] for t in view["topics"]["unwanted"]] == [T2]
    preview, _ = wizard.preview_topics(conn, user_id=uid, slug=slug)
    by = {t["id"]: t for t in preview["topics"]}
    assert by[T2]["profile"] == rt.NOT_INTEREST
    # The box says "the topics it does not want stay off wherever the seeds bring them in".
    assert by[T2]["on"] is False, by[T2]
    conn.close()


def test_a_profile_only_topic_a_new_seed_also_carries_stays_on_after_remove(tmp_path, monkeypatch):
    doc = {"research_interests": [_topic("T10003", "Three", weight=0.5, method="declared")]}
    conn, uid, pid, slug, _ = _saved(tmp_path, monkeypatch, doc)
    wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=True)
    papers_repo.upsert(conn, {"openalex_id": "https://openalex.org/W2", "title": "Seed 2", "source": "openalex",
                              "primary_topic": {"id": T3, "display_name": "Three"}, "topics": [{"id": T3, "display_name": "Three"}]})
    profiles_repo.attach_seed(conn, pid, "https://openalex.org/W2")
    preview, new = wizard.preview_topics(conn, user_id=uid, slug=slug)
    wizard.save_topics(conn, user_id=uid, slug=slug, selected_topic_ids=[i for i, on in _states(preview).items() if on])
    removed = wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=False)
    print(_states(removed), {t["id"]: t.get("source") for t in removed["topics"]})
    # Without the profile, T3 is a topic the seeds gained since the save: on a saved interest it starts off.
    fresh, _ = wizard.preview_topics(conn, user_id=uid, slug=slug)
    assert _states(removed)[T3] is False, _states(removed)
    conn.close()


def test_an_apply_does_not_overwrite_a_topics_save_that_landed_meanwhile(tmp_path, monkeypatch):
    """Round 14 (reviewer a, #4): the stored list is read, the profile's
    labels are mapped (seconds, the first time), then the list is written.
    A topics save from another tab in between was overwritten."""
    import json as _json
    from rag_lib.api.services import wizard as w
    conn, uid, pid, slug, _rid = _saved(tmp_path, monkeypatch, {"research_interests": [_topic("T10003", "Three", weight=0.5, method="declared")]})
    real = w._researcher_topics_on_stored
    calls = {"n": 0}

    def racing(conn_, profile_id, stored, *, enabled):
        calls["n"] += 1
        if calls["n"] == 1:   # another tab saves while this one maps labels
            other = _json.loads(conn_.execute("SELECT topic_filters_json FROM profiles WHERE id = ?", (profile_id,)).fetchone()[0])
            other["marker"] = "saved elsewhere"
            conn_.execute("UPDATE profiles SET topic_filters_json = ? WHERE id = ?", (_json.dumps(other), profile_id))
            conn_.commit()
        return real(conn_, profile_id, stored, enabled=enabled)

    monkeypatch.setattr(w, "_researcher_topics_on_stored", racing)
    w.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=True)
    final = _json.loads(conn.execute("SELECT topic_filters_json FROM profiles WHERE id = ?", (pid,)).fetchone()[0])
    assert calls["n"] == 2                     # read again and redone on top of the other save
    assert final.get("marker") == "saved elsewhere" and final.get("researcher_topics") is True
    conn.close()
