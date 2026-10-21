"""Review round 12 (reviewer a): an adopted topic is copied from the stored list whole,
so a remembered pre-apply state (``on_before_profile``) outlives the mark
it belonged to and is "restored" over the user's later choice."""

from __future__ import annotations

import json

from rag_lib import researcher_topics as rt
from rag_lib.api.services import wizard
from rag_lib.db.repos import profiles as profiles_repo, researchers as researchers_repo
from tests.test_researcher_topics import _topic
from tests.test_review_round10 import _live_interest
from tests.test_review_round4 import _db

T1, T2, T3 = ("https://openalex.org/T10001", "https://openalex.org/T10002", "https://openalex.org/T10003")


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("RADAR_UMLS_ENABLED", "false")
    conn, uid = _db(tmp_path)
    pid, slug = _live_interest(conn, uid)
    # The researcher's profile, as re-read now: it says nothing about T1 any more.
    rid = researchers_repo.upsert(conn, user_id=uid, source="prosopia", key="p", name="P",
                                  document_json=json.dumps({"research_interests": [_topic("T10003", "Three", weight=0.5, method="declared")]}))
    researchers_repo.set_profile_researcher(conn, pid, rid)
    # What the last aggregation stored: T1 and T2 were adopted from an earlier
    # researcher; this one's profile then marked T1 not-wanted (off, with the
    # state before that remembered).
    stored = {"topics": [
        {"id": T1, "display_name": "One", "count": 0, "on": False, "source": rt.ADOPTED,
         "profile": rt.NOT_INTEREST, rt.BEFORE: True},
        {"id": T2, "display_name": "Two", "count": 0, "on": True, "source": rt.ADOPTED},
        {"id": T3, "display_name": "Three", "count": 0, "on": True, "source": rt.SOURCE, "profile": rt.INTEREST, rt.BEFORE: True},
    ], rt.FLAG: True}
    conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE id = ?", (json.dumps(stored), pid))
    conn.commit()
    return conn, uid, pid, slug


def _by_id(filters):
    return {t["id"]: t for t in filters["topics"]}


def test_a_vanished_mark_leaves_no_remembered_state_on_an_adopted_topic(tmp_path, monkeypatch):
    conn, uid, pid, slug = _setup(tmp_path, monkeypatch)
    out, _new = wizard.preview_topics(conn, user_id=uid, slug=slug)
    t1 = _by_id(out)[T1]
    assert "profile" not in t1                    # the profile has no opinion on T1 now
    assert t1["on"] is True                        # ...so it went back to what it was before
    # The seeds' topics and the UMLS copies drop the remembered state with the
    # mark (``_carry_before_states`` keeps marked topics' only); the adopted
    # copy keeps it although nothing marks it any more.
    assert rt.BEFORE not in t1, t1
    conn.close()


def test_the_stale_state_is_restored_over_the_users_later_choice(tmp_path, monkeypatch):
    conn, uid, pid, slug = _setup(tmp_path, monkeypatch)
    # The user switches T1 off (keeping T2 and the profile's T3)...
    saved = wizard.save_topics(conn, user_id=uid, slug=slug, selected_topic_ids=[T2, T3])
    assert _by_id(saved)[T1]["on"] is False
    # ...and later takes the profile's topics off. T1 was not marked by the
    # profile: removing it must not touch T1.
    after = wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=False)
    assert T3 not in _by_id(after)
    assert _by_id(after)[T1]["on"] is False, _by_id(after)[T1]
    assert _by_id(profiles_repo.topic_filters(conn, pid))[T1]["on"] is False
    conn.close()
