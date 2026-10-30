"""A saved interest after a seed is removed and its topics changed
(FUNCTIONAL_SPEC T).

Removing a seed takes it out of the seed list and re-ranks what was found;
the topics stay as saved. The edit page lists the saved topics, the ones no
seed brings any more marked ``unmatched``; saving keeps them, on or off.
Changing which topics are on is only said at the threshold step (the
recommendations came from the topics the last scan searched), not acted on.
"""

from __future__ import annotations

import json

import pytest

from rag_lib.api.services import wizard
from rag_lib.db import connect
from rag_lib.db.repos import gather_runs as gather_runs_repo, papers as papers_repo, profiles as profiles_repo
from rag_lib.scheduler import jobs as scheduler_jobs
from tests.test_review_round4 import _db
from tests.test_scheduler_jobs import (  # noqa: F401 -- fixtures
    _candidate_paper, _patch_dependencies, db_path, profile_id, settings,
)

T1, T2, T3, T4, T9 = (f"https://openalex.org/T1000{i}" for i in (1, 2, 3, 4, 9))
W1, W2, W3 = ("https://openalex.org/W1", "https://openalex.org/W2", "https://openalex.org/W3")


def _paper(conn, oa_id, *topics):
    papers_repo.upsert(conn, {"openalex_id": oa_id, "title": oa_id, "source": "openalex",
                              "primary_topic": {"id": topics[0][0], "display_name": topics[0][1]},
                              "topics": [{"id": t, "display_name": n} for t, n in topics]})


@pytest.fixture()
def live(tmp_path, monkeypatch):
    """A saved interest: W1 brings One and Two, W2 brings Three; all three
    saved on, and a Claude-generated topic (source llm) on too."""
    monkeypatch.setenv("RADAR_UMLS_ENABLED", "false")
    monkeypatch.setenv("RADAR_CONTENT_TOPICS_ENABLED", "false")
    from rag_lib.api import settings as s
    s.get_settings.cache_clear()
    conn, uid = _db(tmp_path)
    pid = profiles_repo.upsert(conn, user_id=uid, name="Live", embedding_model="placeholder-v1",
                               n_seed=2, topic_filters={"topics": []}, threshold=0.5)
    conn.execute("UPDATE profiles SET is_draft = 0 WHERE id = ?", (pid,))
    _paper(conn, W1, (T1, "One"), (T2, "Two"))
    _paper(conn, W2, (T3, "Three"))
    for w in (W1, W2):
        profiles_repo.attach_seed(conn, pid, w)
    slug = profiles_repo.get(conn, pid)["slug"]
    wizard.aggregate_draft_topics(conn, user_id=uid, slug=slug)
    stored = _stored(conn, pid)
    stored["topics"].append({"id": T9, "display_name": "Nine", "count": 0, "source": "llm",
                             "llm_label": "nine", "on": True})
    conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE id = ?", (json.dumps(stored), pid))
    conn.commit()
    return conn, uid, pid, slug


def _stored(conn, pid):
    return json.loads(conn.execute("SELECT topic_filters_json FROM profiles WHERE id = ?", (pid,)).fetchone()[0])


def _by_id(f):
    return {t["id"]: t for t in f["topics"]}


def _remove(conn, pid, oa_id):
    """What ``remove_seed`` does to the seeds (its refit needs vectors)."""
    assert profiles_repo.detach_seed(conn, pid, oa_id)


def test_removing_a_seed_leaves_the_saved_topics_as_they_are(live, monkeypatch):
    """T-1 as revised by spec E (T.5, E.4 Q5): a saved interest's seed is
    removed on its Edit interest copy. Saved, the seed is gone, the topic
    list is as it was (the one only it brought still on, E.2-3 as revised),
    nothing is refitted and no scan runs."""
    conn, uid, pid, slug = live
    before = _by_id(_stored(conn, pid))
    from rag_lib.db.repos import embeddings as embeddings_repo
    from tests.edit_helpers import edit_saved_db
    embeddings_repo.upsert(conn, W1, "placeholder-v1", [0.1] * 8)     # the seed left can be placed
    monkeypatch.setattr(wizard, "embed_missing", lambda *a, **k: None)
    refits = []
    monkeypatch.setattr(wizard, "refit_profile", lambda conn, **kw: refits.append(kw))
    with pytest.raises(wizard.SavedInterest):
        wizard.remove_seed(conn, user_id=uid, slug=slug, openalex_id=W2)
    with edit_saved_db(conn, uid, slug) as copy:
        assert wizard.remove_seed(conn, user_id=uid, slug=copy, openalex_id=W2)
    assert profiles_repo.list_seed_openalex_ids(conn, pid) == [W1]
    assert refits == []                                      # the next scan refits
    assert _by_id(_stored(conn, pid)) == before
    assert conn.execute("SELECT COUNT(*) FROM gather_runs WHERE profile_id = ?", (pid,)).fetchone()[0] == 0


def test_the_edit_list_keeps_a_removed_seeds_topic_as_saved_and_marks_it(live):
    """T-2, T-10."""
    conn, uid, pid, slug = live
    before = _stored(conn, pid)
    _remove(conn, pid, W2)
    prev, new = wizard.preview_topics(conn, user_id=uid, slug=slug)
    got = _by_id(prev)
    assert got[T3]["on"] is True and got[T3].get("unmatched") is True and got[T3]["papers"] == 0
    assert not got[T1].get("unmatched") and not got[T2].get("unmatched")
    assert not got[T9].get("unmatched")                       # not a seed's topic
    assert new == set()
    assert _stored(conn, pid) == before                       # a preview writes nothing


def test_saving_keeps_it_on(live):
    """T-3; the mark is not stored (worked out from the seeds each time)."""
    conn, uid, pid, slug = live
    _remove(conn, pid, W2)
    wizard.save_topics(conn, user_id=uid, slug=slug, selected_topic_ids=[T1, T2, T3, T9])
    stored = _by_id(_stored(conn, pid))
    assert stored[T3]["on"] is True
    assert all("unmatched" not in t for t in stored.values())


def test_switched_off_it_stays_listed_off_and_marked(live):
    """T-4."""
    conn, uid, pid, slug = live
    _remove(conn, pid, W2)
    wizard.save_topics(conn, user_id=uid, slug=slug, selected_topic_ids=[T1, T2, T9])
    assert _by_id(_stored(conn, pid))[T3]["on"] is False
    prev, _ = wizard.preview_topics(conn, user_id=uid, slug=slug)
    got = _by_id(prev)
    assert got[T3]["on"] is False and got[T3].get("unmatched") is True
    # and it is not searched
    from rag_lib.openalex_tiers import enabled_topic_ids
    assert T3 not in enabled_topic_ids(_stored(conn, pid))


def test_a_new_seeds_topic_is_new_and_off_not_unmatched(live):
    """T-5."""
    conn, uid, pid, slug = live
    _paper(conn, W3, (T4, "Four"))
    profiles_repo.attach_seed(conn, pid, W3)
    prev, new = wizard.preview_topics(conn, user_id=uid, slug=slug)
    got = _by_id(prev)
    assert new == {T4} and got[T4]["on"] is False and not got[T4].get("unmatched")


def test_the_seed_added_back_brings_it_back_matched(live):
    conn, uid, pid, slug = live
    _remove(conn, pid, W2)
    wizard.save_topics(conn, user_id=uid, slug=slug, selected_topic_ids=[T1, T2, T3])
    profiles_repo.attach_seed(conn, pid, W2)
    got = _by_id(wizard.preview_topics(conn, user_id=uid, slug=slug)[0])
    assert not got[T3].get("unmatched") and got[T3]["papers"] == 1 and got[T3]["on"] is True


def test_a_drafts_list_still_follows_its_seeds(live):
    conn, uid, pid, slug = live
    conn.execute("UPDATE profiles SET is_draft = 1 WHERE id = ?", (pid,))
    conn.commit()
    _remove(conn, pid, W2)
    f = wizard.aggregate_draft_topics(conn, user_id=uid, slug=slug)
    assert T3 not in _by_id(f)


def _scan(conn, pid, uid, topic_ids, *, error=None, tier="manual"):
    run = gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used=tier)
    gather_runs_repo.finish(conn, run, n_fetched=0, n_new=0, n_redup=0, error=error, topic_ids=topic_ids)


def test_changed_since_scan(live):
    """T-7 to T-9 at the service: what the threshold step is told of the
    topics (the seeds: tests/test_edit_copy.py)."""
    conn, uid, pid, slug = live
    changed = lambda: wizard.scan_outdated(conn, user_id=uid, slug=slug)["topics"]  # noqa: E731
    assert changed() is False                                 # no scan recorded its topics
    _scan(conn, pid, uid, [T1, T2, T3, T9])
    assert changed() is False
    _remove(conn, pid, W2)
    assert changed() is False                                 # a seed removed is not a topic change
    wizard.save_topics(conn, user_id=uid, slug=slug, selected_topic_ids=[T1, T2, T9])
    assert changed() is True                                  # T-7
    wizard.save_topics(conn, user_id=uid, slug=slug, selected_topic_ids=[T1, T2, T3, T9])
    assert changed() is False                                 # T-9: back as the scan searched
    wizard.save_topics(conn, user_id=uid, slug=slug, selected_topic_ids=[T1])
    _scan(conn, pid, uid, [T2], error="OpenAlex unavailable")  # a failed scan searched nothing
    _scan(conn, pid, uid, None, tier="dry_run")              # a trial records no topics
    assert changed() is True
    _scan(conn, pid, uid, [T1])
    assert changed() is False                                 # T-8


def test_a_scan_records_the_topics_it_searched(monkeypatch, db_path, settings, profile_id):
    conn = connect(db_path)
    conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE id = ?", (json.dumps({"topics": [
        {"id": "T1", "display_name": "A", "count": 1, "on": False},
        {"id": "T2", "display_name": "B", "count": 1, "on": True},
        {"id": "T3", "display_name": "C", "count": 1}]}), profile_id))
    conn.commit()
    _patch_dependencies(monkeypatch, [_candidate_paper("W100")])
    run = scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings, days=7)
    assert json.loads(gather_runs_repo.get(conn, run)["topic_ids_json"]) == ["T2", "T3"]
    assert gather_runs_repo.last_scan(conn, profile_id)[0] == ["T2", "T3"]
    # every topic off: a scan that searched nothing, and says so
    conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE id = ?", (json.dumps({"topics": [
        {"id": "T1", "display_name": "A", "count": 1, "on": False}]}), profile_id))
    conn.commit()
    run = scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings, days=7)
    assert gather_runs_repo.get(conn, run)["tier_used"] == "no-enabled-topics"
    assert gather_runs_repo.last_scan(conn, profile_id)[0] == []
    conn.close()


def test_a_failed_scan_records_no_topics(monkeypatch, db_path, settings, profile_id):
    class Boom:
        def __init__(self, **_):
            pass

        def fetch(self, *a, **k):
            raise RuntimeError("OpenAlex down")

        def cost(self):
            return {"api_calls": 0}

    monkeypatch.setattr(scheduler_jobs, "OpenAlexGatherer", Boom)
    run = scheduler_jobs.gather_for_profile(user_id=1, profile_id=profile_id, settings=settings, days=7)
    conn = connect(db_path)
    row = gather_runs_repo.get(conn, run)
    assert row["error"] and row["topic_ids_json"] is None
    assert gather_runs_repo.last_scan(conn, profile_id) is None
    conn.close()


def test_a_researchers_topic_is_not_kept_as_a_seeds(live):
    """A topic from a researcher's profile that is no longer applied goes,
    as before: only the seeds' own topics are kept as unmatched."""
    conn, uid, pid, slug = live
    stored = _stored(conn, pid)
    stored["topics"].append({"id": T4, "display_name": "Four", "count": 0, "source": "researcher", "on": True})
    conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE id = ?", (json.dumps(stored), pid))
    conn.commit()
    assert T4 not in _by_id(wizard.preview_topics(conn, user_id=uid, slug=slug)[0])


def test_a_topic_the_profile_does_not_want_stays_off_when_its_seed_goes(tmp_path, monkeypatch):
    """Review: kept after the profile was applied, the profile's "not
    wanted" read as a mark that vanished, and the topic came back on with
    the mark gone. Kept before it, the profile marks it as before."""
    from rag_lib import researcher_topics as rt
    from tests.test_review_round15_b_topics import DOC, T1, T2, _setup
    conn, uid, pid, slug, _ = _setup(tmp_path, monkeypatch, DOC, draft=False)
    _paper(conn, W2, (T4, "Four"))                           # a second seed, so W1 can go
    profiles_repo.attach_seed(conn, pid, W2)
    wizard.aggregate_draft_topics(conn, user_id=uid, slug=slug)
    wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=True)
    wizard.set_not_interest_choice(conn, user_id=uid, slug=slug, topic_id=T2, choice="filter")
    assert _by_id(_stored(conn, pid))[T2]["on"] is False
    _remove(conn, pid, "https://openalex.org/W1")             # brought T1 and T2
    got = _by_id(wizard.preview_topics(conn, user_id=uid, slug=slug)[0])
    assert got[T2]["on"] is False and got[T2].get("profile") == rt.NOT_INTEREST
    assert got[T2].get("unmatched") is True
    wizard.save_topics(conn, user_id=uid, slug=slug,
                       selected_topic_ids=[t for t, v in got.items() if v["on"]])
    saved = _by_id(_stored(conn, pid))[T2]
    assert saved["on"] is False and saved.get("profile") == rt.NOT_INTEREST


def test_a_topic_a_seed_still_brings_is_not_unmatched(live):
    """Review: "not in this rebuild" is not "no seed brings it". A UMLS
    topic (UMLS switched off here) or a PDF-text topic (content topics off)
    of a seed still there is not said to come from a removed seed; a UMLS
    topic of the removed seed is."""
    conn, uid, pid, slug = live
    T5, T6, T7 = "https://openalex.org/T10005", "https://openalex.org/T10006", "https://openalex.org/T10007"
    conn.execute("UPDATE papers SET umls_mapped_topics_json = ? WHERE openalex_id = ?",
                 (json.dumps([{"topic_id": T5, "display_name": "Five", "confidence": 0.95}]), W1))
    conn.execute("UPDATE papers SET umls_mapped_topics_json = ? WHERE openalex_id = ?",
                 (json.dumps([{"topic_id": T7, "display_name": "Seven", "confidence": 0.95}]), W2))
    papers_repo.upsert(conn, {"openalex_id": W3, "title": "W3", "source": "user_pdf"})
    conn.execute("UPDATE papers SET topics_json = ? WHERE openalex_id = ?", (json.dumps(
        {"source": "pdf_content", "primary_topic": None, "topics": [{"id": T6, "display_name": "Six"}]}), W3))
    profiles_repo.attach_seed(conn, pid, W3)
    stored = _stored(conn, pid)
    stored["topics"] += [
        {"id": T5, "display_name": "Five", "count": 1, "source": "umls", "on": True, "seeds": [W1]},
        {"id": T6, "display_name": "Six", "count": 1, "source": "pdf_content", "on": True},
        {"id": T7, "display_name": "Seven", "count": 1, "source": "umls", "on": False, "seeds": [W2]},
    ]
    conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE id = ?", (json.dumps(stored), pid))
    conn.commit()
    _remove(conn, pid, W2)
    got = _by_id(wizard.preview_topics(conn, user_id=uid, slug=slug)[0])
    assert T5 not in got or not got[T5].get("unmatched")
    assert got[T6]["on"] is True and not got[T6].get("unmatched")
    assert got[T7]["on"] is False and got[T7].get("unmatched") is True
    assert got[T3].get("unmatched") is True


def _profiled(tmp_path, monkeypatch, wanted_four=False):
    """Live, profile applied; W2 brings only T4 (Four)."""
    from tests.test_researcher_topics import _topic
    from tests.test_review_round15_b_topics import _setup
    doc = {"research_interests": [_topic("T10001", "One", weight=0.5, method="declared"),
                                  _topic("T10003", "Three", weight=0.5, method="declared")]}
    conn, uid, pid, slug, _ = _setup(tmp_path, monkeypatch, doc, draft=False)
    _paper(conn, W2, (T4, "Four"))
    profiles_repo.attach_seed(conn, pid, W2)
    wizard.aggregate_draft_topics(conn, user_id=uid, slug=slug)
    return conn, uid, pid, slug, doc


def test_the_not_wanted_default_is_the_same_on_every_path(tmp_path, monkeypatch):
    """Review 2: the rebuild and the profile view must give a topic only a
    removed seed brought the same not-wanted default. E.8: both count it as
    the seeds' still (keep), so removing the seed leaves it on. (Before E.7 the profile was re-imported after the
    removal to say so, leaving the "no" unrecorded; the interest now keeps
    the profile it applied, so it says so from the start, and the apply
    records the default it had then -- keep, W2 still a seed. Its record
    is taken out by hand, as on a list from before choices were recorded.)"""
    from rag_lib import researcher_topics as rt
    from tests.test_researcher_topics import _topic
    from tests.test_review_round15_b_topics import _setup
    doc = {"research_interests": [_topic("T10001", "One", weight=0.5, method="declared"),
                                  _topic("T10003", "Three", weight=0.5, method="declared"),
                                  _topic("T10004", "Four", weight=-0.5, method="declared")]}
    conn, uid, pid, slug, _ = _setup(tmp_path, monkeypatch, doc, draft=False)
    _paper(conn, W2, (T4, "Four"))
    profiles_repo.attach_seed(conn, pid, W2)
    wizard.aggregate_draft_topics(conn, user_id=uid, slug=slug)
    wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=True)
    _remove(conn, pid, W2)
    s = _stored(conn, pid)
    assert s[rt.CHOICES].pop("T10004") == rt.KEEP
    conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE id = ?", (json.dumps(s), pid))
    conn.commit()
    view = wizard.researcher_topics_view(conn, user_id=uid, slug=slug)
    shown = next(r for r in view["topics"]["not_interest"] if r["id"].endswith("T10004"))["choice"]
    assert shown == rt.KEEP
    prev, _ = wizard.preview_topics(conn, user_id=uid, slug=slug)
    assert prev[rt.CHOICES]["T10004"] == shown
    assert _by_id(prev)[T4]["on"] is True and _by_id(prev)[T4].get("unmatched") is True


def test_a_removed_seeds_umls_topic_put_back_after_a_failed_merge_is_marked(tmp_path, monkeypatch):
    """Review 2: a merge that fails puts every stored UMLS topic back; one
    only a removed seed brought is still said to be."""
    conn, uid, pid, slug, _ = _profiled(tmp_path, monkeypatch)
    T5 = "https://openalex.org/T10005"
    conn.execute("UPDATE papers SET umls_mapped_topics_json = ? WHERE openalex_id = ?",
                 (json.dumps([{"topic_id": T5, "display_name": "Five", "confidence": 0.95}]), W2))
    s = _stored(conn, pid)
    s["topics"].append({"id": T5, "display_name": "Five", "count": 1, "source": "umls", "on": True, "seeds": [W2]})
    conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE id = ?", (json.dumps(s), pid))
    conn.commit()
    _remove(conn, pid, W2)

    def boom(*a, **k):
        raise RuntimeError("merge down")

    monkeypatch.setattr(wizard, "_merge_umls_topics", boom)
    t5 = _by_id(wizard.preview_topics(conn, user_id=uid, slug=slug)[0])[T5]
    assert t5["on"] is True and t5.get("unmatched") is True and t5["papers"] == 0
