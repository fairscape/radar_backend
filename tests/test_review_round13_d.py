"""Review round 13 (reviewer d, browser QA): what the walkthrough found.

* "Use profile topics" then "Remove profile topics" on a saved interest whose
  seeds changed since its topics were saved rebuilt the list from the seeds
  on both clicks, so the stored list came back different (17 on, then 8).
* Adding a researcher's papers to a draft that already had uploaded seeds
  linked the draft to them: "4 papers from the profile of X" over two PDFs.
* The Overview's feedback log showed the selector's name and the paper's
  DOI: debug output where a title and a time to the minute belong.
"""

from __future__ import annotations

import json

from rag_lib import researcher_topics as rt
from rag_lib.api.services import wizard
from rag_lib.db import connect
from rag_lib.db.repos import papers as papers_repo, profiles as profiles_repo, researchers as researchers_repo
from rag_lib.feedback.log import format_event_line
from tests.test_prosopia_import import StubProsopia, _app, _request, _start, _stub_openalex, _stub_profile, env  # noqa: F401
from tests.test_researcher_topics import _topic
from tests.test_review_round10 import _live_interest
from tests.test_review_round4 import _db

T1, T2, T3 = ("https://openalex.org/T10001", "https://openalex.org/T10002", "https://openalex.org/T10003")


def _saved_interest_whose_seeds_changed(tmp_path, monkeypatch):
    """A saved interest: its stored list has T1 (a seed's) and T2 (UMLS,
    from a seed since removed) on; the seeds now give T1 only. Its
    researcher's profile wants T3 and does not want T2."""
    monkeypatch.setenv("RADAR_UMLS_ENABLED", "false")
    conn, uid = _db(tmp_path)
    pid, slug = _live_interest(conn, uid)
    papers_repo.upsert(conn, {"openalex_id": "https://openalex.org/W1", "title": "Seed", "source": "openalex",
                              "primary_topic": {"id": T1, "display_name": "One"}, "topics": [{"id": T1, "display_name": "One"}]})
    profiles_repo.attach_seed(conn, pid, "https://openalex.org/W1")
    rid = researchers_repo.upsert(conn, user_id=uid, source="prosopia", key="p", name="P", document_json=json.dumps(
        {"research_interests": [_topic("T10003", "Three", weight=0.5, method="declared"),
                                _topic("T10002", "Two", weight=-0.5, method="declared")]}))
    researchers_repo.set_profile_researcher(conn, pid, rid)
    stored = {"topics": [
        {"id": T1, "display_name": "One", "count": 2, "on": True},
        {"id": T2, "display_name": "Two", "count": 1, "on": True, "source": "umls"},
    ]}
    conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE id = ?", (json.dumps(stored), pid))
    conn.commit()
    return conn, uid, pid, slug, stored


def _states(filters):
    return {t["id"]: bool(t.get("on", True)) for t in filters["topics"]}


def test_d_apply_then_remove_leaves_a_saved_interests_topics_as_they_were(tmp_path, monkeypatch):
    conn, uid, pid, slug, stored = _saved_interest_whose_seeds_changed(tmp_path, monkeypatch)
    # The edit page shows the recomputed list -- that is the preview: T1
    # from the seeds, T2 kept as saved but marked as no seed's any more
    # (FUNCTIONAL_SPEC T.2-2; it used to drop out of the preview)...
    preview, _new = wizard.preview_topics(conn, user_id=uid, slug=slug)
    assert _states(preview) == {T1: True, T2: True}
    by_id = {t["id"]: t for t in preview["topics"]}
    assert by_id[T2].get("unmatched") is True and by_id[T2]["count"] == 0 and not by_id[T1].get("unmatched")
    # ...and the stored list is untouched by looking.
    assert _states(profiles_repo.topic_filters(conn, pid)) == {T1: True, T2: True}

    applied = wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=True)
    # Folded into the stored list, not into a rebuild: T2 is still there,
    # off because the profile says no, with its state remembered; T3 joined.
    assert _states(applied) == {T1: True, T2: False, T3: True}
    by_id = {t["id"]: t for t in applied["topics"]}
    assert by_id[T2]["profile"] == rt.NOT_INTEREST and by_id[T2][rt.BEFORE] is True
    assert by_id[T3]["source"] == rt.SOURCE and rt.is_applied(applied)
    assert _states(profiles_repo.topic_filters(conn, pid)) == {T1: True, T2: False, T3: True}

    removed = wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=False)
    assert _states(removed) == {T1: True, T2: True}
    assert not rt.is_applied(removed)
    assert all("profile" not in t and rt.BEFORE not in t for t in removed["topics"])
    assert _states(profiles_repo.topic_filters(conn, pid)) == _states(stored)
    conn.close()


def test_d_re_applying_after_the_profile_changed_drops_its_stale_marks(tmp_path, monkeypatch):
    """Since FUNCTIONAL_SPEC E.7 the interest keeps the copy of the profile
    it applied: applying again while applied changes nothing, however the
    researcher changed. Taking the profile off and applying it again reads
    the researcher as now -- and drops the old profile's marks."""
    conn, uid, pid, slug, _stored = _saved_interest_whose_seeds_changed(tmp_path, monkeypatch)
    first = wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=True)
    # The profile is re-read: it no longer minds T2 and wants T1 instead of T3.
    conn.execute("UPDATE researchers SET document_json = ? WHERE id = (SELECT researcher_id FROM profiles WHERE id = ?)",
                 (json.dumps({"research_interests": [_topic("T10001", "One", weight=0.5, method="declared")]}), pid))
    conn.commit()
    same = wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=True)
    assert _states(same) == _states(first) == {T1: True, T2: False, T3: True}
    assert {t["id"]: t.get("profile") for t in same["topics"]} == {t["id"]: t.get("profile") for t in first["topics"]}
    wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=False)
    again = wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=True)
    assert _states(again) == {T1: True, T2: True}          # T2 back as it was, T3 gone with the profile
    by_id = {t["id"]: t for t in again["topics"]}
    assert by_id[T1]["profile"] == rt.INTEREST and "profile" not in by_id[T2]
    conn.close()


def test_d_a_draft_still_rebuilds_from_its_seeds(tmp_path, monkeypatch):
    """A draft's list is what step 3 shows, rebuilt from the seeds every
    time; applying the profile there keeps doing that (a new seed's topic
    appears, on)."""
    conn, uid, pid, slug, _stored = _saved_interest_whose_seeds_changed(tmp_path, monkeypatch)
    conn.execute("UPDATE profiles SET is_draft = 1 WHERE id = ?", (pid,))
    conn.commit()
    applied = wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=True)
    assert _states(applied) == {T1: True, T3: True}        # T2: no seed gives it, the rebuild drops it
    removed = wizard.set_researcher_topics(conn, user_id=uid, slug=slug, enabled=False)
    assert _states(removed) == {T1: True}
    conn.close()


def test_d_importing_into_a_draft_with_uploaded_seeds_does_not_make_it_the_researchers(env):  # noqa: F811
    app = _app(StubProsopia(_stub_profile()), _stub_openalex())
    slug = _request(app, "POST", "/api/profiles/draft", json={"name": "Mine"}).json()["slug"]
    conn = connect(env)
    pid = int(conn.execute("SELECT id FROM profiles WHERE slug = ?", (slug,)).fetchone()["id"])
    papers_repo.upsert(conn, {"openalex_id": "local:pdf1", "title": "My PDF", "source": "user_pdf", "body_text": "words " * 30})
    profiles_repo.attach_seed(conn, pid, "local:pdf1")
    conn.commit()
    conn.close()
    resp = _start(app, ref="sheffield-nathan", into=slug)
    assert resp.status_code == 200, resp.text
    conn = connect(env)
    try:
        row = conn.execute("SELECT researcher_id FROM profiles WHERE id = ?", (pid,)).fetchone()
        assert row["researcher_id"] is None
        n = conn.execute("SELECT COUNT(*) AS n FROM profile_seeds WHERE profile_id = ?", (pid,)).fetchone()["n"]
        assert n == 6          # the PDF and the five imported papers
    finally:
        conn.close()


def test_d_feedback_line_reads_as_a_sentence_not_a_log_record():
    line = format_event_line({"ts": "2026-10-21T20:27:06Z", "action": "dismissed", "selector": "CentroidSelector",
                              "openalex_id": "https://openalex.org/W1", "doi": "10.1186/s12887-026-07822-2",
                              "title": "Heart-rate  variability in\nneonates", "score": 0.9751})
    assert line == "2026-10-21T20:27  DISMISSED  Heart-rate variability in neonates  score=0.975"
    # No title stored: the DOI, then the id.
    assert "10.1186/s12887-026-07822-2" in format_event_line({"ts": "2026-10-21T20:27:06Z", "action": "saved", "openalex_id": "x",
                                                              "doi": "10.1186/s12887-026-07822-2", "title": None, "score": None})
    assert "https://openalex.org/W1  score=NA" in format_event_line({"ts": "", "action": "saved", "openalex_id": "https://openalex.org/W1", "score": None})
