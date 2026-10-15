"""Fixes from the 2026-10-14 integration review: links between features."""

from __future__ import annotations

import json

from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import (
    candidates as candidates_repo,
    gather_runs as gather_runs_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    researchers as researchers_repo,
    users as users_repo,
    vault as vault_repo,
)


def _db(tmp_path):
    conn = connect(tmp_path / "r.db")
    apply_migrations(conn)
    uid = int(users_repo.upsert(conn, "a@example.com")["id"])
    return conn, uid


def _interest(conn, uid, name):
    return profiles_repo.upsert(conn, user_id=uid, name=name, embedding_model="placeholder-v1",
                                n_seed=0, topic_filters={"topics": []}, threshold=0.8)


def test_a_feed_paper_made_a_seed_of_another_interest_is_listed_with_its_seeds(tmp_path):
    conn, uid = _db(tmp_path)
    a, b = _interest(conn, uid, "Alpha"), _interest(conn, uid, "Beta")
    papers_repo.upsert(conn, {"openalex_id": "W1", "title": "Saved in alpha", "source": "openalex_gatherer"})
    candidates_repo.insert_dedup(conn, profile_id=a, openalex_id="W1", score=0.9)
    conn.execute("UPDATE profile_candidates SET saved_at = datetime('now')")
    conn.commit()
    profiles_repo.attach_seed(conn, b, "W1")
    slug_b = profiles_repo.get(conn, b)["slug"]
    # It used to count only as a seed that was uploaded or a saved researcher's.
    assert [r["openalex_id"] for r in vault_repo.list_for_user(conn, uid, tag=slug_b)] == ["W1"]
    assert vault_repo.tag_counts_for_user(conn, uid)[slug_b] == 1


def test_forgetting_a_researcher_keeps_the_seeds_in_the_vault_and_the_interests_runs(tmp_path):
    conn, uid = _db(tmp_path)
    pid = _interest(conn, uid, "Gamma")
    rid = researchers_repo.upsert(conn, user_id=uid, source="orcid", key="0000-0001-2345-6789", name="R")
    rid = int(rid["id"]) if not isinstance(rid, int) else rid
    for oa in ("W1", "W2"):
        papers_repo.upsert(conn, {"openalex_id": oa, "title": oa, "source": "orcid"})
        researchers_repo.attach_paper(conn, rid, oa)
    profiles_repo.attach_seed(conn, pid, "W1")        # W1 seeds the interest, W2 does not
    run_into_interest = gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="orcid_import", researcher_id=rid)
    run_alone = gather_runs_repo.start(conn, profile_id=None, user_id=uid, tier_used="researcher_import", researcher_id=rid)

    researchers_repo.delete(conn, rid)
    assert [r["openalex_id"] for r in vault_repo.list_for_user(conn, uid)] == ["W1"]
    ids = {r[0] for r in conn.execute("SELECT id FROM gather_runs")}
    assert run_into_interest in ids and run_alone not in ids


def test_runs_left_open_by_a_restart_are_closed_at_boot(tmp_path):
    conn, uid = _db(tmp_path)
    pid = _interest(conn, uid, "Delta")
    open_run = gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="manual")
    done = gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="manual")
    gather_runs_repo.finish(conn, done, n_fetched=1, n_new=1, n_redup=0)
    assert gather_runs_repo.close_interrupted(conn) == 1
    row = conn.execute("SELECT finished_at, error FROM gather_runs WHERE id = ?", (open_run,)).fetchone()
    assert row[0] is not None and row[1] == gather_runs_repo.INTERRUPTED
    assert conn.execute("SELECT error FROM gather_runs WHERE id = ?", (done,)).fetchone()[0] is None


def test_an_interest_called_new_does_not_take_the_wizards_address(tmp_path):
    conn, uid = _db(tmp_path)
    assert profiles_repo.get(conn, _interest(conn, uid, "New"))["slug"] != "new"


def test_a_stored_umls_topic_is_kept_only_for_a_seed_that_cannot_be_read(tmp_path):
    from rag_lib.api.services.wizard import _keep_stored_umls

    conn, uid = _db(tmp_path)
    pid = _interest(conn, uid, "Eps")
    stored = {"topics": [
        {"id": "T1", "display_name": "From the unreadable seed", "source": "umls", "on": False, "seeds": ["W1"]},
        {"id": "T2", "display_name": "From a removed seed", "source": "umls", "on": True, "seeds": ["W9"]},
        {"id": "T3", "display_name": "Stored before seeds were recorded", "source": "umls", "on": True},
    ]}
    conn.execute("UPDATE profiles SET topic_filters_json = ? WHERE id = ?", (json.dumps(stored), pid))
    conn.commit()
    kept = _keep_stored_umls(conn, pid, {"topics": []}, {"W1"})
    assert [t["id"] for t in kept["topics"]] == ["T1", "T3"] and kept["topics"][0]["on"] is False
    # The whole merge failed: everything stored is kept.
    assert len(_keep_stored_umls(conn, pid, {"topics": []}, None)["topics"]) == 3
