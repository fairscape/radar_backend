"""Round 4 of review fixes: an independent functional review's findings."""

from __future__ import annotations

import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from rag_lib.api import settings as settings_module
from rag_lib.api.services import radar, wizard
from rag_lib.db import apply_migrations, connect
from rag_lib.db.migrate import MIGRATIONS_DIR
from rag_lib.db.repos import (
    embeddings as embeddings_repo,
    gather_runs as gather_runs_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    users as users_repo,
)


def _db(tmp_path):
    conn = connect(tmp_path / "r.db")
    apply_migrations(conn)
    uid = int(users_repo.upsert(conn, "a@example.com")["id"])
    return conn, uid


def _live(conn, uid, name, threshold=0.5):
    pid = profiles_repo.upsert(conn, user_id=uid, name=name, embedding_model="m",
                               n_seed=1, topic_filters={"topics": []}, threshold=threshold)
    conn.execute("UPDATE profiles SET is_draft = 0 WHERE id = ?", (pid,))
    return pid


def _candidates(conn, pid, tag, n, *, raw=lambda j, n: 0.9 - j / 1000, pct=lambda j, n: 1 - j / n):
    for j in range(n):
        oa = f"https://openalex.org/W{tag}x{j}"
        conn.execute("INSERT INTO papers (openalex_id, title, year, source) VALUES (?, ?, 2025, 'openalex')",
                     (oa, f"{tag} {j}"))
        conn.execute(
            "INSERT INTO profile_candidates (profile_id, openalex_id, score, score_raw, score_pct, tier_used, fetched_at)"
            " VALUES (?, ?, ?, ?, ?, 'topic', datetime('now'))",
            (pid, oa, pct(j, n), raw(j, n), pct(j, n)))
    conn.commit()


# --- H1: triaged papers no longer hold the feed's places --------------------


def test_the_feed_refills_once_its_cards_are_triaged(tmp_path):
    """10 interests x 100 papers: the 200 shown are dismissed; the next feed is
    200 new ones, not the same 200 hidden behind 'All caught up'."""
    conn, uid = _db(tmp_path)
    for i in range(10):
        _candidates(conn, _live(conn, uid, f"I{i}"), i, 100)
    first = radar.daily(conn, uid)
    assert len(first.cards) == 200
    for c in first.cards:
        radar.dismiss(conn, uid, c.id, profile_slug=c.profile, on=True)
    again = radar.daily(conn, uid)
    fresh = [c for c in again.cards if c.state is None]
    assert len(fresh) == 200
    assert not {c.id for c in fresh} & {c.id for c in first.cards}
    # The dismissed ones still come back, for "show saved and dismissed".
    assert {c.id for c in again.cards if c.state == "dismissed"} == {c.id for c in first.cards}


def test_below_the_bar_is_ordered_by_similarity(tmp_path):
    """There ``score`` is a rank from each paper's own scan; the closest to the
    bar is the highest similarity."""
    conn, uid = _db(tmp_path)
    pid = _live(conn, uid, "I", threshold=0.95)
    # score_pct runs opposite to the similarity.
    _candidates(conn, pid, "b", 5, raw=lambda j, n: 0.80 + j / 100, pct=lambda j, n: 1 - j / n)
    cards = radar.daily(conn, uid, bucket="low").cards
    sims = [c.similarity for c in cards]
    assert sims == sorted(sims, reverse=True) and len(sims) == 5


# --- H2: the feedback suggestion when the top save rounds up ----------------


def _triaged(conn, pid, saves, dismisses):
    for i, (s, saved) in enumerate([(s, True) for s in saves] + [(d, False) for d in dismisses]):
        oa = f"W{i}"
        conn.execute("INSERT INTO papers (openalex_id, title, source) VALUES (?, 't', 'openalex')", (oa,))
        conn.execute(
            "INSERT INTO profile_candidates (profile_id, openalex_id, score, score_raw, tier_used, fetched_at,"
            " saved_at, dismissed_at) VALUES (?, ?, ?, ?, 'topic', datetime('now'), ?, ?)",
            (pid, oa, s, s, "2026-10-01" if saved else None, None if saved else "2026-10-01"))
    conn.commit()


def test_a_top_save_that_rounds_up_does_not_divide_by_zero(tmp_path):
    from rag_lib.feedback.autotune import recommend_threshold

    conn, uid = _db(tmp_path)
    pid = _live(conn, uid, "I", threshold=0.85)
    # Only the top save clears the dismisses, and 0.9126 rounds up to 0.913.
    _triaged(conn, pid, [0.9126], [0.86 + i * 0.001 for i in range(29)])
    th = recommend_threshold(conn, pid)
    assert th == 0.912          # rounded down: the save it was cut at stays in
    assert th <= 0.9126


# --- M1: no commit (or trial) while an import is still attaching seeds ------


def _draft_mid_import(conn, uid):
    rng = np.random.default_rng(1)
    base = rng.standard_normal(16).astype("float32")
    ids = []
    for i in range(4):
        oa = f"https://openalex.org/W{9000 + i}"
        papers_repo.upsert(conn, {"openalex_id": oa, "title": f"t{i}", "source": "openalex",
                                  "primary_topic": {"id": "T1", "display_name": "Topic 1"}, "topics": []})
        v = base + 0.3 * rng.standard_normal(16).astype("float32")
        embeddings_repo.upsert(conn, oa, "placeholder-v1", v / np.linalg.norm(v))
        ids.append(oa)
    d = wizard.create_draft(conn, user_id=uid, name="X", embedding_model="placeholder-v1")
    pid = int(profiles_repo.get_by_slug(conn, uid, d["slug"])["id"])
    run = gather_runs_repo.start(conn, profile_id=pid, user_id=uid, tier_used="prosopia_import")
    profiles_repo.attach_seed(conn, pid, ids[0])
    profiles_repo.attach_seed(conn, pid, ids[1])
    return d["slug"], pid, run


def test_a_draft_cannot_be_committed_while_its_import_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("RADAR_UMLS_ENABLED", "false")
    settings_module.get_settings.cache_clear()
    conn, uid = _db(tmp_path)
    slug, pid, run = _draft_mid_import(conn, uid)
    with pytest.raises(wizard.ImportInProgress):
        wizard.commit_draft(conn, user_id=uid, slug=slug, threshold=0.5, selected_topic_ids=["T1"])
    # Coherence (and so topics and the trial behind it) waits too, though
    # seeds have arrived -- it used to wait only while there were none.
    with pytest.raises(wizard.ImportInProgress):
        wizard.compute_draft_coherence(conn, user_id=uid, slug=slug)
    assert profiles_repo.get_by_slug(conn, uid, slug)["is_draft"] == 1
    gather_runs_repo.finish(conn, run, n_fetched=4, n_new=4, n_redup=0)
    wizard.compute_draft_coherence(conn, user_id=uid, slug=slug)
    settings_module.get_settings.cache_clear()


# --- M3: a DOI from the reference list is not the paper ---------------------


_SELF_CITE = {"title": "Refget: a reference sequence API",
              "authorships": [{"author": {"display_name": "Jane Doe"}},
                              {"author": {"display_name": "Kim Smith"}}]}
_BODY = ("Fast Sequence Collections for Genome Assemblies\nJane Doe1, Alex Roe2\n"
         "Abstract. We present a new method.\n" + "text " * 1500 +
         "\nReferences\n[1] Doe J, Smith K. Refget: a reference sequence API. doi:10.1093/bioinformatics/btab524\n")


def test_a_self_citation_doi_from_the_body_is_not_the_paper():
    from rag_lib.api.services import vault

    record = SimpleNamespace(title="Fast Sequence Collections for Genome Assemblies", body_text=_BODY,
                             doi="10.1093/bioinformatics/btab524", doi_from_body=True)
    assert not vault._verified_doi_match(record, _SELF_CITE)
    # The same DOI printed on page 1 is the paper's own, and its first author
    # is enough, as before.
    record.doi_from_body = False
    assert vault._verified_doi_match(record, _SELF_CITE)


def test_a_cited_title_in_a_short_pdfs_references_is_not_its_title():
    from rag_lib.api.services import vault

    short = "A short note\nJane Doe\nWe extend earlier work.\nReferences\nRefget: a reference sequence API. 2021.\n"
    record = SimpleNamespace(title="A short note", body_text=short)
    assert not vault.title_rule(record, _SELF_CITE)


def test_ingest_marks_a_doi_found_only_in_the_body(tmp_path, monkeypatch):
    import rag_lib.vault as vault_mod

    pages = ["Fast Sequence Collections\nJane Doe\nAbstract.", "more\nReferences\ndoi:10.1093/bioinformatics/btab524"]

    class _Page:
        def __init__(self, t):
            self.t = t

        def extract_text(self, **k):
            return self.t

        def extract_words(self, **k):
            return []

    class _Pdf:
        def __init__(self, *a, **k):
            self.pages = [_Page(t) for t in pages]
            self.metadata = {}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("pdfplumber.open", _Pdf)
    # the body is read by pdfminer (rag_lib.pdf_text): one block a page
    monkeypatch.setattr(vault_mod.pdf_text, "read_pages", lambda _p: [t.splitlines() + [""] for t in pages])
    rec = vault_mod.ingest_pdf(tmp_path / "x.pdf")
    assert rec.doi == "10.1093/bioinformatics/btab524" and rec.doi_from_body is True
    pages[0] += "\nhttps://doi.org/10.1000/own.1"
    rec = vault_mod.ingest_pdf(tmp_path / "x.pdf")
    assert rec.doi == "10.1000/own.1" and rec.doi_from_body is False


# --- L2: nightly scans off in code, not only in a deployment's .env --------


def test_nightly_scans_are_off_by_default(monkeypatch):
    monkeypatch.delenv("RADAR_NIGHTLY_GATHER_ENABLED", raising=False)
    settings_module.get_settings.cache_clear()
    try:
        assert settings_module.Settings(_env_file=None).RADAR_NIGHTLY_GATHER_ENABLED is False
    finally:
        settings_module.get_settings.cache_clear()


# --- L5 / L6: an import of someone already importing -------------------------


class _Prof:
    def __init__(self):
        from researcher_profiles import PaperRecord
        self.slug = "doe-jane"; self.name = "Jane Doe"; self.rid = "0000-0001-2345-6789"
        self.affiliation = None; self.metadata = {}; self.expertise = ""; self.soul = ""; self.grants = None
        self.papers = [PaperRecord(paper_id="p1", title="A paper about things", year=2020)]
        self.summaries = {}


class _Client:
    def resolve_orcid(self, o):
        return "doe-jane"

    def fetch_profile(self, slug):
        return _Prof()


def test_a_second_prosopia_import_of_the_same_person_is_the_running_one(tmp_path):
    from rag_lib.api.services import orcid as orcid_service, prosopia as prosopia_service

    conn = connect(tmp_path / "r.db")
    apply_migrations(conn)
    s = settings_module.get_settings()
    plan = prosopia_service.prepare_import(conn, s, user_id=1, ref="doe-jane", prosopia_client=_Client())
    run = gather_runs_repo.start(conn, profile_id=plan.profile_id, researcher_id=plan.researcher_id,
                                 user_id=1, tier_used="prosopia_import")
    with pytest.raises(orcid_service.ImportAlreadyRunning) as same:
        prosopia_service.prepare_import(conn, s, user_id=1, ref="doe-jane", prosopia_client=_Client())
    assert (same.value.slug, same.value.run_id) == (plan.draft_slug, run)
    assert conn.execute("SELECT COUNT(*) FROM profiles WHERE is_draft = 1").fetchone()[0] == 1

    # Into another draft: not that run (it fills a different draft) -- a 409.
    other = wizard.create_draft(conn, user_id=1, name="Other", embedding_model="placeholder-v1")
    with pytest.raises(orcid_service.ImportAlreadyRunning) as elsewhere:
        prosopia_service.prepare_import(conn, s, user_id=1, ref="doe-jane", prosopia_client=_Client(),
                                        into=other["slug"])
    assert elsewhere.value.slug is None
    # Into the draft that run is filling: that run.
    with pytest.raises(orcid_service.ImportAlreadyRunning) as into_same:
        prosopia_service.prepare_import(conn, s, user_id=1, ref="doe-jane", prosopia_client=_Client(),
                                        into=plan.draft_slug)
    assert into_same.value.slug == plan.draft_slug


# --- L15: a migration is judged on the violations it makes ------------------


def test_an_orphan_the_db_already_had_does_not_block_0021(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    for f in MIGRATIONS_DIR.glob("*.sql"):
        if f.stem < "0021":
            shutil.copy(f, old / f.name)
    conn = connect(tmp_path / "o.db")
    apply_migrations(conn, old)
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("INSERT INTO profile_seeds (profile_id, openalex_id) VALUES (999, 'Wzz')")
    conn.commit()
    conn.execute("PRAGMA foreign_keys=ON")
    assert "0021_profile_names_per_user" in apply_migrations(conn)


# --- through the API ----------------------------------------------------------

from tests.test_review_round3 import _requests, app  # noqa: E402,F401  (fixture)


def test_commit_and_trial_answer_409_while_the_import_runs(app, tmp_path):
    slug = _requests(app, [("POST", "/api/profiles/draft", {"json": {"name": "Mid import"}})])[0].json()["slug"]
    conn = connect(tmp_path / "radar.db")
    pid = int(conn.execute("SELECT id FROM profiles WHERE slug = ?", (slug,)).fetchone()[0])
    gather_runs_repo.start(conn, profile_id=pid, user_id=1, tier_used="orcid_import")
    conn.close()
    commit, trial = _requests(app, [
        ("POST", "/api/profiles", {"json": {"slug": slug, "threshold": 0.5}}),
        ("POST", f"/api/profiles/draft/{slug}/dry-run", {"json": {}}),
    ])
    assert commit.status_code == 409 and "still filling" in commit.json()["detail"]
    assert trial.status_code == 409


def test_reranker_comparison_counts_every_reranked_paper(app, tmp_path):
    conn = connect(tmp_path / "radar.db")
    pid = int(conn.execute("SELECT id FROM profiles WHERE slug = 'chromatin'").fetchone()[0])
    for j in range(60):
        oa = f"https://openalex.org/WR{j}"
        conn.execute("INSERT INTO papers (openalex_id, title, source) VALUES (?, 't', 'openalex')", (oa,))
        conn.execute(
            "INSERT INTO profile_candidates (profile_id, openalex_id, score, score_raw, score_blended,"
            " score_reranker_raw, tier_used, fetched_at) VALUES (?, ?, 0.9, ?, ?, 1.0, 'topic', datetime('now'))",
            (pid, oa, 0.9 - j / 1000, 0.5 + j / 1000))
    conn.commit()
    conn.close()
    res = _requests(app, [("GET", "/api/profiles/chromatin/reranker-comparison", {"params": {"limit": 10}})])[0]
    body = res.json()
    assert res.status_code == 200 and len(body["candidates"]) == 10 and body["n"] == 60


def test_a_researcher_import_already_running_says_so(app, monkeypatch):
    from rag_lib.api.services import researchers as researchers_service

    def running(*a, **k):
        raise researchers_service.ResearcherImportRunning(7, 42)
    monkeypatch.setattr(researchers_service, "prepare_researcher_import", running)
    res = _requests(app, [("POST", "/api/researchers/import",
                           {"json": {"source": "orcid", "ref": "0000-0001-2345-6789", "paper_ids": ["W1"]}})])[0]
    assert res.status_code == 200
    assert res.json() == {"researcher_id": 7, "run_id": 42, "already_running": True}
