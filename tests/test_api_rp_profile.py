"""API tests for ``POST /api/profiles/draft/from-profile``.

Runs the import inline through the injected ``FakeOpenAlexClient`` (same
seam as the ORCID tests) and stubs the phrase→topic mapping so no ollama
or topic index is needed. The Sheffield fixture's ORCID is the one the
ORCID test corpus is built around, so the works, seeds and concept list
are the same as in ``test_api_orcid.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_lib.api.services import rp_profile_import as rp
from rag_lib.db import connect
from tests.test_api_orcid import HDR, HDR_B, _app, _fake, _request, _requests, _select, env  # noqa: F401
from tests.test_orcid_import import ORCID

FIXTURE = Path(__file__).parent / "fixtures" / "sheffield_profile.jsonld"


def _profile_text(**overrides) -> str:
    d = json.loads(FIXTURE.read_text(encoding="utf-8"))
    d.update(overrides)
    return json.dumps(d)


@pytest.fixture()
def stub_mapping(monkeypatch):
    """expertise 'chromatin-accessibility-analysis' → the corpus's main topic (already on),
    'dna-methylation-analysis' → the middle-author-only topic (off by default) and a brand-new
    topic; not_interest 'insect-evolutionary-biology' → the single-cell topic carried by a seed."""
    def fake_map(phrases, *, settings, min_sim, top_k):
        hits = []
        for ph in phrases:
            if ph == "chromatin-accessibility-analysis":
                hits.append(rp.PhraseHit(ph, "https://openalex.org/T10222", "Genomics and Chromatin Dynamics", 0.91))
            elif ph == "dna-methylation-analysis":
                hits.append(rp.PhraseHit(ph, "https://openalex.org/T10885", "Gene expression and cancer classification", 0.62))
                hits.append(rp.PhraseHit(ph, "https://openalex.org/T99001", "DNA Methylation and Epigenetics", 0.81))
            elif ph == "insect-evolutionary-biology":
                hits.append(rp.PhraseHit(ph, "https://openalex.org/T11289", "Single-cell and spatial transcriptomics", 0.58))
        return hits, []
    monkeypatch.setattr(rp, "map_phrases_to_topics", fake_map)


def test_profile_with_orcid_runs_the_import_and_applies_the_signal(env, stub_mapping):
    app = _app(_fake())
    r = _request(app, "POST", "/api/profiles/draft/from-profile", json={"profile_json": _profile_text()}, headers=HDR)
    assert r.status_code == 200, r.text
    start = r.json()
    assert start["mode"] == "orcid" and start["orcid"] == ORCID and start["run_id"]
    assert start["name"] == "Nathan C. Sheffield"
    slug = start["slug"]

    st = _request(app, "GET", f"/api/profiles/draft/{slug}/import/{start['run_id']}", headers=HDR).json()
    assert st["run"]["finished_at"] and st["run"]["error"] is None, st
    res = st["result"]
    assert res["phase"] == "fetch" and res["n_works"] == 6 and res["n_seeds"] == 0 and res["author"]["orcid"] == ORCID
    assert res["rp"]["level"] == "full" and res["rp"]["provenance"] == "self_published"
    assert res["rp"]["n_expertise"] == 9 and res["rp"]["n_not_interests"] == 4

    # Phase B with the default pick; the profile's signal applies on top of the seed-only concepts.
    _select(app, slug)
    # Concepts are now built from the selected seeds only, so the middle-author-only
    # topic T10885 is absent; the expertise hit that used to switch it on adds nothing,
    # while the not_interest still switches the seeded single-cell topic off and the
    # expertise phrase still adds the new methylation topic.

    seeds = _request(app, "GET", f"/api/profiles/draft/{slug}/seeds", headers=HDR).json()
    assert [s["id"] for s in seeds] == ["W1008", "W1002", "W1001"]

    topics = _request(app, "GET", f"/api/profiles/draft/{slug}/topics", headers=HDR).json()
    by = {t["name"]: t for t in topics}
    assert by["Genomics and Chromatin Dynamics"]["on"] is True
    assert by["Single-cell and spatial transcriptomics"]["on"] is False      # not_interest
    assert "Gene expression and cancer classification" not in by             # not carried by a selected seed
    assert by["DNA Methylation and Epigenetics"]["source"] == "rp_expertise" and by["DNA Methylation and Epigenetics"]["on"] is True

    conn = connect(env["db"])
    row = conn.execute("SELECT orcid, researcher_name, rp_meta_json, topic_filters_json FROM profiles WHERE slug = ?", (slug,)).fetchone()
    meta = json.loads(row["rp_meta_json"])
    assert row["orcid"] == ORCID and row["researcher_name"] == "Nathan C. Sheffield"
    assert meta["summary"].startswith("Nathan Sheffield") and meta["mapped"]["not_interests"][0]["phrase"] == "insect-evolutionary-biology"
    tf = json.loads(row["topic_filters_json"])
    off = next(t for t in tf["topics"] if t["id"].endswith("T11289"))
    assert off["rp_off_by"] == ["insect-evolutionary-biology"]
    conn.close()

    drafts = _request(app, "GET", "/api/profiles/drafts", headers=HDR).json()
    assert drafts[0]["slug"] == slug and drafts[0]["rp"] is True and drafts[0]["orcid"] == ORCID

    # The detail/list mappers expose the metadata.
    detail = _request(app, "GET", f"/api/profiles/{slug}/detail", headers=HDR)
    # (drafts are hidden from the profile routes; commit first)
    commit = _request(app, "POST", "/api/profiles", json={"slug": slug, "threshold": 0.8, "selected_topic_ids": [t["id"] for t in topics if t["on"]]}, headers=HDR)
    assert commit.status_code == 200, commit.text
    assert commit.json()["rp_meta"]["affiliation"] == "University of Virginia"
    assert commit.json()["rp_meta"]["expertise"][0] == "chromatin-accessibility-analysis"
    del detail


def test_profile_without_orcid_creates_a_pdf_mode_draft(env, stub_mapping):
    app = _app(_fake())
    text = _profile_text(rid="local:nathan-abc123", **{"@id": "#me"})
    r = _request(app, "POST", "/api/profiles/draft/from-profile", json={"profile_json": text, "name": "Methylation watch"}, headers=HDR)
    assert r.status_code == 200, r.text
    start = r.json()
    assert start["mode"] == "pdf" and start["run_id"] is None and start["orcid"] is None
    assert start["name"] == "Methylation watch"
    assert any("upload PDFs" in w for w in start["warnings"])

    conn = connect(env["db"])
    row = conn.execute("SELECT orcid, researcher_name, rp_meta_json, is_draft FROM profiles WHERE slug = ?", (start["slug"],)).fetchone()
    assert row["orcid"] is None and row["is_draft"] == 1 and row["researcher_name"] == "Nathan C. Sheffield"
    assert json.loads(row["rp_meta_json"])["rid"] == "local:nathan-abc123"
    assert conn.execute("SELECT COUNT(*) FROM gather_runs").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM profile_seeds").fetchone()[0] == 0
    conn.close()

    drafts = _request(app, "GET", "/api/profiles/drafts", headers=HDR).json()
    assert drafts[0]["rp"] is True and drafts[0]["orcid"] is None and drafts[0]["import_run_id"] is None
    assert _request(app, "GET", "/api/profiles/drafts", headers=HDR_B).json() == []


def test_pdf_mode_draft_applies_the_signal_at_step_3(env, stub_mapping):
    """No import ran, so Step 3 aggregates from the uploaded seeds and only then applies the profile."""
    from rag_lib.db.repos import embeddings as embeddings_repo, papers as papers_repo, profiles as profiles_repo
    from tests.test_orcid_import import CORPUS

    app = _app(_fake())
    text = _profile_text(rid="local:nathan-abc123", **{"@id": "#me"})
    start = _request(app, "POST", "/api/profiles/draft/from-profile", json={"profile_json": text}, headers=HDR).json()

    # Stand in for a PDF upload: attach the corpus's single-cell paper (T11289) as the seed.
    conn = connect(env["db"])
    pid = int(profiles_repo.get_by_slug(conn, 1, start["slug"])["id"])
    work = next(w for w in CORPUS if w["id"].endswith("W1008"))
    from rag_lib.api.services import orcid_import as oi
    rec = oi.build_records([work], ORCID, None, client=_fake())[0][0]
    papers_repo.upsert(conn, oi._paper_dict(rec))
    embeddings_repo.upsert(conn, rec.paper.openalex_id, "placeholder-v1", [0.1] * 8)
    profiles_repo.attach_seed(conn, pid, rec.paper.openalex_id)
    conn.commit(); conn.close()

    topics = _request(app, "GET", f"/api/profiles/draft/{start['slug']}/topics", headers=HDR).json()
    by = {t["name"]: t for t in topics}
    assert by["Single-cell and spatial transcriptomics"]["on"] is False          # not_interest hit
    assert by["DNA Methylation and Epigenetics"]["source"] == "rp_expertise"    # expertise addition

    conn = connect(env["db"])
    meta = json.loads(conn.execute("SELECT rp_meta_json FROM profiles WHERE id = ?", (pid,)).fetchone()[0])
    assert "mapped" in meta  # cached after the first Step 3 call
    conn.close()


def test_bad_document_is_422_and_creates_nothing(env):
    app = _app(_fake())
    for body in ({"profile_json": "not json"}, {"profile_json": json.dumps({"@type": "Person"})}):
        r = _request(app, "POST", "/api/profiles/draft/from-profile", json=body, headers=HDR)
        assert r.status_code == 422, r.text
    conn = connect(env["db"])
    assert conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0] == 0
    conn.close()


def test_unknown_orcid_is_404_and_creates_nothing(env):
    app = _app(_fake())
    text = _profile_text(rid="0000-0002-1825-0097", **{"@id": "https://orcid.org/0000-0002-1825-0097"}, identifier=[])
    r = _request(app, "POST", "/api/profiles/draft/from-profile", json={"profile_json": text}, headers=HDR)
    assert r.status_code == 404 and "not found on OpenAlex" in r.json()["detail"]
    conn = connect(env["db"])
    assert conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0] == 0
    conn.close()


def test_author_id_fallback_when_orcid_unknown_to_openalex(env, stub_mapping):
    """OpenAlex does not know the ORCID but the profile names the OpenAlex author id."""
    from tests.test_orcid_import import AUTHOR, CORPUS
    from tests.fake_openalex_client import FakeOpenAlexClient

    other = "0000-0002-1825-0097"
    fake = FakeOpenAlexClient(authors={ORCID: AUTHOR}, orcid_works=CORPUS)  # only the fixture's ORCID is known
    app = _app(fake)
    text = _profile_text(rid=other, **{"@id": f"https://orcid.org/{other}"})
    r = _request(app, "POST", "/api/profiles/draft/from-profile", json={"profile_json": text}, headers=HDR)
    assert r.status_code == 200, r.text
    st = _request(app, "GET", f"/api/profiles/draft/{r.json()['slug']}/import/{r.json()['run_id']}", headers=HDR).json()
    # The fake returns the same corpus for any ORCID, so the import proceeds via the author id.
    assert st["run"]["error"] is None and st["result"]["author"]["openalex_author_id"] == AUTHOR["id"]
    assert st["result"]["phase"] == "fetch" and st["result"]["n_works"] == 6


def test_inflight_same_orcid_is_409(env, stub_mapping):
    app = _app(_fake())
    state: dict = {}

    def between(i, out):
        if i == 0:
            state["start"] = out[0].json()
            conn = connect(env["db"])
            conn.execute("UPDATE gather_runs SET finished_at = NULL WHERE id = ?", (state["start"]["run_id"],))
            conn.commit(); conn.close()

    resps = _requests(app, [
        ("POST", "/api/profiles/draft/from-profile", {"json": {"profile_json": _profile_text()}, "headers": HDR}),
        ("POST", "/api/profiles/draft/from-orcid", {"json": {"orcid": ORCID}, "headers": HDR}),
    ], between=between)
    assert resps[0].status_code == 200
    assert resps[1].status_code == 409 and resps[1].json()["detail"]["slug"] == state["start"]["slug"]
