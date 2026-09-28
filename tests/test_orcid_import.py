"""Unit tests for the ORCID import service (no HTTP, no network)."""

from __future__ import annotations

import hashlib
import json
import stat

import pytest

from rag_lib.api.services import orcid_import as oi
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import embeddings as embeddings_repo, papers as papers_repo, profiles as profiles_repo, users as users_repo
from tests.fake_openalex_client import FakeOpenAlexClient, canned_authorship, canned_openalex_work

ORCID = "0000-0001-5643-4068"
ME = "https://openalex.org/A5055872618"


def _me(position="first", corresponding=False):
    return canned_authorship("Nathan C. Sheffield", position=position, orcid=ORCID,
                             author_id=ME, corresponding=corresponding)


def _other(name, position="middle"):
    return canned_authorship(name, position=position)


def _work(i, *, title, year, position="first", corresponding=False, work_type="article",
          cited=0, topic=("T10222", "Genomics and Chromatin Dynamics"), doi=None, abstract=True,
          extra_authors=2, me=True):
    auths = []
    if position == "first" and me:
        auths.append(_me("first", corresponding))
    for k in range(extra_authors):
        auths.append(_other(f"Coauthor {k}", "middle"))
    if position == "middle" and me:
        auths.insert(1, _me("middle", corresponding))
    if position == "last" and me:
        auths.append(_me("last", corresponding))
    return canned_openalex_work(
        doi=doi or f"10.1000/w{i}", openalex_id=f"W{1000 + i}", title=title, year=year,
        venue="Journal", abstract_words=["some", "abstract", "text"] if abstract else None,
        primary_topic_id=topic[0], primary_topic_name=topic[1],
        work_type=work_type, cited_by_count=cited, authorships=auths,
    )


CORPUS = [
    _work(1, title="LOLA: enrichment analysis for genomic region sets and regulatory elements", year=2015, cited=500),
    _work(2, title="PEPATAC: an optimized pipeline for ATAC-seq data analysis with serial alignments", year=2021,
          position="last", corresponding=True),
    _work(3, title="A middle-author consortium paper about something else entirely here", year=2012,
          position="middle", topic=("T10885", "Gene expression and cancer classification"), extra_authors=60),
    _work(4, title="Additional file 1 of COCOA: coordinate covariation analysis of epigenetic heterogeneity", year=2020,
          work_type="dataset", topic=("T10499", "Folate and B Vitamins Research")),
    _work(5, title="LOLA: enrichment analysis for genomic region sets and regulatory elements", year=2015,
          doi="10.9999/preprint", cited=3, abstract=False),  # preprint copy, same title
    _work(6, title="A software deposit that the type gate removes", year=2019, work_type="software"),
    _work(7, title="Paper where the ORCID is not among the authors at all", year=2018, me=False,
          position="middle"),
    _work(8, title="Corresponding-but-middle author paper on single-cell methods", year=2023,
          position="middle", corresponding=True, topic=("T11289", "Single-cell and spatial transcriptomics")),
]
AUTHOR = {
    "id": ME, "display_name": "Nathan C. Sheffield", "works_count": 8,
    "last_known_institutions": [{"display_name": "University of Virginia", "ror": "https://ror.org/0153tk833"}],
    "topics": [{"id": "https://openalex.org/T10222", "display_name": "Genomics and Chromatin Dynamics",
                "subfield": {"display_name": "Molecular Biology"}, "field": {"display_name": "Biochemistry"}}],
}


def _client():
    return FakeOpenAlexClient(authors={ORCID: AUTHOR}, orcid_works=CORPUS)


# --------------------------------------------------------------------------- normalize


@pytest.mark.parametrize("raw", [ORCID, f"https://orcid.org/{ORCID}", f"orcid.org/{ORCID}/", " 0000-0001-5643-4068 "])
def test_normalize_orcid_accepts_forms(raw):
    assert oi.normalize_orcid(raw) == ORCID


def test_normalize_orcid_uppercases_x():
    # 0000-0002-1825-0097 has check digit 7; 0000-0002-9079-593X is a valid X example.
    assert oi.normalize_orcid("0000-0002-9079-593x") == "0000-0002-9079-593X"


@pytest.mark.parametrize("raw", ["0000-0001-5643-4069", "0000-0001-5643-406", "abcd", "",
                                 "0000-0001-5643-4068,publication_year:2020", "../../etc"])
def test_normalize_orcid_rejects(raw):
    with pytest.raises(ValueError):
        oi.normalize_orcid(raw)


# --------------------------------------------------------------------------- authorship


def test_classify_authorship_positions():
    assert oi.classify_authorship(CORPUS[0], ORCID) == ("first", False, 1, 3)
    assert oi.classify_authorship(CORPUS[1], ORCID) == ("last", True, 3, 3)
    pos, corr, idx, total = oi.classify_authorship(CORPUS[2], ORCID)
    assert (pos, corr, total) == ("middle", False, 61)
    assert oi.classify_authorship(CORPUS[6], ORCID) is None


def test_classify_authorship_matches_by_author_id_when_orcid_missing():
    work = _work(9, title="No ORCID on the authorship but the OpenAlex id matches", year=2020)
    work["authorships"][0]["author"]["orcid"] = None
    assert oi.classify_authorship(work, ORCID) is None
    assert oi.classify_authorship(work, ORCID, ME)[0] == "first"


# --------------------------------------------------------------------------- records / seeds / topics


def test_build_records_applies_only_rp_filters():
    records, report = oi.build_records(CORPUS, ORCID, ME, client=_client())
    assert report == {"fetched": 8, "dropped_type": 1, "dropped_unusable": 0,
                      "dropped_not_author": 1, "duplicate_ids": 0, "kept": 6,
                      "orcid_claimed": 0, "claimed_matched": 0, "dropped_unclaimed_field": 0}
    kept_ids = {r.paper.openalex_id for r in records}
    assert "W1006" not in kept_ids and "W1007" not in kept_ids
    assert "W1004" in kept_ids  # dataset is kept in the corpus ...
    rec4 = next(r for r in records if r.paper.openalex_id == "W1004")
    assert not rec4.seed_eligible and rec4.topic_weight == 0.0  # ... but is neither seed nor topic evidence
    rec5 = next(r for r in records if r.paper.openalex_id == "W1005")
    assert oi._paper_dict(rec5)["abstract"] is None  # "" must not clobber a stored abstract


def test_select_seeds_lead_only_deduped_newest_first():
    records, _ = oi.build_records(CORPUS, ORCID, ME, client=_client())
    seeds, warnings = oi.select_seeds(records, max_seeds=40)
    ids = [r.paper.openalex_id for r in seeds]
    # W1008 (2023, corresponding-middle), W1002 (2021, last), W1001 (2015 first; the
    # preprint copy W1005 collapses into it because W1001 has the abstract + citations)
    assert ids == ["W1008", "W1002", "W1001"]
    assert any("duplicate" in w for w in warnings)


def test_select_seeds_cap_and_fallback():
    records, _ = oi.build_records(CORPUS, ORCID, ME, client=_client())
    seeds, warnings = oi.select_seeds(records, max_seeds=2)
    assert [r.paper.openalex_id for r in seeds] == ["W1008", "W1002"]
    assert any("keeping the 2 newest" in w for w in warnings)

    middle_only = [r for r in records if r.paper.openalex_id in ("W1003", "W1004")]
    seeds, warnings = oi.select_seeds(middle_only, max_seeds=40)
    assert [r.paper.openalex_id for r in seeds] == ["W1003"]  # dataset excluded even in fallback
    assert any("most-cited" in w for w in warnings)


def test_researcher_topic_filters_weights_and_stamps():
    records, _ = oi.build_records(CORPUS, ORCID, ME, client=_client())
    tf = oi.researcher_topic_filters(records, top_k=15)
    by_name = {t["display_name"]: t for t in tf["topics"]}
    # Each canned work carries its topic as primary_topic AND in topics[] (as
    # OpenAlex does), so a paper tallies its topic twice. Lead works weigh 2x:
    # W1001, W1002, W1005 (lead, T10222) -> 3*2*2 = 12; W1008 -> 4; middle W1003 -> 2.
    assert by_name["Genomics and Chromatin Dynamics"]["count"] == 12
    assert by_name["Single-cell and spatial transcriptomics"]["count"] == 4
    assert by_name["Gene expression and cancer classification"]["count"] == 2
    assert "Folate and B Vitamins Research" not in by_name  # dataset weight 0
    assert all(t["on"] is True and t["source"] == "orcid" for t in tf["topics"])
    assert len(oi.researcher_topic_filters(records, top_k=2)["topics"]) == 2


# --------------------------------------------------------------------------- persistence


def test_persist_records_keeps_existing_abstract_and_attaches_after_embedding(tmp_path):
    conn = connect(tmp_path / "t.db")
    apply_migrations(conn)
    user = users_repo.upsert(conn, "a@b.org")
    pid = profiles_repo.create_draft(conn, user_id=int(user["id"]), name="x", slug="x",
                                     embedding_model="placeholder-v1")
    papers_repo.upsert(conn, {"openalex_id": "W1005", "title": "old title",
                              "abstract": "an abstract from an earlier upload", "source": "user_pdf"})

    class _S:
        RADAR_ORCID_MAX_SEEDS = 40
    records, _ = oi.build_records(CORPUS, ORCID, ME, client=_client())
    seeds, _ = oi.select_seeds(records, 40)
    ticks = []
    n = oi.persist_records(conn, _S, profile_id=pid, records=records, seeds=seeds,
                           embedding_model="placeholder-v1", on_embed=lambda: ticks.append(1))
    assert n == 3 and len(ticks) == 3
    row = papers_repo.get_by_openalex_id(conn, "W1005")
    assert row["abstract"] == "an abstract from an earlier upload"
    assert set(profiles_repo.list_seed_openalex_ids(conn, pid)) == {"W1008", "W1002", "W1001"}
    assert embeddings_repo.has(conn, "W1001", "placeholder-v1")
    assert oi.seeds_with_vectors(conn, pid, "placeholder-v1") == 3
    # second pass: nothing re-embedded
    assert oi.persist_records(conn, _S, profile_id=pid, records=records, seeds=seeds,
                              embedding_model="placeholder-v1") == 0


# --------------------------------------------------------------------------- RP by-product


def test_write_rp_profile_layout_and_hashes(tmp_path):
    records, _ = oi.build_records(CORPUS, ORCID, ME, client=_client())
    seeds, _ = oi.select_seeds(records, 40)
    author = oi.resolve_author(_client(), ORCID)
    out = oi.write_rp_profile(tmp_path / "rp", user_id=7, slug="sheffield", author=author,
                              orcid=ORCID, name="Nathan C. Sheffield", records=records, seeds=seeds)
    assert out == tmp_path / "rp" / "7" / "sheffield"
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    profile = json.loads((out / "profile.jsonld").read_text())
    papers = (out / "sources" / "papers.jsonld").read_bytes()
    part = profile["hasPart"][0]
    assert part["bytes"] == len(papers) and part["sha256"] == hashlib.sha256(papers).hexdigest()
    assert profile["rid"] == ORCID and profile["level"] == "lite" and profile["affiliation"]["name"] == "University of Virginia"
    assert profile["paper_stats"]["total"] == 6 and profile["paper_stats"]["first"] == 3
    doc = json.loads(papers)
    ids = [p["@id"] for p in doc["hasPart"]]
    assert len(ids) == len(set(ids)) == 6
    pids = [p["paper_id"] for p in doc["hasPart"]]
    assert all(oi._PAPER_ID_RE.match(p) for p in pids) and len(set(pids)) == 6
    side = json.loads((out / "meta" / "openalex_topics.json").read_text())
    assert sum(1 for s in side if s["is_seed"]) == 3
    # rewriting replaces atomically
    out2 = oi.write_rp_profile(tmp_path / "rp", user_id=7, slug="sheffield", author=author,
                               orcid=ORCID, name="Nathan C. Sheffield", records=records, seeds=seeds)
    assert out2 == out and (out / "profile.jsonld").is_file()


# --------------------------------------------------------------------------- identity (ORCID registry)


def _ecology_work(i, *, year, title):
    """A lead-author work from a *different* Tim W. Clark merged into the same OpenAlex author."""
    w = _work(i, title=title, year=year, topic=("T11937", "Wildlife Ecology and Conservation"))
    w["primary_topic"]["field"] = {"id": "F23", "display_name": "Environmental Science"}
    w["topics"][0]["field"] = {"id": "F23", "display_name": "Environmental Science"}
    return w


MERGED = CORPUS + [
    _ecology_work(20, year=1993, title="Black-footed ferret recovery and prairie dog management"),
    _ecology_work(21, year=2001, title="Grizzly bear conservation in the Greater Yellowstone ecosystem"),
    _ecology_work(22, year=2024, title="Wolves, elk and policy sciences: a retrospective"),
]


def test_claimed_works_match_by_doi_or_title():
    c = oi.ClaimedWorks(dois={"10.1000/w1"}, title_keys={oi.title_key("PEPATAC: an optimized pipeline for ATAC-seq data analysis with serial alignments")}, n_works=2)
    assert c.matches("https://doi.org/10.1000/W1", "whatever")
    assert c.matches(None, "PEPATAC: An Optimized Pipeline for ATAC-seq Data Analysis With Serial Alignments")
    assert not c.matches("10.1000/other", "Some other title")


def test_build_records_drops_unclaimed_works_in_foreign_fields():
    claimed = oi.ClaimedWorks(dois={"10.1000/w1", "10.1000/w2"}, n_works=2)
    records, report = oi.build_records(MERGED, ORCID, ME, client=_client(), claimed=claimed)
    ids = {r.paper.openalex_id for r in records}
    # the three ecology works are unclaimed AND in a field no claimed work touches
    assert not ids & {"W1020", "W1021", "W1022"}
    assert report["dropped_unclaimed_field"] == 3
    assert report["claimed_matched"] == 2
    # unclaimed works in the *same* field as the claimed ones survive (own older papers)
    assert "W1003" in ids and "W1008" in ids
    assert next(r for r in records if r.paper.openalex_id == "W1001").claimed is True
    assert next(r for r in records if r.paper.openalex_id == "W1003").claimed is False


def test_build_records_without_registry_keeps_everything():
    records, report = oi.build_records(MERGED, ORCID, ME, client=_client(), claimed=None)
    assert {"W1020", "W1021", "W1022"} <= {r.paper.openalex_id for r in records}
    assert all(r.claimed is None for r in records)
    # an empty ORCID record is "no evidence" too, not "nothing is mine"
    records, _ = oi.build_records(MERGED, ORCID, ME, client=_client(), claimed=oi.ClaimedWorks())
    assert {"W1020", "W1021", "W1022"} <= {r.paper.openalex_id for r in records}


def test_select_seeds_prefers_claimed_works():
    claimed = oi.ClaimedWorks(dois={"10.1000/w1"}, n_works=1)
    # claimed set only has a *biomedical* DOI; ecology works share the field? no -> they are dropped.
    # Use a corpus where an unclaimed newer work survives, to see ordering.
    records, _ = oi.build_records(CORPUS, ORCID, ME, client=_client(), claimed=claimed)
    seeds, _ = oi.select_seeds(records, max_seeds=40)
    assert [r.paper.openalex_id for r in seeds][0] == "W1001"  # claimed 2015 paper outranks unclaimed 2023


def test_topics_default_on_only_when_a_seed_carries_them():
    records, _ = oi.build_records(MERGED, ORCID, ME, client=_client())
    seeds, _ = oi.select_seeds(records, max_seeds=2)  # newest lead works: W1022 (ecology 2024), W1008
    tf = oi.researcher_topic_filters(records, top_k=15, seeds=seeds)
    by_id = {t["id"]: t for t in tf["topics"]}
    assert by_id["T11289"]["on"] is True and by_id["T11289"]["seed_papers"] == 1
    assert by_id["T11937"]["on"] is True
    # T10222 is carried only by older / non-seed works -> listed but off
    assert by_id["T10222"]["on"] is False and by_id["T10222"]["seed_papers"] == 0
    # legacy behaviour when no seed list is supplied
    assert all(t["on"] for t in oi.researcher_topic_filters(records, top_k=15)["topics"])


def test_fetch_orcid_claimed_parses_registry_payload(monkeypatch):
    payload = {"group": [
        {"external-ids": {"external-id": [{"external-id-type": "doi", "external-id-value": "https://doi.org/10.1000/W1"}]},
         "work-summary": [{"title": {"title": {"value": "LOLA: enrichment analysis"}}}]},
        {"external-ids": {"external-id": [{"external-id-type": "eid", "external-id-value": "2-s2.0-1"}]},
         "work-summary": [{"title": {"title": {"value": "A conference talk about region set enrichment without a DOI"}}}]},
    ]}

    class _Resp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return payload

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())
    c = oi.fetch_orcid_claimed(ORCID)
    assert c.n_works == 2 and c.dois == {"10.1000/w1"}
    assert oi.title_key("A conference talk about region set enrichment without a DOI") in c.title_keys

    class _Down:
        status_code = 503
        def raise_for_status(self): raise RuntimeError("503")
        def json(self): return {}
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Down())
    assert oi.fetch_orcid_claimed(ORCID) is None


# --------------------------------------------------------------------------- seed picker helpers


def test_seed_defaults_reports_duplicate_copies():
    records, _ = oi.build_records(CORPUS, ORCID, ME, client=_client())
    seeds, dup_of, warnings = oi.seed_defaults(records, 40)
    assert [r.paper.openalex_id for r in seeds] == ["W1008", "W1002", "W1001"]
    assert dup_of == {"W1005": "W1001"}  # the preprint copy points at the kept version
    rows = oi.work_rows(records, defaults=seeds, dup_of=dup_of)
    by = {r["openalex_id"]: r for r in rows}
    assert by["W1005"]["dup_of"] == "W1001" and by["W1005"]["default_selected"] is False
    assert by["W1004"]["seed_eligible"] is False and by["W1003"]["default_selected"] is False
    assert all(r["selected"] is None for r in rows)


def test_store_works_then_load_records_round_trip(tmp_path):
    conn = connect(tmp_path / "t.db")
    apply_migrations(conn)
    user = users_repo.upsert(conn, "a@b.org")
    pid = profiles_repo.create_draft(conn, user_id=int(user["id"]), name="x", slug="x",
                                     embedding_model="placeholder-v1")
    records, _ = oi.build_records(CORPUS, ORCID, ME, client=_client())
    seeds, dup_of, _ = oi.seed_defaults(records, 40)
    assert oi.store_works(conn, profile_id=pid, records=records, defaults=seeds, dup_of=dup_of) == 6
    back = oi.load_work_records(conn, pid)
    assert {r.paper.openalex_id for r in back} == {r.paper.openalex_id for r in records}
    b = {r.paper.openalex_id: r for r in back}
    assert b["W1008"].is_lead and b["W1008"].position == "middle" and b["W1008"].is_corresponding
    assert b["W1003"].is_lead is False and b["W1004"].seed_eligible is False
    assert b["W1001"].paper.primary_topic is not None and b["W1001"].authors[0] == "Nathan C. Sheffield"
    assert oi.works_for_draft(conn, pid)[0]["openalex_id"] == "W1008"  # newest first
