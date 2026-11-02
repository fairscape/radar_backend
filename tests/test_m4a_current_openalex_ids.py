"""Prosopia's OpenAlex ids are read as OpenAlex has them today
(FUNCTIONAL_SPEC M.4 rule 6, M-34 to M-36).

OpenAlex merges duplicate works; a merged-away id still answers, with the
work it went into. Prosopia keeps the id it was given. Turned into the
current id where the papers come in -- the listing and the import -- so
"already a seed" and every comparison after see one id per paper.
"""

from __future__ import annotations

from rag_lib.db import connect
from rag_lib.openalex_status import OpenAlexUnavailable
from rag_lib.prosopia_seeds import current_works, prefetch_works
from tests.test_prosopia_import import (
    RECORD_DOI_ONLY,
    RECORD_FULL,
    StubOpenAlex,
    StubProfile,
    StubProsopia,
    StubSummaries,
    _work,
)
from tests.test_researchers import _app, _request, env  # noqa: F401 -- fixture

OLD = "W4415881950"          # RECORD_FULL's id, as Prosopia has it
NEW = "W4999999999"          # the work OpenAlex merged it into
MERGED = _work("10.1101/2025.11.03.685753", NEW, "Atacformer: a foundation model for ATAC-seq", 2025)
KEPT = _work("10.1101/2023.09.19.558267", "W4386953349",
             "Protein Kinase A Inhibition Epigenetically Silences Ren1", 2023)


def _openalex():
    """OpenAlex after the merge: the old id answers with the new work."""
    return StubOpenAlex(by_id={OLD: MERGED, NEW: MERGED, "W4386953349": KEPT},
                        by_doi={"10.1101/2025.11.03.685753": MERGED, "10.1101/2023.09.19.558267": KEPT})


def _record_with_id(record, oa_id):
    return record.model_copy(update={"openalex_id": oa_id})


def _prosopia():
    records = [RECORD_FULL, _record_with_id(RECORD_DOI_ONLY, "W4386953349")]
    return StubProsopia(StubProfile("Nathan C. Sheffield", records, StubSummaries({})))


# --- the lookup ---------------------------------------------------------------------------


def test_current_works_one_batch_and_a_lookup_only_for_the_merged_id():
    client = _openalex()
    found, absent = current_works([OLD, "https://openalex.org/W4386953349", "W1", None, OLD], client)
    assert {k: v["id"].rsplit("/", 1)[-1] for k, v in found.items()} == {OLD: NEW, "W4386953349": "W4386953349"}
    assert absent == {"W1"}
    assert client.calls == [
        ("paginate_filter", f"ids.openalex:{OLD}|W4386953349|W1"),
        ("get_work", OLD),
        ("get_work", "W1"),
    ]


def test_current_works_chunks_at_fifty():
    client = StubOpenAlex(by_id={f"W{i}": _work(f"10.1/{i}", f"W{i}", f"Paper {i}") for i in range(120)})
    found, absent = current_works([f"W{i}" for i in range(120)], client)
    assert len(found) == 120 and absent == set()
    assert [c[0] for c in client.calls] == ["paginate_filter"] * 3


def test_current_works_lets_openalex_unavailable_through():
    class Down(StubOpenAlex):
        def paginate_filter(self, *a, **k):
            raise OpenAlexUnavailable("down")

    try:
        current_works([OLD], Down())
    except OpenAlexUnavailable:
        return
    raise AssertionError("OpenAlexUnavailable was swallowed")


# --- M-34: the listing ----------------------------------------------------------------------


def test_m34_the_listing_carries_the_current_id(env):
    app = _app(_prosopia(), _openalex())
    r = _request(app, "GET", "/api/import/prosopia/works", params={"ref": "sheffield-nathan"})
    assert r.status_code == 200, r.text
    ids = {w["id"]: w["openalex_id"] for w in r.json()["works"]}
    assert ids["leroy2025atacformer"] == f"https://openalex.org/{NEW}"
    assert ids["smith2023protein"] == "https://openalex.org/W4386953349"


def test_m36_the_listing_shows_prosopias_ids_while_openalex_is_unavailable(env):
    class Down(StubOpenAlex):
        def paginate_filter(self, *a, **k):
            raise OpenAlexUnavailable("down")

    app = _app(_prosopia(), Down())
    r = _request(app, "GET", "/api/import/prosopia/works", params={"ref": "sheffield-nathan"})
    assert r.status_code == 200, r.text
    assert {w["id"]: w["openalex_id"] for w in r.json()["works"]}["leroy2025atacformer"] == OLD


def test_m36_a_test_without_an_openalex_stub_never_reaches_openalex(env):
    """The real client in a test: unreachable (conftest), so the listing
    keeps Prosopia's ids instead of asking api.openalex.org."""
    app = _app(_prosopia())
    r = _request(app, "GET", "/api/import/prosopia/works", params={"ref": "sheffield-nathan"})
    assert r.status_code == 200, r.text
    assert {w["id"]: w["openalex_id"] for w in r.json()["works"]}["leroy2025atacformer"] == OLD


# --- M-35: the import ------------------------------------------------------------------------


def test_m35_prefetch_reads_ids_in_a_batch_and_dois_only_for_the_rest():
    client = _openalex()
    records = [RECORD_FULL, RECORD_DOI_ONLY]
    cache = prefetch_works(records, client)
    assert cache.by_id[OLD]["id"].endswith(NEW)
    assert client.calls == [
        ("paginate_filter", f"ids.openalex:{OLD}"),
        ("get_work", OLD),
        ("paginate_filter", "doi:10.1101/2023.09.19.558267"),
    ]


def test_m35_the_import_seeds_the_current_id_without_a_lookup_per_paper(env):
    oa = _openalex()
    app = _app(_prosopia(), oa)
    r = _request(app, "POST", "/api/import/prosopia", json={"ref": "sheffield-nathan"})
    assert r.status_code == 200, r.text
    conn = connect(env)
    try:
        seeds = sorted(row[0] for row in conn.execute("SELECT openalex_id FROM profile_seeds"))
    finally:
        conn.close()
    assert seeds == ["https://openalex.org/W4386953349", f"https://openalex.org/{NEW}"]
    # One id batch; a lookup only for the merged id; nothing per paper after.
    assert oa.calls == [("paginate_filter", f"ids.openalex:{OLD}|W4386953349"), ("get_work", OLD)]


def test_m34_a_merged_id_already_a_seed_is_listed_under_that_seeds_id(env):
    """What the picker compares with ("already a seed"): the seed and the
    listing name the work alike."""
    oa = _openalex()
    app = _app(_prosopia(), oa)
    slug = _request(app, "POST", "/api/import/prosopia",
                    json={"ref": "sheffield-nathan", "paper_ids": ["leroy2025atacformer"]}).json()["draft_slug"]
    seed_ids = {d["id"] for d in _request(app, "GET", "/api/vault/docs", params={"tag": slug}).json()}
    listed = {w["id"]: w["openalex_id"] for w in
              _request(app, "GET", "/api/import/prosopia/works", params={"ref": "sheffield-nathan"}).json()["works"]}
    assert listed["leroy2025atacformer"] in seed_ids


def test_m35_an_id_openalex_does_not_know_is_asked_once_then_the_doi_finds_it():
    from rag_lib.prosopia_seeds import resolve_seed

    client = StubOpenAlex(by_doi={"10.1101/2023.09.19.558267": KEPT})
    record = _record_with_id(RECORD_DOI_ONLY, "W1")
    cache = prefetch_works([record], client)
    res = resolve_seed(record, client, slug="s", cache=cache)
    assert res.openalex_id == "https://openalex.org/W4386953349"
    assert client.calls == [
        ("paginate_filter", "ids.openalex:W1"),
        ("get_work", "W1"),
        ("paginate_filter", "doi:10.1101/2023.09.19.558267"),
    ]
