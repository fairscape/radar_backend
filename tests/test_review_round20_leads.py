"""Round 20: the leads round 19 left unverified, each held by a test.

- Zenodo concept/version DOIs of one short-titled record are one card,
  within a scan and across scans (``paper.paper_keys``).
- A publication year that cannot be right yet is stored as unknown
  (``papers_repo.plausible_year``), so a feed "to now" does not lead with it.
- ORCID works are read newest first, and a cut list says so.
"""

from __future__ import annotations

import datetime as dt

from rag_lib.db.repos import papers as papers_repo
from rag_lib.db.repos.papers import plausible_year
from rag_lib.paper import Paper, paper_keys
from rag_lib.persistence import dedup_and_insert_candidates
from tests.test_review_round19_b_numbers import _interest, _run, conn  # noqa: F401 -- fixture


def _z(oa, doi, title="The Trust Dial", year=2026, authors=("Ann Lee",)):
    return Paper(doi=doi, openalex_id=oa, title=title, abstract="a", year=year,
                 venue="Zenodo", source="openalex_gatherer", authors=list(authors))


def test_zenodo_versions_of_one_short_title_are_one_card(conn):
    uid = conn.execute("SELECT id FROM users LIMIT 1").fetchone()[0]
    pid = _interest(conn, uid)
    r1 = _run(conn, pid, uid)
    dedup_and_insert_candidates(conn, profile_id=pid, gather_run_id=r1, ranked=[
        (0.95, _z("https://openalex.org/W1", "10.5281/zenodo.22968515"), {"score_raw": 0.95}),
        (0.94, _z("https://openalex.org/W2", "10.5281/zenodo.22968516"), {"score_raw": 0.94}),
    ])
    r2 = _run(conn, pid, uid)
    dedup_and_insert_candidates(conn, profile_id=pid, gather_run_id=r2, ranked=[
        (0.95, _z("https://openalex.org/W3", "10.5281/zenodo.22968517", authors=("Lee, Ann",)), {"score_raw": 0.95}),
    ])
    ids = [r[0] for r in conn.execute("SELECT openalex_id FROM profile_candidates WHERE profile_id = ?", (pid,))]
    assert ids == ["https://openalex.org/W1"]


def test_other_short_titles_are_not_merged():
    one = paper_keys("10.1/x", "Editorial", 2026, ["Ann Lee"])
    two = paper_keys("10.1/y", "Editorial", 2026, ["Ann Lee"])
    assert not set(one) & set(two)
    # Zenodo, but another year or other people: another record.
    base = paper_keys("10.5281/zenodo.1", "The Trust Dial", 2026, ["Ann Lee"])
    assert not set(base) & set(paper_keys("10.5281/zenodo.2", "The Trust Dial", 2025, ["Ann Lee"]))
    assert not set(base) & set(paper_keys("10.5281/zenodo.2", "The Trust Dial", 2026, ["Bo Chen"]))


def test_an_implausible_year_is_stored_as_unknown(conn):
    nxt = dt.date.today().year + 1
    assert plausible_year(nxt) == nxt and plausible_year(nxt + 1) is None and plausible_year("x") is None
    papers_repo.upsert(conn, {"openalex_id": "W2050", "title": "t", "year": 2050, "source": "s"})
    assert conn.execute("SELECT year FROM papers WHERE openalex_id = 'W2050'").fetchone()[0] is None


def test_orcid_works_are_read_newest_first_and_a_cut_list_says_so():
    from rag_lib.api.services import orcid

    seen = {}

    class Client:
        def paginate_filter(self, f, *, limit=None, sort=None):
            seen["sort"] = sort
            return [{"id": f"https://openalex.org/W{i}", "title": f"Paper number {i} about things",
                     "publication_year": 2020, "authorships": []} for i in range(limit)]

    out = orcid.list_works("0000-0002-1825-0097", Client(), limit=3, check_registry=False)
    assert seen["sort"] == "publication_date:desc" and out["truncated"] is True
