"""An explicit OpenAlex id that fails to load must not become a title match.

The ORCID picker sends the ids the user ticked -- often the version of
record, with its preprint left unticked as a duplicate. A transient 429 on
get_work used to drop the record to the title rung, which picks whichever
copy ranks first: the excluded preprint got seeded instead.
"""

from __future__ import annotations

import pytest

from rag_lib import prosopia_seeds
from rag_lib.api.services.orcid import WorkRecord
from rag_lib.openalex_client import OpenAlexClient

VOR = {"id": "https://openalex.org/W1", "title": "Shared title of one paper", "publication_year": 2024}
PREPRINT = {"id": "https://openalex.org/W2", "title": "Shared title of one paper", "publication_year": 2024}


class Stub(OpenAlexClient):
    def __init__(self, get_work_results):
        super().__init__(mailto="t@example.com")
        self._results = list(get_work_results)
        self.title_searches = 0

    def get_work(self, openalex_id):
        r = self._results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def lookup_by_doi(self, doi):
        return None

    def lookup_by_pmcid(self, pmcid):
        return None

    def search_by_title(self, title, year=None, per_page=5):
        self.title_searches += 1
        return [PREPRINT]


RECORD = WorkRecord(paper_id="W1", name="Shared title of one paper", openalex_id="W1",
                    doi=None, year=2024, venue=None)


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(prosopia_seeds, "_ID_RETRY_DELAY", 0.0)


def _http_error(status: int):
    import requests

    resp = requests.Response()
    resp.status_code = status
    return requests.HTTPError(f"{status}", response=resp)


def test_a_server_error_is_retried_once():
    client = Stub([_http_error(503), VOR])
    res = prosopia_seeds.resolve_seed(RECORD, client)
    assert (res.rung, res.openalex_id) == ("work_id", "https://openalex.org/W1")


def test_a_rate_limit_is_not_retried():
    """Retrying a 429 burns quota and extends the cooldown."""
    client = Stub([RuntimeError("OpenAlex 429: rate-limited"), VOR])
    res = prosopia_seeds.resolve_seed(RECORD, client)
    assert len(client._results) == 1          # the second answer was never asked for
    assert res.rung == "none"


def test_an_unreachable_id_never_becomes_a_title_match():
    client = Stub([_http_error(503), _http_error(503)])
    res = prosopia_seeds.resolve_seed(RECORD, client)
    assert client.title_searches == 0
    assert res.rung == "none"
    assert res.openalex_id != "https://openalex.org/W2"


def test_an_id_openalex_does_not_know_may_still_be_found_by_title():
    """A 404 is not transient: the id is wrong, and a title is fair game."""
    client = Stub([None])
    res = prosopia_seeds.resolve_seed(RECORD, client)
    assert client.title_searches == 1
    assert res.rung == "title"
