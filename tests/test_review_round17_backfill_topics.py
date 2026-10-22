"""Review 17: a backfill credits each paper to a topic, as a scan does."""

from __future__ import annotations

from rag_lib.paper import Paper
from rag_lib.profile import Profile
from tests.test_year_range import _gatherer


class _Client:
    api_calls = 0

    def paginate_filter(self, filter_str, *, limit=None, per_page=200, sort=None):
        self.api_calls += 1
        return [
            # Best-matching topic first; T9 is not one of the interest's.
            {"id": "https://openalex.org/W1", "title": "a", "topics": [{"id": "https://openalex.org/T9"}, {"id": "https://openalex.org/T2"}]},
            {"id": "https://openalex.org/W2", "title": "b", "topics": [{"id": "https://openalex.org/T1"}]},
            {"id": "https://openalex.org/W3", "title": "c", "topics": [{"id": "https://openalex.org/T3"}]},
        ]

    def paper_from_work(self, w, source=None):
        return Paper(doi=None, openalex_id=w["id"], title=w["title"], year=2020)


def test_backfilled_papers_are_credited_to_the_first_enabled_topic():
    profile = Profile(name="p", topic_filters={"topics": [
        {"id": "https://openalex.org/T1", "on": True}, {"id": "https://openalex.org/T2", "on": True},
        {"id": "https://openalex.org/T3", "on": False}]})
    g = _gatherer(_Client())
    papers = g.fetch_years(profile, 2020, 2020, per_year=10)
    assert len(papers) == 3
    assert g.last_source_topics == {
        "https://openalex.org/W1": "https://openalex.org/T2",
        "https://openalex.org/W2": "https://openalex.org/T1",
    }
