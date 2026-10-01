

def test_paper_from_work_keeps_the_byline_and_the_date():
    from rag_lib.openalex_client import OpenAlexClient

    work = {
        "id": "https://openalex.org/W1", "title": "T", "publication_year": 2026,
        "publication_date": "2026-04-15",
        "authorships": [{"author": {"display_name": "Ada Lovelace"}},
                        {"author": {"display_name": None}},
                        {"author": {"display_name": "Alan Turing"}}],
    }
    p = OpenAlexClient(mailto="t@example.com").paper_from_work(work)
    assert p.authors == ["Ada Lovelace", "Alan Turing"]
    assert p.publication_date == "2026-04-15"
