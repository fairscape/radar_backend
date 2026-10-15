"""A seed's UMLS results are dropped when the text they came from changes,
so the next extraction pass reads the better text."""

from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import papers as papers_repo


def _umls(conn, oa):
    return tuple(conn.execute(
        "SELECT umls_concepts_json, umls_mapped_topics_json FROM papers WHERE openalex_id = ?", (oa,)).fetchone())


def test_body_text_for_a_paper_without_abstract_sends_it_back_to_extraction(tmp_path):
    conn = connect(tmp_path / "r.db")
    apply_migrations(conn)
    papers_repo.upsert(conn, {"openalex_id": "W1", "title": "Insulin pumps", "source": "orcid"})
    conn.execute("UPDATE papers SET umls_concepts_json = '[]', umls_mapped_topics_json = NULL")
    conn.commit()

    # The same record again (no new text): kept.
    papers_repo.upsert(conn, {"openalex_id": "W1", "title": "Insulin pumps", "source": "orcid"})
    assert _umls(conn, "W1") == ("[]", None)

    # Its PDF arrives: body text is now what extraction would read.
    papers_repo.upsert(conn, {"openalex_id": "W1", "title": "Insulin pumps", "source": "user_pdf",
                              "body_text": "Closed-loop insulin delivery in type 1 diabetes ..."})
    assert _umls(conn, "W1") == (None, None)


def test_body_text_does_not_matter_when_the_abstract_is_read(tmp_path):
    conn = connect(tmp_path / "r.db")
    apply_migrations(conn)
    papers_repo.upsert(conn, {"openalex_id": "W2", "title": "T", "abstract": "An abstract.", "source": "orcid"})
    conn.execute("UPDATE papers SET umls_concepts_json = '[{\"cui\": \"C1\"}]', umls_mapped_topics_json = '[]'")
    conn.commit()
    papers_repo.upsert(conn, {"openalex_id": "W2", "title": "T", "source": "user_pdf", "body_text": "Body."})
    assert _umls(conn, "W2")[0] is not None
    # A new abstract does.
    papers_repo.upsert(conn, {"openalex_id": "W2", "title": "T", "abstract": "A longer, better abstract.", "source": "orcid"})
    assert _umls(conn, "W2") == (None, None)


def test_a_gather_refetching_a_seed_keeps_its_pdf_text_and_imported_abstract(tmp_path):
    """A gather passes '' where OpenAlex has no abstract, and body text is
    always ''; COALESCE kept '' over the stored text."""
    conn = connect(tmp_path / "r.db")
    apply_migrations(conn)
    papers_repo.upsert(conn, {"openalex_id": "W3", "title": "T", "abstract": "From Prosopia.",
                              "body_text": "PDF body " * 100, "source": "user_pdf"})
    conn.execute("UPDATE papers SET umls_concepts_json = '[]'")
    conn.commit()
    papers_repo.upsert(conn, {"openalex_id": "W3", "title": "T", "abstract": "", "body_text": "",
                              "source": "openalex_gatherer"})
    abstract, body, umls = conn.execute(
        "SELECT abstract, body_text, umls_concepts_json FROM papers WHERE openalex_id = 'W3'").fetchone()
    assert abstract == "From Prosopia." and body.startswith("PDF body") and umls == "[]"
