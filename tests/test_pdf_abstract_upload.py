"""The abstract printed in an uploaded PDF (FUNCTIONAL_SPEC P.3-3, P.5 Q1).

When OpenAlex has no abstract for the paper, the one found in the PDF is
stored on the uploader's own version, marked ``abstract_source = 'pdf'``;
the public record never gets it. An OpenAlex abstract always wins. What
the paper's topics are read from follows: the abstract, not the body.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_lib.vault import PdfIngestRecord
from tests.test_private_upload_versions import A, B, _row, _upload, pdfs, world  # noqa: F401 -- fixtures
from tests.test_researchers import env  # noqa: F401 -- fixture

BARE = "https://openalex.org/W88"       # OpenAlex has the work, without an abstract
PDF_ABSTRACT = "The abstract as printed in the PDF, long enough to stand for the paper."


@pytest.fixture()
def uploads(pdfs, monkeypatch):
    """Bytes say what the PDF holds: ``abs`` an abstract it prints;
    ``matched`` OpenAlex W77 (with an abstract), ``bare`` W88 (without);
    anything else matches nothing."""
    from rag_lib.api.services import vault as vault_service

    def ingest(path):
        text = Path(path).read_bytes().decode("latin-1")
        return PdfIngestRecord(path=str(path), title="Title on the page", body_text=f"body of {text}",
                               doi=None, n_pages=2, abstract=PDF_ABSTRACT if "abs" in text else None)

    real_enrich = vault_service._enrich_via_openalex

    def enrich(record, *, mailto, filename, file_hash):
        if "bare" in record.body_text:
            return BARE, {"openalex_id": BARE, "title": "Bare Title", "abstract": "", "year": 2021,
                          "doi": "10.1/w88", "source": "user_pdf"}
        return real_enrich(record, mailto=mailto, filename=filename, file_hash=file_hash)

    from rag_lib import content_topics
    from rag_lib.pdf_text import paper_text

    read = pdfs["content_calls"]

    def topics(title, abstract, body, *, settings):     # what the real one embeds
        read.append(paper_text(abstract, body, content_topics.BODY_CHARS))
        return None

    monkeypatch.setattr(vault_service, "ingest_pdf", ingest)
    monkeypatch.setattr(vault_service, "_enrich_via_openalex", enrich)
    monkeypatch.setattr(vault_service.content_topics, "try_topics_json", topics)
    return pdfs


def test_unmatched_upload_keeps_the_pdf_abstract(uploads):
    u = _upload(uploads, b"nothing known abs", A)
    row = _row(uploads["env"], u["id"])
    assert row["abstract"] == PDF_ABSTRACT and row["abstract_source"] == "pdf"
    # what its topics are read from: the abstract, not the body
    assert uploads["content_calls"] == [PDF_ABSTRACT]


def test_matched_without_an_openalex_abstract_keeps_it_on_the_version_only(uploads):
    u = _upload(uploads, b"bare abs", A)
    row = _row(uploads["env"], u["id"])
    assert row["abstract"] == PDF_ABSTRACT and row["abstract_source"] == "pdf"
    public = _row(uploads["env"], BARE)
    assert not public["abstract"] and public["abstract_source"] is None
    # another user's version of the same work does not see it
    v = _upload(uploads, b"bare other file", B)
    assert not _row(uploads["env"], v["id"])["abstract"]


def test_an_openalex_abstract_wins(uploads):
    u = _upload(uploads, b"matched abs", A)
    row = _row(uploads["env"], u["id"])
    assert row["abstract"] == "public abstract" and row["abstract_source"] is None


def test_no_abstract_anywhere_stores_none(uploads):
    u = _upload(uploads, b"nothing known", A)
    row = _row(uploads["env"], u["id"])
    assert not row["abstract"] and row["abstract_source"] is None
    # topics then come from where the body's reading starts
    assert uploads["content_calls"] == [f"body of %PDF-1.4\nnothing known"]


def test_the_same_file_again_keeps_its_pdf_abstract(uploads):
    first = _upload(uploads, b"nothing known abs", A)
    again = _upload(uploads, b"nothing known abs", A)
    assert again["id"] == first["id"]
    row = _row(uploads["env"], first["id"])
    assert row["abstract"] == PDF_ABSTRACT and row["abstract_source"] == "pdf"


def test_the_vault_says_where_the_abstract_came_from(uploads):
    """P.5 Q2: the Vault page shows the abstract, the PDF's marked as such;
    OpenAlex's unmarked; none as none. Another user sees none of it."""
    from tests.test_private_upload_versions import _request

    pdf = _upload(uploads, b"nothing known abs", A)
    public = _upload(uploads, b"matched abs", A)
    bare = _upload(uploads, b"nothing known", A)
    assert (pdf["abstract"], pdf["abstractSource"]) == (PDF_ABSTRACT, "pdf")     # the upload's answer
    docs = {d["id"]: d for d in _request(uploads["app"], "GET", "/api/vault/docs", headers=A).json()}
    assert (docs[pdf["id"]]["abstract"], docs[pdf["id"]]["abstractSource"]) == (PDF_ABSTRACT, "pdf")
    assert (docs[public["id"]]["abstract"], docs[public["id"]]["abstractSource"]) == ("public abstract", None)
    assert (docs[bare["id"]]["abstract"], docs[bare["id"]]["abstractSource"]) == (None, None)
    theirs = _request(uploads["app"], "GET", "/api/vault/docs", headers=B).json()
    assert all(d.get("abstract") != PDF_ABSTRACT for d in theirs)
