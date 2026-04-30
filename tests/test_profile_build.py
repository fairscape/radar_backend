"""Profile.from_csv / Profile.from_pdfs tests — no network.

build_from_* used to live in rag_lib/profile_builder.py; it now lives on
Profile itself as classmethods. These tests exercise the full build
pipeline against a FakeOpenAlexClient (tests/fake_openalex_client.py):

  - CSV is parsed
  - DOI lookup happens for rows with a DOI
  - Title fallback happens when DOI lookup misses
  - Papers that OpenAlex doesn't know about still land as fallback records
  - local_path on the CSV is preserved on the Paper
  - Embeddings are generated and deterministic
  - topic_filters aggregate across seed papers at every hierarchy level
  - Profile JSON round-trips
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_lib.paper import Paper
from rag_lib.profile import Profile

from tests.fake_openalex_client import FakeOpenAlexClient, canned_openalex_work


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    import csv
    path = tmp_path / "manifest.csv"
    cols = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


# ----------------------------------------------------------------------
# Profile.from_csv
# ----------------------------------------------------------------------


def test_from_csv_resolves_by_doi(tmp_path):
    client = FakeOpenAlexClient(canned_works={
        "10.1/a": canned_openalex_work(
            doi="10.1/a", openalex_id="W1", title="Paper A", year=2024,
            abstract_words=["alpha", "beta", "gamma"],
            primary_topic_id="T1",
        ),
    })
    csv_path = _write_csv(tmp_path, [
        {"doi": "10.1/a", "path": "/tmp/a.pdf"},
    ])
    profile = Profile.from_csv(csv_path, name="alpha", openalex_client=client)

    assert isinstance(profile, Profile)
    assert profile.name == "alpha"
    assert len(profile.papers) == 1
    p = profile.papers[0]
    assert p.doi == "10.1/a"
    assert p.openalex_id == "W1"
    assert p.title == "Paper A"
    assert "alpha" in p.abstract
    assert p.local_path == "/tmp/a.pdf"
    assert p.source == "openalex"
    assert p.primary_topic is not None
    assert p.primary_topic.id == "T1"
    # Embedding present, default key + default dim.
    assert "placeholder-v1" in p.embeddings
    assert len(p.embeddings["placeholder-v1"]) == 128


def test_from_csv_title_fallback_on_doi_miss(tmp_path):
    client = FakeOpenAlexClient(canned_works={
        "10.2/b": canned_openalex_work(
            doi="10.2/b", openalex_id="W2",
            title="Something about neonatal HRV", year=2020,
        ),
    })
    csv_path = _write_csv(tmp_path, [
        {"doi": "10.unknown/missing", "title": "neonatal HRV",
         "year": "2020", "path": "/tmp/b.pdf"},
    ])
    profile = Profile.from_csv(csv_path, name="fallback", openalex_client=client)
    p = profile.papers[0]
    assert p.openalex_id == "W2"
    assert p.local_path == "/tmp/b.pdf"
    assert client.lookups_by_doi == ["10.unknown/missing"]
    assert len(client.lookups_by_title) == 1


def test_from_csv_unresolved_papers_still_land(tmp_path):
    """A paywalled or not-indexed paper must still produce a Paper record
    (source='user_csv') so the seed corpus is complete."""
    client = FakeOpenAlexClient(canned_works={})
    csv_path = _write_csv(tmp_path, [
        {"doi": "10.pay/wall", "title": "Obscure", "path": "/tmp/x.pdf",
         "abstract": "user-pasted abstract"},
    ])
    profile = Profile.from_csv(csv_path, name="mixed", openalex_client=client)
    assert len(profile.papers) == 1
    p = profile.papers[0]
    assert p.source == "user_csv"
    assert p.doi == "10.pay/wall"
    assert p.title == "Obscure"
    assert p.abstract == "user-pasted abstract"
    assert p.local_path == "/tmp/x.pdf"
    assert p.primary_topic is None
    # Embedding still generated from title+abstract.
    assert "placeholder-v1" in p.embeddings


def test_from_csv_aggregates_topic_filters(tmp_path):
    client = FakeOpenAlexClient(canned_works={
        "10.1/a": canned_openalex_work(
            doi="10.1/a", openalex_id="W1", title="Paper A",
            primary_topic_id="T1", subfield_id="SF1",
            field_id="F1", domain_id="D1",
        ),
        "10.1/b": canned_openalex_work(
            doi="10.1/b", openalex_id="W2", title="Paper B",
            primary_topic_id="T1", subfield_id="SF1",
            field_id="F1", domain_id="D1",
        ),
        "10.1/c": canned_openalex_work(
            doi="10.1/c", openalex_id="W3", title="Paper C",
            primary_topic_id="T2", subfield_id="SF2",
            field_id="F1", domain_id="D1",
        ),
    })
    csv_path = _write_csv(tmp_path, [
        {"doi": "10.1/a"}, {"doi": "10.1/b"}, {"doi": "10.1/c"},
    ])
    profile = Profile.from_csv(csv_path, name="agg", openalex_client=client)
    f = profile.topic_filters
    topic_ids = {t["id"] for t in f["topics"]}
    assert {"T1", "T2"} <= topic_ids
    # T1 appears in 2 papers (primary + topics[] each), so count >= 4.
    t1 = next(t for t in f["topics"] if t["id"] == "T1")
    assert t1["count"] >= 4
    # Field F1 shared by all 3 papers -> count >= 6 (primary + topics[]).
    f1 = next(fl for fl in f["fields"] if fl["id"] == "F1")
    assert f1["count"] >= 6


def test_from_csv_profile_json_round_trips(tmp_path):
    client = FakeOpenAlexClient(canned_works={
        "10.1/a": canned_openalex_work(
            doi="10.1/a", openalex_id="W1", title="A"),
    })
    csv_path = _write_csv(tmp_path, [{"doi": "10.1/a", "path": "/tmp/a.pdf"}])
    profile = Profile.from_csv(csv_path, name="rt", openalex_client=client)
    out = tmp_path / "profile.json"
    profile.to_json(out)
    reloaded = Profile.from_json(out)
    assert reloaded == profile


def test_from_csv_honors_custom_embedder_and_model_name(tmp_path):
    """Callers can swap in a different embedder (e.g., specter2_embed in
    Phase 1B) and the key on Paper.embeddings reflects the chosen
    embedding_model name."""
    client = FakeOpenAlexClient(canned_works={
        "10.1/a": canned_openalex_work(doi="10.1/a", openalex_id="W1", title="A"),
    })
    csv_path = _write_csv(tmp_path, [{"doi": "10.1/a"}])

    def constant_embed(text: str) -> list[float]:
        return [0.25] * 4

    profile = Profile.from_csv(
        csv_path, name="custom", openalex_client=client,
        embedder=constant_embed, embedding_model="constant-v1",
    )
    p = profile.papers[0]
    assert profile.embedding_model == "constant-v1"
    assert p.embeddings == {"constant-v1": [0.25, 0.25, 0.25, 0.25]}
    # Selectors read via Profile.seed_embeddings using embedding_model key.
    assert profile.seed_embeddings() == [[0.25, 0.25, 0.25, 0.25]]


# ----------------------------------------------------------------------
# Profile.from_pdfs
# ----------------------------------------------------------------------


def test_from_pdfs_empty_dir_returns_empty_profile(tmp_path):
    """No PDFs -> empty but valid Profile. Does not require pdfplumber
    since no PDFs means no ingest_pdf calls."""
    client = FakeOpenAlexClient()
    profile = Profile.from_pdfs(tmp_path, name="empty", openalex_client=client)
    assert isinstance(profile, Profile)
    assert profile.name == "empty"
    assert profile.papers == []
    assert profile.topic_filters == {"topics": [], "subfields": [],
                                     "fields": [], "domains": []}


def test_from_pdfs_surfaces_pdfplumber_missing_error(tmp_path, monkeypatch):
    """If pdfplumber isn't installed and the dir has a PDF, ingest_pdf
    raises ImportError with a helpful message."""
    # Put a dummy file with .pdf extension so the glob picks it up.
    (tmp_path / "fake.pdf").write_bytes(b"not really a pdf")

    # Simulate pdfplumber being unavailable by patching ingest_pdf to
    # raise the ImportError that the real code would raise.
    import rag_lib.vault as vault
    def fake_ingest(path):
        raise ImportError(
            "ingest_pdf requires the phase1b extras. "
            "Install with: pip install -e '.[dev,phase1b]'"
        )
    monkeypatch.setattr(vault, "ingest_pdf", fake_ingest)

    client = FakeOpenAlexClient()
    with pytest.raises(ImportError, match="phase1b"):
        Profile.from_pdfs(tmp_path, name="x", openalex_client=client)


def test_from_pdfs_uses_ingest_pdf_then_openalex(tmp_path, monkeypatch):
    """End-to-end: ingest_pdf is mocked to return a fixed record; OpenAlex
    lookup (fake) resolves; Paper is built with body_text + local_path."""
    from rag_lib.vault import PdfIngestRecord

    (tmp_path / "a.pdf").write_bytes(b"x")

    def fake_ingest(path):
        return PdfIngestRecord(
            path=str(path),
            title="Paper A",
            body_text="Full body text of paper A.",
            doi="10.1/a",
            n_pages=3,
        )

    import rag_lib.vault as vault
    monkeypatch.setattr(vault, "ingest_pdf", fake_ingest)

    client = FakeOpenAlexClient(canned_works={
        "10.1/a": canned_openalex_work(
            doi="10.1/a", openalex_id="W1", title="Paper A",
            abstract_words=["alpha", "beta"]),
    })
    profile = Profile.from_pdfs(tmp_path, name="pdf_test", openalex_client=client)
    assert len(profile.papers) == 1
    p = profile.papers[0]
    assert p.openalex_id == "W1"
    assert p.doi == "10.1/a"
    assert p.body_text == "Full body text of paper A."
    assert p.local_path == str(tmp_path / "a.pdf")
    assert "placeholder-v1" in p.embeddings
