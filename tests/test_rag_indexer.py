"""Indexer tests: pure-Python chunker + ``index_paper`` write semantics.

Chroma is a phase1b extra; we use a tiny in-process fake collection
that records ``add`` calls so the indexer logic can be exercised
without the dependency.
"""

from __future__ import annotations

import pytest

from rag_lib.rag.indexer import (
    chunk_text,
    count_chunks_for_paper,
    index_paper,
)


# -----------------------------------------------------------------------------
# Fakes
# -----------------------------------------------------------------------------


class FakeCollection:
    """In-process stand-in for a Chroma collection.

    Only the surface ``index_paper`` and ``count_chunks_for_paper`` use:
    ``add`` records the inputs, ``get`` filters by ``openalex_id``.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.records: list[dict] = []

    def add(self, *, ids, embeddings, metadatas, documents):  # noqa: D401
        self.calls.append({
            "ids": list(ids),
            "embeddings": [list(v) for v in embeddings],
            "metadatas": [dict(m) for m in metadatas],
            "documents": list(documents),
        })
        for i, md, d in zip(ids, metadatas, documents):
            self.records.append({"id": i, "metadata": dict(md), "document": d})

    def get(self, *, where=None):
        if where is None:
            matched = list(self.records)
        else:
            matched = [r for r in self.records if _matches(r["metadata"], where)]
        return {
            "ids": [r["id"] for r in matched],
            "documents": [r["document"] for r in matched],
            "metadatas": [r["metadata"] for r in matched],
        }


def _matches(metadata: dict, where: dict) -> bool:
    """Lightweight Chroma-style filter eval (only the operators we use)."""
    for k, v in where.items():
        if k == "$or":
            if not any(_matches(metadata, sub) for sub in v):
                return False
        elif isinstance(v, dict):
            field = metadata.get(k, "")
            for op, val in v.items():
                if op == "$contains" and val not in str(field):
                    return False
                if op == "$eq" and field != val:
                    return False
        else:
            if metadata.get(k) != v:
                return False
    return True


def _embedder(_text: str) -> list[float]:
    return [0.1, 0.2, 0.3]


# -----------------------------------------------------------------------------
# chunk_text
# -----------------------------------------------------------------------------


def test_chunk_text_short_input_returns_one_chunk():
    chunks = chunk_text("hello world", target_tokens=500, overlap=50)
    assert chunks == ["hello world"]


def test_chunk_text_empty_returns_empty():
    assert chunk_text("", target_tokens=500, overlap=50) == []
    assert chunk_text("   \n\n  ", target_tokens=500, overlap=50) == []


def test_chunk_text_overlap_is_respected():
    words = " ".join(f"w{i}" for i in range(120))
    chunks = chunk_text(words, target_tokens=50, overlap=10)
    # 120 words with step=40 ⇒ windows starting at 0, 40, 80; each ≤50.
    assert len(chunks) == 3
    assert chunks[0].split()[0] == "w0"
    assert chunks[0].split()[-1] == "w49"
    # Overlap means chunk[1] starts before chunk[0] ends.
    assert chunks[1].split()[0] == "w40"
    # Final chunk reaches the end exactly.
    assert chunks[-1].split()[-1] == "w119"


def test_chunk_text_no_runaway_when_overlap_zero():
    words = " ".join(f"w{i}" for i in range(150))
    chunks = chunk_text(words, target_tokens=50, overlap=0)
    assert len(chunks) == 3
    assert chunks[0].split()[0] == "w0"
    assert chunks[1].split()[0] == "w50"
    assert chunks[2].split()[0] == "w100"


def test_chunk_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("a b c", target_tokens=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("a b c", target_tokens=10, overlap=-1)
    with pytest.raises(ValueError):
        chunk_text("a b c", target_tokens=0, overlap=0)


# -----------------------------------------------------------------------------
# index_paper
# -----------------------------------------------------------------------------


def test_index_paper_writes_chunks_to_collection():
    body = " ".join(f"tok{i}" for i in range(120))
    paper = {
        "openalex_id": "W_PAPER_1",
        "title": "On Neonatal Vitals",
        "body_text": body,
    }
    coll = FakeCollection()
    written = index_paper(
        coll, paper, _embedder,
        profile_slugs=["neonatal-vitals"],
        target_tokens=50,
        overlap=10,
    )
    assert written == 3
    assert len(coll.calls) == 1
    call = coll.calls[0]
    assert call["ids"] == [
        "W_PAPER_1::chunk::0000",
        "W_PAPER_1::chunk::0001",
        "W_PAPER_1::chunk::0002",
    ]
    # Metadata carries the slug envelope so retriever's $contains works.
    assert all(
        md["profile_slugs"] == "|neonatal-vitals|"
        and md["title"] == "On Neonatal Vitals"
        and md["openalex_id"] == "W_PAPER_1"
        for md in call["metadatas"]
    )


def test_index_paper_zero_chunks_for_empty_body():
    paper = {"openalex_id": "W_X", "title": "T", "body_text": "   "}
    coll = FakeCollection()
    assert index_paper(coll, paper, _embedder) == 0
    assert coll.calls == []


def test_index_paper_with_multiple_slugs_encodes_all():
    paper = {
        "openalex_id": "W_M",
        "title": "Multi",
        "body_text": "a b c d e",
    }
    coll = FakeCollection()
    index_paper(coll, paper, _embedder, profile_slugs=["alpha", "beta"])
    md = coll.calls[0]["metadatas"][0]
    assert md["profile_slugs"] == "|alpha|beta|"


def test_count_chunks_for_paper_filters_by_openalex_id():
    coll = FakeCollection()
    index_paper(
        coll,
        {"openalex_id": "W_A", "title": "A", "body_text": "a b c"},
        _embedder,
    )
    index_paper(
        coll,
        {"openalex_id": "W_B", "title": "B", "body_text": "x y z"},
        _embedder,
    )
    assert count_chunks_for_paper(coll, "W_A") == 1
    assert count_chunks_for_paper(coll, "W_B") == 1
    assert count_chunks_for_paper(coll, "W_MISSING") == 0
