"""Retriever tests: scope filtering and result shaping.

A FakeCollection emulates Chroma's ``query`` semantics for the
operators the retriever relies on (``$contains``, ``$or``).
"""

from __future__ import annotations

import pytest

from rag_lib.rag.retriever import retrieve


class FakeCollection:
    def __init__(self, records: list[dict]) -> None:
        self.records = list(records)
        self.last_query: dict | None = None

    def query(self, **kwargs):
        self.last_query = dict(kwargs)
        n = kwargs.get("n_results", 10)
        where = kwargs.get("where")
        candidates = [r for r in self.records if _matches(r["metadata"], where)]
        # Pretend each record carries a fixed cosine distance so we can
        # assert score conversion.
        candidates = candidates[:n]
        return {
            "documents": [[c["document"] for c in candidates]],
            "metadatas": [[c["metadata"] for c in candidates]],
            "distances": [[c.get("distance", 0.2) for c in candidates]],
        }


def _matches(metadata: dict, where: dict | None) -> bool:
    if not where:
        return True
    for k, v in where.items():
        if k == "$or":
            if not any(_matches(metadata, sub) for sub in v):
                return False
        elif isinstance(v, dict):
            field = metadata.get(k, "")
            for op, val in v.items():
                if op == "$contains" and val not in str(field):
                    return False
        else:
            if metadata.get(k) != v:
                return False
    return True


def _embedder(_text: str) -> list[float]:
    return [0.1, 0.2, 0.3]


def _record(doc_id, slugs, distance=0.2, title="T"):
    return {
        "id": doc_id,
        "document": f"text for {doc_id}",
        "distance": distance,
        "metadata": {
            "openalex_id": doc_id,
            "title": title,
            "profile_slugs": "|" + "|".join(slugs) + "|" if slugs else "",
        },
    }


def test_retrieve_with_no_scope_returns_all_records():
    coll = FakeCollection([
        _record("W1", ["alpha"]),
        _record("W2", ["beta"]),
        _record("W3", []),
    ])
    out = retrieve(coll, "anything", scope=[], k=5, embedder=_embedder)
    assert {r["openalex_id"] for r in out} == {"W1", "W2", "W3"}


def test_retrieve_with_single_slug_filters_to_that_profile():
    coll = FakeCollection([
        _record("W1", ["alpha"]),
        _record("W2", ["beta"]),
        _record("W3", ["alpha", "gamma"]),
    ])
    out = retrieve(coll, "anything", scope=["alpha"], k=5, embedder=_embedder)
    assert {r["openalex_id"] for r in out} == {"W1", "W3"}


def test_retrieve_with_multiple_slugs_unions_them():
    coll = FakeCollection([
        _record("W1", ["alpha"]),
        _record("W2", ["beta"]),
        _record("W3", ["gamma"]),
    ])
    out = retrieve(coll, "anything", scope=["alpha", "gamma"], k=5, embedder=_embedder)
    assert {r["openalex_id"] for r in out} == {"W1", "W3"}


def test_retrieve_uses_query_embeddings_when_embedder_supplied():
    coll = FakeCollection([_record("W1", ["alpha"])])
    retrieve(coll, "the question", scope=[], k=5, embedder=_embedder)
    assert coll.last_query is not None
    assert "query_embeddings" in coll.last_query
    assert "query_texts" not in coll.last_query


def test_retrieve_score_converts_from_distance():
    coll = FakeCollection([
        _record("W_NEAR", ["a"], distance=0.1),
        _record("W_FAR",  ["a"], distance=0.9),
    ])
    out = retrieve(coll, "q", scope=["a"], k=5, embedder=_embedder)
    by_id = {r["openalex_id"]: r["score"] for r in out}
    # 1 - distance, both clamped to [0, 1].
    assert by_id["W_NEAR"] == 0.9
    assert by_id["W_FAR"] == pytest.approx(0.1, abs=1e-9)


def test_retrieve_empty_query_returns_empty_list():
    coll = FakeCollection([_record("W1", ["a"])])
    assert retrieve(coll, "", scope=[], embedder=_embedder) == []
    assert retrieve(coll, "   ", scope=[], embedder=_embedder) == []
