"""scripts/build_topic_index.py --rich: the index the UMLS mapper searches
(CTX-2) — topics with their description and keywords, embedded as
``rich_topic_text``, written to the RICH files with their model.
OpenAlex and the embedder are stubs."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def script(monkeypatch):
    spec = importlib.util.spec_from_file_location("build_topic_index", ROOT / "scripts" / "build_topic_index.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    return mod


def _topic(i):
    return {"id": f"https://openalex.org/T{i}", "display_name": f"Topic {i}", "description": f"About {i}.",
            "keywords": [{"display_name": "low", "score": 0.1}, {"display_name": "high", "score": 0.9}],
            "subfield": {"display_name": "Sub"}, "field": {"display_name": "F"}, "domain": {"display_name": "D"}}


def test_the_topics_are_fetched_with_description_and_keywords_page_by_page(script, monkeypatch):
    from rag_lib.openalex_client import OpenAlexClient

    pages = {1: [_topic(i) for i in range(200)], 2: [_topic(200)]}
    asked = []

    def page(self, n, per_page=200):
        asked.append((n, per_page))
        self._api_calls += 1
        return {"results": pages.get(n, []), "meta": {"count": 201}}
    monkeypatch.setattr(OpenAlexClient, "topics_page", page)
    topics = script.fetch_rich_topics()
    assert asked == [(1, 200), (2, 200)]
    assert len(topics) == 201
    assert topics[0] == {"id": "https://openalex.org/T0", "display_name": "Topic 0", "description": "About 0.",
                         "keywords": _topic(0)["keywords"], "subfield": "Sub", "field": "F", "domain": "D"}


def test_topics_page_asks_for_the_description_and_keywords(monkeypatch):
    from rag_lib.openalex_client import OpenAlexClient

    seen = {}
    monkeypatch.setattr(OpenAlexClient, "_get", lambda self, path, params=None: seen.update(path=path, **params) or {})
    OpenAlexClient("x@example.com", api_key="made-up").topics_page(3)
    assert seen["path"] == "/topics" and seen["page"] == 3 and seen["per-page"] == 200
    assert {"description", "keywords", "display_name", "id"} <= set(seen["select"].split(","))


def test_rich_build_from_a_saved_file_writes_the_rich_index(script, tmp_path, monkeypatch):
    from rag_lib import embedders
    from rag_lib.umls import cache

    topics = [{**_topic(i), "subfield": "Sub", "field": "F", "domain": "D"} for i in range(3)]
    src = tmp_path / "topics.json"
    src.write_text(json.dumps(topics))
    texts = []
    monkeypatch.setattr(script, "get_embedder", lambda name: (lambda t: texts.append(t) or [1.0, float(len(texts)), 0.0]))
    out = tmp_path / "cache"
    monkeypatch.setattr(sys, "argv", ["x", "--rich", "--topics-json", str(src), "--output-dir", str(out),
                                      "--embedding-model", "model-x"])
    script.main()
    assert texts == [cache.rich_topic_text(t) for t in topics]
    assert texts[0] == "Topic 0. About 0. Keywords: high; low"
    # the name-only index is not touched
    assert sorted(p.name for p in out.iterdir()) == sorted([cache.RICH_TOPICS_FILE, cache.RICH_EMBEDDINGS_FILE,
                                                             cache.RICH_META_FILE])
    meta = json.loads((out / cache.RICH_META_FILE).read_text())
    assert (meta["embedding_model"], meta["dim"], meta["n_topics"]) == ("model-x", 3, 3)
    assert np.load(out / cache.RICH_EMBEDDINGS_FILE)["embeddings"].shape == (3, 3)
    monkeypatch.setattr(cache, "_rich_index", None)
    index = cache.get_rich_topic_index(out)
    index.check_compatible("model-x", dim=3)
    assert index.topics[2]["display_name"] == "Topic 2"


def test_the_name_only_build_is_unchanged(script, tmp_path, monkeypatch):
    from rag_lib.umls import cache

    monkeypatch.setattr(script, "fetch_all_topics", lambda mailto: [{"id": "T1", "display_name": "Name"},
                                                                  {"id": "T2", "display_name": " ", "subfield": "Sub"}])
    texts = []
    monkeypatch.setattr(script, "get_embedder", lambda name: (lambda t: texts.append(t) or [1.0, 0.0]))
    out = tmp_path / "cache"
    monkeypatch.setattr(sys, "argv", ["x", "--output-dir", str(out), "--embedding-model", "model-x"])
    script.main()
    assert texts == ["Name", "Sub"]
    assert sorted(p.name for p in out.iterdir()) == sorted([cache.TOPICS_FILE, cache.EMBEDDINGS_FILE, cache.META_FILE])


def test_topics_json_needs_rich(script, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--topics-json", str(tmp_path / "t.json")])
    with pytest.raises(SystemExit):
        script.main()
