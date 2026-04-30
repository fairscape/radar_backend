"""Chat API tests.

Stand up the app, patch the Ollama client and the Chroma resolver so
no network call (or chromadb extra) is required, and exercise the
happy path + the 503 path mandated by the build spec.

Two failure modes the test verifies:

  - ``OllamaUnreachable`` raised by the chat service must be mapped
    to HTTP 503 with an actionable detail naming ``RADAR_OLLAMA_URL``.
  - The user turn is persisted even when the assistant call fails,
    so chat history surfaces the question the user asked.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from rag_lib.api import settings as settings_module
from rag_lib.db import apply_migrations, connect
from rag_lib.db.repos import (
    papers as papers_repo,
    profiles as profiles_repo,
    users as users_repo,
)
from rag_lib.rag import indexer as rag_indexer
from rag_lib.rag import llm as rag_llm
from rag_lib.rag.exceptions import OllamaUnreachable


# -----------------------------------------------------------------------------
# Fakes
# -----------------------------------------------------------------------------


class FakeCollection:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def add(self, *, ids, embeddings, metadatas, documents):
        for i, md, d in zip(ids, metadatas, documents):
            self.records.append({"id": i, "metadata": dict(md), "document": d})

    def get(self, *, where=None):
        matched = [
            r for r in self.records
            if where is None or _matches(r["metadata"], where)
        ]
        return {
            "ids": [r["id"] for r in matched],
            "documents": [r["document"] for r in matched],
            "metadatas": [r["metadata"] for r in matched],
        }

    def query(self, **kwargs):
        n = kwargs.get("n_results", 10)
        where = kwargs.get("where")
        candidates = [
            r for r in self.records
            if where is None or _matches(r["metadata"], where)
        ][:n]
        return {
            "documents": [[r["document"] for r in candidates]],
            "metadatas": [[r["metadata"] for r in candidates]],
            "distances": [[0.2 for _ in candidates]],
        }

    def count(self) -> int:
        return len(self.records)


def _matches(metadata: dict, where: dict) -> bool:
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


class FakeOllama:
    def __init__(self, url: str, model: str, timeout: float = 60.0) -> None:
        self.url = url
        self.model = model
        self.last_messages: list[dict] | None = None

    def generate(self, messages: list[dict]) -> str:
        self.last_messages = list(messages)
        return "Patched assistant reply citing [1]."


class FailingOllama:
    def __init__(self, url: str, model: str, timeout: float = 60.0) -> None:
        self.url = url

    def generate(self, messages: list[dict]) -> str:
        raise OllamaUnreachable("connection refused at http://localhost:11434")


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _seed_db(db_path) -> dict:
    conn = connect(db_path)
    apply_migrations(conn)
    user = users_repo.upsert(conn, "demo@example.com")
    user_id = int(user["id"])
    profile_id = profiles_repo.upsert(
        conn,
        user_id=user_id,
        name="neonatal_vitals",
        embedding_model="placeholder-v1",
        n_seed=0,
        topic_filters={},
    )
    papers_repo.upsert(conn, {
        "openalex_id": "W_VAULT_DOC_1",
        "title": "Neonatal temperature thresholds",
        "source": "user_pdf",
        "uploaded_by_user_id": user_id,
        "file_hash": "abc",
        "n_pages": 4,
        "body_text": (
            "We used a temperature threshold of 36.5 degrees Celsius "
            "for at-risk neonates, escalating warming intervention "
            "above that point. Outcomes were measured at 48 hours."
        ),
    })
    profiles_repo.attach_seed(conn, profile_id, "W_VAULT_DOC_1")
    conn.close()
    return {"user_id": user_id, "profile_id": profile_id}


@pytest.fixture()
def collection() -> FakeCollection:
    return FakeCollection()


@pytest.fixture()
def app(tmp_path, monkeypatch, collection):
    db = tmp_path / "radar.db"
    monkeypatch.setenv("RADAR_DB_PATH", str(db))
    monkeypatch.setenv("RADAR_VAULT_DIR", str(tmp_path / "vault"))
    monkeypatch.setenv("RADAR_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("RADAR_OLLAMA_URL", "http://localhost:11434")
    monkeypatch.setenv("RADAR_OLLAMA_MODEL", "llama3.1:8b")
    monkeypatch.setenv("RADAR_LOG_JSON", "false")
    monkeypatch.setenv("RADAR_SCHEDULER_ENABLED", "false")
    settings_module.get_settings.cache_clear()

    _seed_db(db)

    # Pre-populate the fake collection with a chunk for the seeded paper.
    collection.add(
        ids=["W_VAULT_DOC_1::chunk::0000"],
        embeddings=[[0.1, 0.2, 0.3]],
        metadatas=[{
            "openalex_id": "W_VAULT_DOC_1",
            "title": "Neonatal temperature thresholds",
            "profile_slugs": "|neonatal-vitals|",
            "chunk_index": 0,
        }],
        documents=[
            "We used a temperature threshold of 36.5 degrees Celsius for at-risk neonates."
        ],
    )

    # Patch the per-user Chroma resolver to hand back our in-process fake.
    def _fake_index_user_collection(_settings, _user_id):
        return collection
    monkeypatch.setattr(
        rag_indexer, "index_user_collection", _fake_index_user_collection,
    )

    # Patch the OllamaClient so /api/chat never makes a real network call.
    monkeypatch.setattr(rag_llm, "OllamaClient", FakeOllama)

    from rag_lib.api.app import create_app
    app = create_app()
    yield app

    settings_module.get_settings.cache_clear()


def _request(app, method: str, path: str, **kwargs) -> httpx.Response:
    async def _run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                return await c.request(method, path, **kwargs)
    return asyncio.run(_run())


# -----------------------------------------------------------------------------
# History
# -----------------------------------------------------------------------------


def test_history_starts_empty(app):
    resp = _request(app, "GET", "/api/chat/history")
    assert resp.status_code == 200
    assert resp.json() == []


# -----------------------------------------------------------------------------
# POST /api/chat (happy path)
# -----------------------------------------------------------------------------


def test_chat_post_returns_assistant_turn_with_sources(app):
    resp = _request(
        app, "POST", "/api/chat",
        json={"query": "What temperature thresholds were used?",
              "scope": ["neonatal-vitals"]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["who"] == "assistant"
    assert "Patched assistant reply" in body["body"]
    assert body["t"]
    assert isinstance(body["sources"], list) and len(body["sources"]) == 1
    src = body["sources"][0]
    assert src["n"] == 1
    assert src["title"] == "Neonatal temperature thresholds"
    assert 0.0 <= src["score"] <= 1.0


def test_chat_post_records_user_then_assistant_in_history(app):
    _request(
        app, "POST", "/api/chat",
        json={"query": "Q1", "scope": []},
    )
    history = _request(app, "GET", "/api/chat/history").json()
    assert [t["who"] for t in history] == ["user", "assistant"]
    assert history[0]["body"] == "Q1"


def test_chat_post_unscoped_searches_all_chunks(app, collection):
    # Add a chunk that is not in any profile scope; with an empty scope
    # the retriever should still surface it.
    collection.add(
        ids=["W_FREE::chunk::0000"],
        embeddings=[[0.1, 0.2, 0.3]],
        metadatas=[{
            "openalex_id": "W_FREE",
            "title": "Free-floating note",
            "profile_slugs": "",
            "chunk_index": 0,
        }],
        documents=["Free-floating note text body."],
    )
    resp = _request(
        app, "POST", "/api/chat",
        json={"query": "anything", "scope": []},
    )
    assert resp.status_code == 200
    titles = {s["title"] for s in resp.json()["sources"]}
    assert "Free-floating note" in titles


def test_chat_post_rejects_empty_query(app):
    resp = _request(app, "POST", "/api/chat", json={"query": "", "scope": []})
    assert resp.status_code == 400


# -----------------------------------------------------------------------------
# 503 when Ollama is unreachable
# -----------------------------------------------------------------------------


def test_chat_post_returns_503_when_ollama_unreachable(app, monkeypatch):
    monkeypatch.setattr(rag_llm, "OllamaClient", FailingOllama)
    resp = _request(
        app, "POST", "/api/chat",
        json={"query": "x", "scope": []},
    )
    assert resp.status_code == 503
    detail = resp.json()["detail"].lower()
    assert "llm service unavailable" in detail
    assert "radar_ollama_url" in detail


def test_chat_post_persists_user_turn_even_on_503(app, monkeypatch):
    monkeypatch.setattr(rag_llm, "OllamaClient", FailingOllama)
    _request(
        app, "POST", "/api/chat",
        json={"query": "doomed question", "scope": []},
    )
    history = _request(app, "GET", "/api/chat/history").json()
    # User turn was written even though no assistant turn followed.
    assert any(
        t["who"] == "user" and t["body"] == "doomed question"
        for t in history
    )
    assert all(t["who"] != "assistant" for t in history)
