"""Chat service — orchestration for the Phase 9 RAG chat routes.

``post_chat`` is the write surface:
  1. Persist the user turn (so chat history reflects intent even if
     Ollama is down).
  2. Resolve the per-user Chroma collection.
  3. Retrieve top-k scoped chunks with the same embedder used at
     index time.
  4. Build the prompt and call Ollama. ``OllamaUnreachable`` propagates
     to the router for a 503 — the assistant turn is *not* persisted in
     that branch on purpose.
  5. Persist the assistant turn with ``sources_json``.
  6. Return the assistant ChatTurn dict (router validates it).

``history`` reads the most-recent turns for the demo's flat history view.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from ...db.repos import chat as chat_repo
from ...embedders import get_embedder
from ...rag import indexer as rag_indexer
from ...rag import llm as rag_llm
from ...rag import retriever as rag_retriever


TOP_K = 10


def post_chat(
    conn: sqlite3.Connection,
    settings: Any,
    *,
    user_id: int,
    query: str,
    scope: list[str] | None = None,
) -> dict:
    """Run a single user → assistant turn. Returns the assistant ChatTurn."""
    scope_list = list(scope or [])

    chat_repo.append_turn(
        conn,
        user_id=user_id,
        role="user",
        body=query,
        scope=scope_list,
    )

    collection = rag_indexer.index_user_collection(settings, user_id)

    embedding_model = settings.RADAR_DEFAULT_EMBEDDING_MODEL
    embedder = get_embedder(embedding_model)

    retrieved = rag_retriever.retrieve(
        collection,
        query,
        scope=scope_list,
        k=TOP_K,
        embedder=embedder,
    )

    messages = rag_llm.build_prompt(query, retrieved)
    client = rag_llm.OllamaClient(
        settings.RADAR_OLLAMA_URL,
        settings.RADAR_OLLAMA_MODEL,
    )
    answer = client.generate(messages)  # raises OllamaUnreachable

    sources = [
        {"n": i + 1, "title": c["title"], "score": float(c["score"])}
        for i, c in enumerate(retrieved)
    ]

    assistant_row = chat_repo.append_turn(
        conn,
        user_id=user_id,
        role="assistant",
        body=answer,
        sources=sources,
        scope=scope_list,
    )

    return _row_to_chat_turn(assistant_row)


def history(conn: sqlite3.Connection, user_id: int) -> list[dict]:
    rows = chat_repo.recent(conn, user_id)
    return [_row_to_chat_turn(r) for r in rows]


def _row_to_chat_turn(row: sqlite3.Row) -> dict:
    sources = chat_repo.decode_sources(row)
    turn: dict[str, Any] = {
        "who": row["role"],
        "t": row["ts"],
        "body": row["body"],
    }
    if sources:
        turn["sources"] = sources
    return turn
