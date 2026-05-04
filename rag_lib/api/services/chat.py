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

import structlog

from ...db.repos import chat as chat_repo
from ...embedders import get_embedder
from ...rag import indexer as rag_indexer
from ...rag import llm as rag_llm
from ...rag import retriever as rag_retriever


log = structlog.get_logger("rag_lib.api.services.chat")

TOP_K = 20


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

    log.info(
        "chat.post.start",
        user_id=user_id,
        query_preview=query[:120],
        scope=scope_list,
    )

    chat_repo.append_turn(
        conn,
        user_id=user_id,
        role="user",
        body=query,
        scope=scope_list,
    )

    # Prefer the chat-optimized collection (indexed at upload time with
    # ``RADAR_CHAT_EMBEDDING_MODEL``). Fall back to the SPECTER2 vault
    # if the chat collection is empty — happens when ollama was down at
    # upload time, or the deployment runs without the chat embedder.
    chat_model = (settings.RADAR_CHAT_EMBEDDING_MODEL or "").strip()
    chat_collection = None
    chat_count = 0
    if chat_model:
        try:
            chat_collection = rag_indexer.index_user_chat_collection(
                settings, user_id
            )
            chat_count = int(chat_collection.count())
        except Exception as exc:
            log.warning(
                "chat.post.chat_collection_unavailable",
                user_id=user_id,
                reason=type(exc).__name__,
                detail=str(exc)[:200],
            )

    if chat_collection is not None and chat_count > 0:
        collection = chat_collection
        embedding_model = chat_model
    else:
        collection = rag_indexer.index_user_collection(settings, user_id)
        embedding_model = settings.RADAR_DEFAULT_EMBEDDING_MODEL

    try:
        collection_count = int(collection.count())
    except Exception:
        collection_count = -1
    log.info(
        "chat.post.collection",
        user_id=user_id,
        collection_count=collection_count,
        embedder=embedding_model,
        used_chat_collection=(collection is chat_collection),
        chat_collection_count=chat_count,
    )

    embedder = get_embedder(embedding_model)

    retrieved = rag_retriever.retrieve(
        collection,
        query,
        scope=scope_list,
        k=TOP_K,
        embedder=embedder,
    )

    log.info(
        "chat.post.retrieved",
        user_id=user_id,
        embedder=embedding_model,
        top_k=TOP_K,
        n_retrieved=len(retrieved),
        retrieved=[
            {
                "title": c["title"][:80],
                "openalex_id": c["openalex_id"],
                "score": round(c["score"], 4),
                "chunk_chars": len(c["text"]),
            }
            for c in retrieved
        ],
    )

    messages = rag_llm.build_prompt(query, retrieved)
    total_chars = sum(len(m.get("content", "")) for m in messages)
    log.info(
        "chat.post.prompt",
        user_id=user_id,
        model=settings.RADAR_OLLAMA_MODEL,
        n_messages=len(messages),
        total_chars=total_chars,
    )
    client = rag_llm.OllamaClient(
        settings.RADAR_OLLAMA_URL,
        settings.RADAR_OLLAMA_MODEL,
    )
    answer = client.generate(messages)  # raises OllamaUnreachable
    log.info(
        "chat.post.answer",
        user_id=user_id,
        n_chars=len(answer),
        preview=answer[:200],
    )

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


def clear_history(conn: sqlite3.Connection, user_id: int) -> int:
    """Delete every chat turn for a user. Returns rows removed."""
    deleted = chat_repo.delete_all_for_user(conn, user_id)
    log.info("chat.clear_history", user_id=user_id, deleted=deleted)
    return deleted


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
