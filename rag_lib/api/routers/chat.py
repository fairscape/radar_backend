"""Chat routes — Phase 9.

Two endpoints:

  GET  /api/chat/history  → ChatTurn[] (most-recent N turns, oldest-first)
  POST /api/chat          → ChatTurn (assistant reply with sources)

Ollama is hard-required: when the backing service can't be reached the
service raises ``OllamaUnreachable`` which the router maps to a 503
with an actionable message naming the env var the operator should
check (``RADAR_OLLAMA_URL``) and reminding them the configured model
must be pulled. We do not fall back to a templated reply — chat is a
demo-blocking feature and a fake answer would be worse than a clear
failure.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ..deps import get_current_user, get_db, get_settings
from ..schemas import ChatRequest, ChatTurn
from ..services import chat as chat_service
from ..settings import Settings
from ...rag.exceptions import OllamaUnreachable

router = APIRouter()


@router.get("/history", response_model=list[ChatTurn])
def chat_history(
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> list[ChatTurn]:
    turns = chat_service.history(db, int(user["id"]))
    return [ChatTurn.model_validate(t) for t in turns]


@router.delete("/history")
def clear_chat_history(
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> dict[str, int]:
    deleted = chat_service.clear_history(db, int(user["id"]))
    return {"deleted": deleted}


@router.post("", response_model=ChatTurn)
def chat(
    body: ChatRequest,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatTurn:
    query = (body.query or "").strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="query is required",
        )
    try:
        turn = chat_service.post_chat(
            db, settings,
            user_id=int(user["id"]),
            query=query,
            scope=body.scope,
        )
    except OllamaUnreachable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"LLM service unavailable: {exc}. "
                "Confirm RADAR_OLLAMA_URL and that the configured model "
                "is pulled."
            ),
        ) from exc
    except ImportError as exc:
        # Chroma extras missing — surface the same 503 contract so the
        # operator gets an actionable hint instead of a 500.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vector index unavailable: {exc}",
        ) from exc
    return ChatTurn.model_validate(turn)
