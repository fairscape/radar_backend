"""Chat routes — Phase 9 + provider abstraction.

Three endpoints:

  GET  /api/chat/history    → ChatTurn[] (most-recent N turns, oldest-first)
  GET  /api/chat/providers  → ProvidersResponse (default + per-provider status)
  POST /api/chat            → ChatTurn (assistant reply with sources)

The LLM is hard-required: when the active provider can't be reached the
service raises ``LLMUnreachable`` which the router maps to a 503 with
an actionable message naming the env var the operator should check for
the active provider (``RADAR_OLLAMA_URL`` for Ollama,
``RADAR_ANTHROPIC_API_KEY`` for Anthropic, ``RADAR_OPENAI_API_KEY`` for
OpenAI). ``LLMNotConfigured`` (missing key or missing extras) maps to
the same 503 with a fix-the-config hint. We do not fall back to a
templated reply — chat is a demo-blocking feature and a fake answer
would be worse than a clear failure.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ..deps import get_current_user, get_db, get_settings
from ..schemas import (
    ChatRequest,
    ChatTurn,
    ProviderInfo,
    ProvidersResponse,
)
from ..services import chat as chat_service
from ..settings import Settings
from ...rag import llm as rag_llm
from ...rag.exceptions import LLMNotConfigured, LLMUnreachable

router = APIRouter()


_PROVIDER_ENV_HINT: dict[str, str] = {
    "ollama": "Confirm RADAR_OLLAMA_URL and that the configured model is pulled.",
    "anthropic": "Set RADAR_ANTHROPIC_API_KEY in .env and restart the server.",
    "openai": "Set RADAR_OPENAI_API_KEY in .env and restart the server.",
}


def _env_hint(provider: str) -> str:
    return _PROVIDER_ENV_HINT.get(
        provider,
        "Check the LLM provider configuration in .env.",
    )


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


@router.get("/providers", response_model=ProvidersResponse)
def chat_providers(
    settings: Annotated[Settings, Depends(get_settings)],
) -> ProvidersResponse:
    """Provider status for the chat UI.

    Returns one entry per supported provider with ``configured: bool``
    so the frontend can disable un-configured options. ``configured``
    intentionally surfaces nothing about the key value — not its
    length, not its prefix, not whether it was loaded from .env vs
    process env. Operators rotate keys; the API does not.
    """
    available = [
        ProviderInfo(
            id=name,
            model=rag_llm.provider_model(settings, name),
            configured=rag_llm.provider_configured(settings, name),
        )
        for name in rag_llm.SUPPORTED_PROVIDERS
    ]
    return ProvidersResponse(
        default=settings.RADAR_LLM_PROVIDER,
        available=available,
    )


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

    # Resolve the active provider up front so the 503 messages below
    # can name the relevant env var even if construction itself fails.
    try:
        active_provider = rag_llm.resolve_provider(settings, body.provider)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    try:
        turn = chat_service.post_chat(
            db, settings,
            user_id=int(user["id"]),
            query=query,
            scope=body.scope,
            provider=active_provider,
        )
    except LLMNotConfigured as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"LLM provider not configured: {exc}. {_env_hint(active_provider)}",
        ) from exc
    except LLMUnreachable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"LLM service unavailable: {exc}. "
                f"{_env_hint(active_provider)}"
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
