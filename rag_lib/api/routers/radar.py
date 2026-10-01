"""Radar routes.

Reads ``profile_candidates`` for daily cards, toggles save / dismiss
state, and (Phase 8) writes a feedback event to JSONL + DB whenever a
card transitions into ``saved`` or ``dismissed``. Toggling a state
*off* does not produce a feedback event.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from rag_lib.feedback import FeedbackEvent, log_event

from ..deps import get_current_user, get_db
from ..schemas import (
    Bucket,
    CardActionResponse,
    DailyRadarResponse,
)


class CardActionRequest(BaseModel):
    """Body of ``POST /api/radar/cards/save|dismiss``.

    ``card_id`` is the OpenAlex id stored on ``profile_candidates``,
    which is the full URL form (``https://openalex.org/W…``). Sending it
    in the body avoids the slash-encoding issue that breaks path-param
    routing.
    """

    card_id: str
    # The interest the card was shown under (its ``profile`` slug). Without
    # it the server picks the interest that fetched the paper most recently,
    # which need not be the one on screen -- and since save is a toggle, a
    # click could un-save the paper in another interest. Optional only for
    # older clients.
    profile: str | None = None
from ..services import radar as radar_service
from ..settings import Settings, get_settings

router = APIRouter()


@router.get("/daily", response_model=DailyRadarResponse)
def daily_radar(
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    profile: str | None = Query(default=None),
    bucket: Bucket | None = Query(default=None),
) -> DailyRadarResponse:
    return radar_service.daily(
        db, int(user["id"]),
        profile_slug=profile,
        bucket=bucket,
    )


def _log_feedback_for(
    db: sqlite3.Connection,
    settings: Settings,
    user_id: int,
    card_id: str,
    action: str,
    profile: str | None = None,
) -> None:
    """Append a feedback event for a card transition.

    Best-effort: caller already updated the candidate row, so logging
    failures (e.g. missing vault dir) should not 500 the API. Resolves
    fresh from the DB so the score / selector / doi reflect what the
    user actually saw.
    """
    ctx = radar_service.card_feedback_context(db, user_id, card_id, profile_slug=profile)
    if ctx is None:
        return
    log_event(
        settings.RADAR_VAULT_DIR,
        user_id,
        db,
        FeedbackEvent(
            profile_id=ctx["profile_id"],
            profile_slug=ctx["profile_slug"],
            openalex_id=ctx["openalex_id"],
            action=action,
            score=ctx["score_pct"],
            selector=ctx["selector"],
            selector_config_hash=ctx["selector_config_hash"],
            doi=ctx["doi"],
        ),
    )


@router.post("/cards/save", response_model=CardActionResponse)
def save_card(
    body: CardActionRequest,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> CardActionResponse:
    res = radar_service.save(db, int(user["id"]), body.card_id, profile_slug=body.profile)
    if res is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"card '{body.card_id}' not found for current user",
        )
    if res.state == "saved":
        _log_feedback_for(db, settings, int(user["id"]), body.card_id, "saved", body.profile)
    return res


@router.post("/cards/dismiss", response_model=CardActionResponse)
def dismiss_card(
    body: CardActionRequest,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> CardActionResponse:
    res = radar_service.dismiss(db, int(user["id"]), body.card_id, profile_slug=body.profile)
    if res is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"card '{body.card_id}' not found for current user",
        )
    if res.state == "dismissed":
        _log_feedback_for(db, settings, int(user["id"]), body.card_id, "dismissed", body.profile)
    return res
