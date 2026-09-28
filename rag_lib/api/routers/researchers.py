"""Researchers routes — the user's library of imported Researcher Profiles.

  GET    /api/researchers        → list (this user's rows, newest first)
  POST   /api/researchers        → import a profile.jsonld (paste / file text)
  GET    /api/researchers/{id}   → full detail (parsed view + raw document)
  DELETE /api/researchers/{id}   → remove

Independent of topics: importing here never creates a topic, and the
wizard's FROM PROFILE path never writes here.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from ..deps import get_current_user, get_db
from ..schemas import (
    ResearcherDetail,
    ResearcherImportRequest,
    ResearcherImportResult,
    ResearcherSummary,
)
from ..services import researchers as service
from ..services.rp_profile_import import RpProfileError

router = APIRouter()


@router.get("", response_model=list[ResearcherSummary])
def list_researchers(
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> list[ResearcherSummary]:
    return [ResearcherSummary(**service.summary_dict(r)) for r in service.list_researchers(db, int(user["id"]))]


@router.post("", response_model=ResearcherImportResult)
def import_researcher(
    body: ResearcherImportRequest,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> ResearcherImportResult:
    try:
        row, created = service.import_researcher(
            db, user_id=int(user["id"]), text=body.profile_json, source_kind=body.source_kind,
        )
    except RpProfileError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    detail = service.detail_dict(row)
    return ResearcherImportResult(
        researcher=ResearcherSummary(**service.summary_dict(row)),
        created=created,
        warnings=detail["warnings"],
    )


@router.get("/{researcher_id}", response_model=ResearcherDetail)
def get_researcher(
    researcher_id: int,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> ResearcherDetail:
    row = service.get_researcher(db, int(user["id"]), researcher_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"researcher {researcher_id} not found")
    return ResearcherDetail(**service.detail_dict(row))


@router.delete("/{researcher_id}")
def delete_researcher(
    researcher_id: int,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> dict:
    if not service.delete_researcher(db, int(user["id"]), researcher_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"researcher {researcher_id} not found")
    return {"ok": True}
