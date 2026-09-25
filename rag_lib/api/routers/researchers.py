"""Researcher routes — stored people and interests built from them.

  ``POST   /api/researchers/import``          read a Prosopia profile or an
                                              ORCID, write the person, and
                                              import their papers in the
                                              background (``run_id``).
  ``GET    /api/researchers``                 the user's stored people.
  ``GET    /api/researchers/{id}``            one person with papers,
                                              metadata and the interests
                                              built from them.
  ``DELETE /api/researchers/{id}``            forget the person (papers and
                                              interests stay).
  ``POST   /api/researchers/{id}/interests``  a draft seeded with the
                                              chosen papers, synchronously —
                                              they are already embedded.

The import job, its progress row and its status route are the ones
``routers/prosopia.py`` already runs for the wizard; only the target
differs. The client seams (``get_prosopia_client`` /
``get_import_openalex_client``) are shared for the same reason, so a
test that overrides them for the wizard's import covers this one too.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..deps import get_current_user, get_db, get_settings
from ..schemas import (
    Researcher,
    ResearcherDetail,
    ResearcherImportRequest,
    ResearcherImportStart,
    ResearcherInterestRequest,
    ResearcherInterestResponse,
    ResearcherSuggestions,
)
from ..settings import Settings
from .prosopia import get_import_openalex_client, get_prosopia_client, launch_import

router = APIRouter()


TIER = "researcher_import"


@router.post("/import", response_model=ResearcherImportStart)
def import_researcher(
    body: ResearcherImportRequest,
    request: Request,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    prosopia_client: Annotated[object | None, Depends(get_prosopia_client)] = None,
    openalex_client: Annotated[
        object | None, Depends(get_import_openalex_client)
    ] = None,
) -> ResearcherImportStart:
    """Store a person and start importing their papers.

    400 for a bad source, ref or selection; 404 when the source has no
    such person (or none of the selected papers); 502 when the source
    could not be read. The researcher row exists by the time this
    returns, marked as importing until the run finishes.
    """
    from ..services import orcid as orcid_service
    from ..services import prosopia as prosopia_service
    from ..services import researchers as researchers_service

    source = (body.source or "").strip().lower()
    if source not in researchers_service.SOURCES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"source must be one of {', '.join(researchers_service.SOURCES)}",
        )
    try:
        plan = researchers_service.prepare_researcher_import(
            db,
            settings,
            user_id=int(user["id"]),
            source=source,
            ref=body.ref,
            base_url=body.base_url or settings.RADAR_PROSOPIA_BASE_URL,
            paper_ids=body.paper_ids,
            embedding_model=body.embedding_model,
            prosopia_client=prosopia_client,
            openalex_client=openalex_client,
        )
    except (prosopia_service.PapersNotFound, orcid_service.WorksNotFound) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except prosopia_service.ProfileNotFound as exc:
        message = str(exc)
        if body.ref.strip().lower() not in message.lower():
            message = f"prosopia profile '{body.ref}' not found"
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message) from exc
    except prosopia_service.ProsopiaError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"could not read prosopia profile '{body.ref}': {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — OpenAlex could not be read
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"could not read works for '{body.ref}': {exc}",
        ) from exc

    start = launch_import(
        request, db, settings, plan, user_id=int(user["id"]),
        tier=TIER, openalex_client=openalex_client,
    )
    assert plan.researcher_id is not None
    return ResearcherImportStart(researcher_id=plan.researcher_id, run_id=start.run_id)


@router.get("", response_model=list[Researcher])
def list_researchers(
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> list[Researcher]:
    from ..services import researchers as researchers_service

    return researchers_service.list_researchers(db, int(user["id"]))


@router.get("/{researcher_id}", response_model=ResearcherDetail)
def get_researcher(
    researcher_id: int,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> ResearcherDetail:
    from ..services import researchers as researchers_service

    try:
        return researchers_service.get_researcher(db, int(user["id"]), researcher_id)
    except researchers_service.ResearcherNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/{researcher_id}/suggestions", response_model=ResearcherSuggestions)
def suggest_interests(
    researcher_id: int,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ResearcherSuggestions:
    """Two or three interests worth building from this researcher's papers.

    A read over the embeddings the import already stored: nothing is
    embedded or fetched. See ``services.suggestions`` for how the
    groups are chosen; the wizard's coherence check is where an
    accepted suggestion gets validated.
    """
    from ..services import researchers as researchers_service
    from ..services import suggestions as suggestions_service

    try:
        researchers_service.get_researcher_row(db, int(user["id"]), researcher_id)
    except researchers_service.ResearcherNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    result = suggestions_service.suggest_interests(
        db, researcher_id=researcher_id,
        embedding_model=settings.RADAR_DEFAULT_EMBEDDING_MODEL,
    )
    return ResearcherSuggestions(researcher_id=researcher_id, **result)


@router.delete("/{researcher_id}")
def delete_researcher(
    researcher_id: int,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> dict:
    from ..services import researchers as researchers_service

    try:
        researchers_service.delete_researcher(db, int(user["id"]), researcher_id)
    except researchers_service.ResearcherNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return {"ok": True}


@router.post("/{researcher_id}/interests", response_model=ResearcherInterestResponse)
def create_interest(
    researcher_id: int,
    body: ResearcherInterestRequest,
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ResearcherInterestResponse:
    """A new draft interest seeded with this researcher's papers.

    Synchronous: the papers were embedded when the researcher was
    imported, so this is a draft row plus the seed links. The wizard
    resumes it at the coherence check.
    """
    from ..services import researchers as researchers_service

    try:
        draft, n = researchers_service.create_interest(
            db,
            settings,
            user_id=int(user["id"]),
            researcher_id=researcher_id,
            name=body.name,
            openalex_ids=body.openalex_ids,
            embedding_model=body.embedding_model,
        )
    except researchers_service.ResearcherNotFound as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return ResearcherInterestResponse(draft=draft, n_seeds=n)
