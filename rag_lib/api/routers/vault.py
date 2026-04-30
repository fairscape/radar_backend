"""Vault routes — Phase 6.

POST /api/vault/upload  multipart file → VaultDoc (idempotent on hash).
GET  /api/vault/docs    optional ?tag=<profile_slug>|all → VaultDoc[].
GET  /api/vault/stats   → VaultStats.
GET  /api/vault/meta    → VaultMeta.
GET  /api/vault/tags    → {slug: count, "all": total}.
"""

from __future__ import annotations

import sqlite3
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from ..deps import get_current_user, get_db, get_settings
from ..schemas import VaultDoc, VaultMeta, VaultStats
from ..services import vault as vault_service
from ..settings import Settings

router = APIRouter()


@router.post("/upload", response_model=VaultDoc)
def upload(
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    file: UploadFile = File(...),
    profile_slug: Annotated[str | None, Form()] = None,
) -> VaultDoc:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="upload missing filename",
        )
    data = file.file.read()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="uploaded file is empty",
        )
    try:
        doc = vault_service.upload(
            db, settings,
            user_id=int(user["id"]),
            filename=file.filename,
            data=data,
            profile_slug=profile_slug,
        )
    except ImportError as exc:
        # ingest_pdf raises this when phase1b extras (pdfplumber) are missing.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"PDF ingestion unavailable: {exc}",
        ) from exc
    return VaultDoc.model_validate(doc)


@router.get("/docs", response_model=list[VaultDoc])
def list_docs(
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    tag: str | None = None,
) -> list[VaultDoc]:
    docs = vault_service.list_docs(db, int(user["id"]), tag=tag, settings=settings)
    return [VaultDoc.model_validate(d) for d in docs]


@router.get("/stats", response_model=VaultStats)
def vault_stats(
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> VaultStats:
    return VaultStats.model_validate(
        vault_service.stats(db, int(user["id"]), settings)
    )


@router.get("/meta", response_model=VaultMeta)
def vault_meta(
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> VaultMeta:
    return VaultMeta.model_validate(
        vault_service.meta(db, settings, int(user["id"]))
    )


@router.get("/tags", response_model=dict[str, int])
def vault_tags(
    user: Annotated[sqlite3.Row, Depends(get_current_user)],
    db: Annotated[sqlite3.Connection, Depends(get_db)],
) -> dict[str, int]:
    return vault_service.tag_counts(db, int(user["id"]))
