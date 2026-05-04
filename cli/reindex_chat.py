"""Backfill the per-user ``vault_chat`` Chroma collection.

Re-chunks every paper a user has uploaded and indexes it with the
chat-side embedder (``RADAR_CHAT_EMBEDDING_MODEL``). The SPECTER2
``vault`` collection is untouched.

Run inside the backend container after switching on the chat embedder
for the first time, or any time you change ``RADAR_CHAT_EMBEDDING_MODEL``::

    python -m cli.reindex_chat --user-id 2
    python -m cli.reindex_chat --all-users
"""

from __future__ import annotations

import argparse
import sys

from rag_lib.api.settings import get_settings
from rag_lib.db import connect
from rag_lib.db.repos import vault as vault_repo
from rag_lib.embedders import get_embedder
from rag_lib.rag import indexer as rag_indexer


def reindex_user(conn, settings, user_id: int) -> tuple[int, int]:
    """Returns (papers_processed, chunks_written)."""
    chat_model = (settings.RADAR_CHAT_EMBEDDING_MODEL or "").strip()
    if not chat_model:
        print("RADAR_CHAT_EMBEDDING_MODEL is empty — nothing to do.")
        return (0, 0)

    embedder = get_embedder(chat_model)
    chat_collection = rag_indexer.index_user_chat_collection(settings, user_id)

    rows = conn.execute(
        """
        SELECT openalex_id FROM papers
        WHERE uploaded_by_user_id = ? AND body_text IS NOT NULL
        """,
        (user_id,),
    ).fetchall()

    papers = 0
    chunks = 0
    for row in rows:
        oa = row["openalex_id"]
        paper_row = conn.execute(
            "SELECT * FROM papers WHERE openalex_id = ?", (oa,)
        ).fetchone()
        if paper_row is None:
            continue
        slugs = vault_repo.tags_for_paper(conn, user_id, oa)
        try:
            # Match the smaller chunk size used at upload time so we
            # stay under the 512-token input cap on mxbai/nomic/bge.
            n = rag_indexer.index_paper(
                chat_collection,
                paper_row,
                embedder,
                profile_slugs=slugs,
                target_tokens=300,
                overlap=40,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"  [skip] {oa}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        papers += 1
        chunks += n
        print(f"  [ok]   {oa}  → {n} chunks")
    return (papers, chunks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--user-id", type=int, help="single user to reindex")
    grp.add_argument("--all-users", action="store_true",
                     help="reindex every user with uploaded papers")
    args = ap.parse_args()

    settings = get_settings()
    conn = connect(settings.RADAR_DB_PATH)

    if args.user_id is not None:
        user_ids = [args.user_id]
    else:
        rows = conn.execute(
            "SELECT DISTINCT uploaded_by_user_id AS id FROM papers "
            "WHERE uploaded_by_user_id IS NOT NULL"
        ).fetchall()
        user_ids = [int(r["id"]) for r in rows]

    chat_model = settings.RADAR_CHAT_EMBEDDING_MODEL
    print(f"chat embedder: {chat_model!r}")
    print(f"users: {user_ids}")

    grand_papers = 0
    grand_chunks = 0
    for uid in user_ids:
        print(f"\n=== user {uid} ===")
        p, c = reindex_user(conn, settings, uid)
        print(f"  user {uid}: {p} papers, {c} chunks")
        grand_papers += p
        grand_chunks += c

    print(f"\nDONE — {grand_papers} papers, {grand_chunks} chunks across {len(user_ids)} user(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
