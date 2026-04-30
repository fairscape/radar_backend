"""Bridge between domain objects (Profile, Paper) and the SQLite layer.

Phase 2 of the productionization plan. The CLI (cli/ingest.py,
cli/gather.py) and the Phase 4+ API service both go through this module
rather than touching the repos directly, so the rules for "how a
profile gets persisted" live in one place.
"""

from .db_store import (
    dedup_and_insert_candidates,
    resolve_user_id,
    store_papers,
    store_profile_from_object,
)

__all__ = [
    "dedup_and_insert_candidates",
    "resolve_user_id",
    "store_papers",
    "store_profile_from_object",
]
