"""SQLite persistence layer.

The system of record for profiles, papers, embeddings, candidate triage
state, and gather-run history. Designed in ``DB_DESIGN.md``; extended
for multi-user via the ``users`` table in migration 0002.

Public surface:
    connect(path)              -> sqlite3.Connection (WAL, FKs ON, Row rows)
    apply_migrations(conn)     -> applies any pending migration files
    encode_vector / decode_vector — float32 BLOB codec for embeddings
"""

from .connection import connect
from .migrate import apply_migrations, applied_versions
from .codec import encode_vector, decode_vector

__all__ = [
    "connect",
    "apply_migrations",
    "applied_versions",
    "encode_vector",
    "decode_vector",
]
