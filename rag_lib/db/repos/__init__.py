"""Hand-rolled repos.

Each module exposes thin module-level functions that take a
``sqlite3.Connection`` as the first arg and return ``sqlite3.Row`` (or
plain dicts where convenient). No ORM. Higher layers (CLI, API services)
convert rows ↔ domain objects.
"""

from . import (
    users, profiles, papers, embeddings, candidates,
    gather_runs, schedules, feedback, chat,
)

__all__ = [
    "users",
    "profiles",
    "papers",
    "embeddings",
    "candidates",
    "gather_runs",
    "schedules",
    "feedback",
    "chat",
]
