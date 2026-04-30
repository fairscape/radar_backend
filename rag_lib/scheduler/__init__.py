"""Background scheduler for daily gather jobs.

Owns the in-process APScheduler instance the FastAPI lifespan hook
boots in :mod:`rag_lib.api.app`. Public surface is intentionally tiny:

  build_scheduler(settings) -> BackgroundScheduler
  start(scheduler, settings) -> None
  stop(scheduler) -> None
  gather_for_profile(user_id, profile_id, *, settings=None) -> int

Routers reach the live scheduler through the ``get_scheduler`` FastAPI
dependency so per-request code never imports this module directly.
"""

from .runner import build_scheduler, start, stop
from .jobs import gather_for_profile

__all__ = ["build_scheduler", "start", "stop", "gather_for_profile"]
