"""Structured logging setup.

structlog is configured to feed stdlib logging so uvicorn / fastapi
internals end up in the same stream as our own loggers. JSON formatter
in production (``RADAR_LOG_JSON=True``); a colorized console renderer
otherwise for friendlier local dev output.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(json: bool = True, level: int = logging.INFO) -> None:
    """Wire structlog + stdlib logging into a single processor chain.

    Idempotent — safe to call from both the FastAPI lifespan and tests.
    """
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
    ]

    if json:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)

    for noisy in ("uvicorn.access",):
        logging.getLogger(noisy).setLevel(logging.WARNING)
