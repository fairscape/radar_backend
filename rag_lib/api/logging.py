"""Structured logging setup.

structlog is configured to feed stdlib logging so uvicorn / fastapi
internals end up in the same stream as our own loggers. JSON formatter
in production (``RADAR_LOG_JSON=True``); a colorized console renderer
otherwise for friendlier local dev output.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

import structlog


def configure_logging(
    json: bool = True,
    level: int = logging.INFO,
    file_path: str | Path | None = None,
    file_max_bytes: int = 10_000_000,
    file_backup_count: int = 5,
) -> None:
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
        # Use the stdlib factory (not PrintLoggerFactory) so structlog
        # records flow through the root logger's handlers — that is what
        # gets them into the rotating log file, not just stdout.
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    handlers: list[logging.Handler] = []
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter("%(message)s"))
    handlers.append(stdout_handler)

    if file_path:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=file_max_bytes,
            backupCount=file_backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(file_handler)

    root = logging.getLogger()
    root.handlers[:] = handlers
    root.setLevel(level)

    # Strip uvicorn's own handlers and force propagation so its records
    # (access lines, startup banners, exception tracebacks) flow up to
    # root and therefore into the rotating file. By default uvicorn sets
    # propagate=False on these loggers, which is why uvicorn lines were
    # visible in docker stdout but never landed in the log file.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
        lg.setLevel(logging.INFO)
