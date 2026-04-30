"""Run the FastAPI service.

Usage::

    python -m cli.serve
    python -m cli.serve --host 0.0.0.0 --port 8000 --reload

Defaults are pulled from ``rag_lib.api.settings.Settings``; CLI flags
override on a per-flag basis. ``--reload`` is local-dev only — uvicorn
runs the worker in a subprocess when reload is enabled.
"""

from __future__ import annotations

import argparse

import uvicorn

from rag_lib.api.settings import get_settings


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="serve", description="Run the RADAR API.")
    parser.add_argument("--host", default=settings.RADAR_HOST)
    parser.add_argument("--port", type=int, default=settings.RADAR_PORT)
    parser.add_argument("--reload", action="store_true",
                        help="Enable autoreload (local development only).")
    parser.add_argument("--log-level", default="info",
                        choices=["critical", "error", "warning", "info", "debug", "trace"])
    args = parser.parse_args(argv)

    uvicorn.run(
        "rag_lib.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
