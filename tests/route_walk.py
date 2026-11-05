"""Every API route as it is served: full path, methods, and every
dependency that runs for it (its router's and the include's too).

FastAPI 0.141 keeps an included router in ``app.routes`` as one
``_IncludedRouter``, not as its ``APIRoute``s: a test that filtered
``app.routes`` by ``isinstance(r, APIRoute)`` checked nothing and always
passed (review 2026-11-05 BL-1). This walks the routes the way they are
served and is checked against the OpenAPI listing, so a FastAPI that lays
them out differently fails here instead of passing silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Served:
    path: str
    methods: frozenset
    calls: frozenset          # every dependency callable, nested ones included


def _calls(dep) -> set:
    out = {dep.call}
    for d in dep.dependencies:
        out |= _calls(d)
    return out


def served_routes(app) -> list[Served]:
    from fastapi.routing import APIRoute

    out: list[Served] = []
    for entry in app.routes:
        if isinstance(entry, APIRoute):
            out.append(Served(entry.path, frozenset(entry.methods), frozenset(_calls(entry.dependant))))
        elif hasattr(entry, "effective_route_contexts"):
            for ctx in entry.effective_route_contexts():
                out.append(Served(ctx.path, frozenset(ctx.methods), frozenset(_calls(ctx.dependant))))
    ops = {(m, r.path) for r in out for m in r.methods if r.path.startswith(("/api", "/auth"))}
    listed = {(m.upper(), path) for path, item in app.openapi()["paths"].items() for m in item}
    # OpenAPI drops ":path" from a path parameter.
    assert {(m, re.sub(r":\w+}", "}", p)) for m, p in ops} == listed, "routes walked != OpenAPI listing"
    return out
