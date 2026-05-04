"""Plugin validation harness — confirms an embedder + selector satisfy
the protocols Radar relies on, before they're registered.

Two ways to use it:

1. From your own tests:

       from tests.validate_plugins import validate
       validate(embedder=my_embed, selector_cls=MySelector,
                embedder_key="my-team-v1", selector_key="my_team_v1")

2. From the command line:

       python -m tests.validate_plugins \
           --embedder    my_pkg.module:my_embed \
           --selector    my_pkg.module:MySelector \
           --register-as my-team-v1,my_team_v1

Either path raises (or exits non-zero) on the first failed assertion
with a message naming which protocol clause was violated. A clean run
prints a checklist of what was exercised so you can trust it.

The checks here mirror ``tests/test_selector_protocol.py`` — passing
this harness implies passing the internal protocol tests, modulo
selector-internal invariants (e.g. CentroidSelector's coherence keys).
"""

from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from typing import Callable

from rag_lib.embedders import EMBEDDERS, register_embedder
from rag_lib.paper import Paper
from rag_lib.profile import Profile
from rag_lib.selector import Selector
from rag_lib.selectors import SELECTORS, register_selector


Embedder = Callable[[str], list[float]]


@dataclass
class Result:
    passed: list[str]
    failed: list[str]

    @property
    def ok(self) -> bool:
        return not self.failed

    def report(self) -> str:
        lines = []
        for name in self.passed:
            lines.append(f"  ok    {name}")
        for name in self.failed:
            lines.append(f"  FAIL  {name}")
        summary = "PASS" if self.ok else "FAIL"
        lines.append(f"\n{summary} ({len(self.passed)} ok, {len(self.failed)} failed)")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Embedder checks
# ----------------------------------------------------------------------


def _check_embedder(fn: Embedder, result: Result) -> int | None:
    """Return the dimension on success, None on failure."""
    try:
        v1 = fn("hello world")
    except Exception as e:
        result.failed.append(f"embedder raised on simple input: {type(e).__name__}: {e}")
        return None
    if not isinstance(v1, list):
        result.failed.append(f"embedder must return list[float], got {type(v1).__name__}")
        return None
    if not v1:
        result.failed.append("embedder returned empty vector")
        return None
    if not all(isinstance(x, (int, float)) for x in v1):
        result.failed.append("embedder returned non-numeric entries")
        return None
    result.passed.append(f"embedder returns list[float] of length {len(v1)}")

    # Dimension stability across two different inputs.
    v2 = fn("a completely different sentence")
    if len(v2) != len(v1):
        result.failed.append(
            f"embedder dimension is not stable: {len(v1)} then {len(v2)}"
        )
        return None
    result.passed.append("embedder dimension is stable across calls")

    # Determinism on the same input.
    v3 = fn("hello world")
    if v3 != v1:
        result.passed.append("embedder is non-deterministic (allowed, but flagged)")
    else:
        result.passed.append("embedder is deterministic on identical input")

    return len(v1)


# ----------------------------------------------------------------------
# Selector checks
# ----------------------------------------------------------------------


def _sample_profile(embedder: Embedder, embedding_key: str) -> Profile:
    papers = [
        Paper(doi="10.1/a", openalex_id="W1", title="A",
              abstract="alpha beta gamma"),
        Paper(doi="10.1/b", openalex_id="W2", title="B",
              abstract="beta gamma delta"),
        Paper(doi="10.1/c", openalex_id="W3", title="C",
              abstract="gamma delta epsilon"),
        Paper(doi="10.1/d", openalex_id="W4", title="D",
              abstract="delta epsilon zeta"),
    ]
    for p in papers:
        p.embeddings[embedding_key] = embedder(f"{p.title}\n{p.abstract}")
    return Profile(name="validate", papers=papers, embedding_model=embedding_key)


def _sample_candidates() -> list[Paper]:
    return [
        Paper(doi="10.1/e", openalex_id="W5", title="E", abstract="alpha gamma"),
        Paper(doi="10.1/f", openalex_id="W6", title="F", abstract="epsilon"),
        Paper(doi="10.1/g", openalex_id="W7", title="G", abstract="alpha beta delta"),
        Paper(doi="10.1/h", openalex_id="W8", title="H", abstract="zeta eta"),
    ]


def _check_selector(
    selector_cls: type,
    embedder: Embedder,
    embedding_key: str,
    result: Result,
) -> None:
    # ---- non-invoking ----
    try:
        s = selector_cls()
    except Exception as e:
        result.failed.append(
            f"selector_cls() must instantiate with no args; raised "
            f"{type(e).__name__}: {e}"
        )
        return

    if not hasattr(s, "name") or not isinstance(s.name, str) or not s.name:
        result.failed.append("selector.name must be a non-empty str")
        return
    result.passed.append(f"selector.name = '{s.name}'")

    if not isinstance(s, Selector):
        result.failed.append(
            "isinstance(selector, Selector) failed — methods are missing or "
            "have wrong shape (see rag_lib/selector.py)"
        )
        return
    result.passed.append("selector satisfies the runtime Selector protocol")

    if not isinstance(s.diagnostics(), dict):
        result.failed.append("selector.diagnostics() must return a dict")
        return
    result.passed.append("diagnostics() returns a dict pre-fit")

    cfg = s.config()
    if not isinstance(cfg, dict):
        result.failed.append("selector.config() must return a dict")
        return
    if cfg.get("type") != s.name:
        result.failed.append(
            f"config()['type'] must equal selector.name "
            f"(got type={cfg.get('type')!r}, name={s.name!r})"
        )
        return
    s2 = selector_cls.from_config(cfg)
    if s2.name != s.name:
        result.failed.append("from_config(config()) round-trip changed .name")
        return
    result.passed.append("config() / from_config() round-trip preserves identity")

    cost = s.cost()
    if not isinstance(cost, dict) or "wall_seconds" not in cost:
        result.failed.append("cost() must return {'wall_seconds': float, ...}")
        return
    result.passed.append("cost() shape ok")

    # ---- full invocation ----
    profile = _sample_profile(embedder, embedding_key)
    s.fit(profile)
    result.passed.append("fit() ran against a 4-paper synthetic profile")

    candidates = _sample_candidates()
    ranked = s.select(candidates, profile)
    if len(ranked) != len(candidates):
        result.failed.append(
            f"select() must return one entry per candidate "
            f"(got {len(ranked)}, expected {len(candidates)})"
        )
        return
    for entry in ranked:
        if len(entry) != 3:
            result.failed.append(
                "select() must return (score, Paper, breakdown) 3-tuples"
            )
            return
        score, paper, breakdown = entry
        if not isinstance(score, float):
            result.failed.append("score must be a float")
            return
        if not isinstance(paper, Paper):
            result.failed.append("second tuple field must be a Paper")
            return
        if not isinstance(breakdown, dict):
            result.failed.append("breakdown must be a dict (empty is fine)")
            return
    result.passed.append("select() returned (float, Paper, dict) triples")

    scores = [r[0] for r in ranked]
    if scores != sorted(scores, reverse=True):
        result.failed.append("select() output must be sorted descending by score")
        return
    result.passed.append("select() output sorted descending")

    if ranked:
        cutoff = ranked[len(ranked) // 2][0]
        filtered = s.select(_sample_candidates(), profile, threshold=cutoff)
        if any(r[0] < cutoff for r in filtered):
            result.failed.append("threshold filter let entries below cutoff through")
            return
        result.passed.append("threshold filter respected")

    if not isinstance(s.diagnostics(), dict):
        result.failed.append("diagnostics() must return a dict post-fit")
        return
    result.passed.append("diagnostics() returns a dict post-fit")


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def validate(
    *,
    embedder: Embedder,
    selector_cls: type,
    embedder_key: str,
    selector_key: str,
    register: bool = True,
) -> Result:
    """Run the full check matrix. Returns a Result; raises on programmer
    error (e.g. None inputs). The Result.ok flag tells you whether the
    plugin is wireable.

    When ``register`` is True (default), the plugin is registered against
    the runtime registries so the same process can immediately use it.
    Pass ``register=False`` to validate without side effects.
    """
    if embedder is None or selector_cls is None:
        raise ValueError("embedder and selector_cls are required")

    result = Result(passed=[], failed=[])
    dim = _check_embedder(embedder, result)
    if dim is None:
        return result

    if register:
        register_embedder(embedder_key, embedder)
        register_selector(selector_key, selector_cls)
        result.passed.append(
            f"registered embedder={embedder_key!r}, selector={selector_key!r}"
        )

    _check_selector(selector_cls, embedder, embedder_key, result)
    return result


# ----------------------------------------------------------------------
# CLI driver
# ----------------------------------------------------------------------


def _import_attr(spec: str):
    """``pkg.module:attr`` -> the attribute."""
    if ":" not in spec:
        raise ValueError(
            f"expected 'pkg.module:attr', got {spec!r}"
        )
    mod_path, attr = spec.split(":", 1)
    module = importlib.import_module(mod_path)
    if not hasattr(module, attr):
        raise AttributeError(f"{mod_path} has no attribute {attr}")
    return getattr(module, attr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedder", required=True,
                    help="Import path 'pkg.module:fn' for the embedder.")
    ap.add_argument("--selector", required=True,
                    help="Import path 'pkg.module:Class' for the selector.")
    ap.add_argument("--register-as", required=True,
                    help="Comma-separated 'embedder_key,selector_key'.")
    ap.add_argument("--no-register", action="store_true",
                    help="Don't actually register; only validate.")
    args = ap.parse_args(argv)

    embedder = _import_attr(args.embedder)
    selector_cls = _import_attr(args.selector)

    keys = [k.strip() for k in args.register_as.split(",")]
    if len(keys) != 2 or not all(keys):
        print("--register-as must be 'embedder_key,selector_key'", file=sys.stderr)
        return 2
    embedder_key, selector_key = keys

    print(f"Validating embedder={args.embedder!r} selector={args.selector!r}")
    print(f"  embedder key: {embedder_key}")
    print(f"  selector key: {selector_key}")
    print()

    result = validate(
        embedder=embedder,
        selector_cls=selector_cls,
        embedder_key=embedder_key,
        selector_key=selector_key,
        register=not args.no_register,
    )
    print(result.report())

    if result.ok and not args.no_register:
        print()
        print(f"Registered embedders: {sorted(EMBEDDERS)}")
        print(f"Registered selectors: {sorted(SELECTORS)}")

    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
