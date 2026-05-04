# Plugins: custom embedders and selectors

Radar supports two extension points: an **embedder** (how seed papers and
candidates are turned into vectors) and a **selector** (how candidates
are scored against a profile's seeds). Both are looked up by string key
on a per-profile basis, so different profiles can run different
combinations without code changes.

This document covers:
1. Writing an embedder.
2. Writing a selector.
3. Registering both.
4. Validating with `validate_plugins.py`.
5. Picking them at profile creation.

---

## 1. Writing an embedder

An embedder is **a function** with this signature:

```python
def my_embed(text: str) -> list[float]:
    ...
```

Requirements:
- Returns the same length vector for every input. The length is the
  embedding dimension and must not change across calls.
- The output is JSON-serializable (a `list[float]`).
- Pure on input — no global mutation. Caching expensive model loads
  internally is fine and recommended.

Minimal example:

```python
import hashlib
import numpy as np

def hash_embed(text: str, *, dim: int = 64) -> list[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    v = rng.standard_normal(dim)
    v /= np.linalg.norm(v) + 1e-9
    return v.tolist()
```

For a real model, load it lazily and cache it on first call so
importing the module stays cheap:

```python
_model = None

def _load():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("my-org/my-model")
    return _model

def my_embed(text: str) -> list[float]:
    return _load().encode(text, normalize_embeddings=True).tolist()
```

Built-ins live in `rag_lib/embedders.py` — see `placeholder_embed` and
`specter2_embed` for reference.

---

## 2. Writing a selector

A selector is **a class** with these methods (this is the `Selector`
protocol from `rag_lib/selector.py`):

```python
class MySelector:
    name: str  # stable string id, e.g. "my_team_v1"

    def __init__(self, *, embedding_model: str | None = None,
                 threshold: float | None = None, **state):
        ...

    def fit(self, profile: Profile) -> None:
        """Fit from profile.seed_embeddings(profile.embedding_model)."""

    def select(
        self,
        candidates: list[Paper],
        profile: Profile,
        threshold: float | None = None,
    ) -> list[tuple[float, Paper, dict]]:
        """Return (score, paper, breakdown) sorted descending by score.
        ``breakdown`` is a dict — at minimum {"score_raw": float}."""

    def diagnostics(self) -> dict:
        """Health metrics. Empty dict is fine before fit()."""

    def config(self) -> dict:
        """Serializable state. MUST include {"type": self.name}."""

    @classmethod
    def from_config(cls, config: dict) -> "MySelector":
        """Round-trip from config()."""

    def cost(self) -> dict:
        """{"wall_seconds": float} for the most recent fit/select."""
```

A few rules that downstream code depends on:

- **Score range.** Whatever `select()` returns as the first tuple field
  is what the wizard's threshold slider compares against. The built-in
  selectors emit raw cosine in `[-1, 1]`. Your selector can use any
  range, but pick one and stick with it — the wizard sweep
  (`0.50…0.95`) is calibrated against cosine.
- **Sorting.** `select()` must return entries sorted descending by the
  primary score.
- **Embedding lookup.** Use `profile.seed_embeddings(profile.embedding_model)`
  to read seed vectors. For candidates that are missing a vector under
  that key, embed on the fly via:

  ```python
  from rag_lib.embedders import get_embedder
  from rag_lib.embed import build_embedding_input
  embedder = get_embedder(profile.embedding_model)
  vec = embedder(build_embedding_input(paper))
  ```
- **`config()` round-trip.** `from_config(s.config())` must reconstruct
  an equivalent selector. The `"type"` key is what the registry
  dispatches on at scheduler boot, so include it.

A complete reference implementation: `rag_lib/selectors/centroid.py`.
A minimal random-scoring reference: `tests/fixture_selector.py`.

---

## 3. Registration

Both extension points use the same shape:

```python
from rag_lib.embedders import register_embedder
from rag_lib.selectors import register_selector

from my_pkg.my_embed import my_embed
from my_pkg.my_selector import MySelector

register_embedder("my-team-v1", my_embed)
register_selector("my_team_v1", MySelector)
```

**Where to put these calls.** Put them in your package's top-level
`__init__.py` so importing your package once at process start is enough.
For the FastAPI service, that means importing your package in
`rag_lib/api/app.py` (or via a `RADAR_PLUGINS` env var if your
deployment prefers indirection — wire up to taste).

**Naming.**
- Embedder keys are user-facing (they show up in JSON and the API);
  use the same convention as `placeholder-v1` / `specter2`.
- Selector keys go through the database (`selector_config_json["type"]`);
  use lowercase-with-underscores like `centroid` / `max_seed`.

**Validation timing.** Both `register_embedder` and `register_selector`
overwrite an existing key without warning. Pick a unique prefix per
team (e.g. `myteam-`) to avoid collisions.

---

## 4. Validate before shipping

Run `tests/validate_plugins.py` against your embedder + selector — it's
a self-contained script that exercises the full protocol surface and
prints a pass/fail summary:

```bash
python -m tests.validate_plugins \
    --embedder    my_pkg.my_embed:my_embed \
    --selector    my_pkg.my_selector:MySelector \
    --register-as my-team-v1,my_team_v1
```

Or import and call from your own test suite:

```python
from tests.validate_plugins import validate

validate(embedder=my_embed, selector_cls=MySelector,
         embedder_key="my-team-v1", selector_key="my_team_v1")
```

The script asserts:
- The embedder returns a non-empty `list[float]` and is dimension-stable.
- The selector instantiates, `isinstance(s, Selector)` passes, and
  `from_config(config())` round-trips.
- `fit()` runs against a synthetic 4-paper profile, `select()` returns
  `(score, paper, breakdown)` triples sorted descending, and
  `threshold` filters.
- `diagnostics()` and `cost()` return dicts of the expected shape.

If everything passes, you can register and ship. If not, the failing
assertion names exactly which protocol clause is missing.

---

## 5. Selecting per-profile

The HTTP API accepts both keys at draft creation:

```http
POST /api/profiles/draft
Content-Type: application/json

{
  "name": "Neonatal HRV",
  "embedding_model": "my-team-v1",
  "selector": "my_team_v1"
}
```

Both fields are optional:
- `embedding_model` defaults to `RADAR_DEFAULT_EMBEDDING_MODEL` from
  settings (`specter2` out of the box).
- `selector` defaults to `centroid`.

Unknown keys return **HTTP 400** with the registry error
("Unknown selector 'foo'. Known: ['centroid', 'max_seed', ...]"). The
selector choice is persisted on the draft as
`selector_config_json={"type": <key>}` so the dry-run + commit steps
fit the right class. The embedding model key is persisted on
`profiles.embedding_model` and read by every downstream consumer
(vault upload, chat retrieval, scheduler refit).

CLI equivalents:

```bash
# pick a selector for the offline gather CLI
python -m cli.gather --profile … --selector my_team_v1 --out runs.json

# build a profile with a custom embedder programmatically
from rag_lib.embedders import register_embedder
from rag_lib.profile import Profile

register_embedder("my-team-v1", my_embed)
profile = Profile.from_csv(
    "seeds.csv", name="Demo", openalex_client=client,
    embedder=my_embed, embedding_model="my-team-v1",
)
```

---

## Reference: file layout

| File | What lives here |
|------|------------------|
| `rag_lib/embedders.py`               | Embedder registry, built-ins, `register_embedder` |
| `rag_lib/selector.py`                | `Selector` protocol — the contract |
| `rag_lib/selectors/__init__.py`      | Selector registry, `register_selector`, `selector_from_config` |
| `rag_lib/selectors/centroid.py`      | Reference selector |
| `rag_lib/selectors/max_seed.py`      | Reference selector |
| `tests/validate_plugins.py`          | Validation harness |
| `tests/test_selector_protocol.py`    | Internal protocol tests (the harness mirrors these) |
