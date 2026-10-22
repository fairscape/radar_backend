"""Research topics for an interest, generated the way Prosopia generates them.

Nathan Sheffield (Prosopia) suggested feeding RADAR's recommendation path
with the topics his pipeline produces. This module runs *his* code -- the
``researcher_profiles`` SDK RADAR already depends on -- rather than a
re-implementation, so the topics match what Prosopia would produce:

- An interest backed by a Prosopia profile reads that profile's own
  ``research_interests`` / ``expertise`` / ``subfields`` / ``not_interests``.
- Any other interest becomes a *lite* RP profile (the SDK's format: a
  ``profile.jsonld`` plus ``sources/papers.jsonld`` of its seeds), the SDK
  builds its embedding index (``st:all-MiniLM-L6-v2``), and topics come
  from the SDK's topic extraction: the LLM method when the user supplied an
  Anthropic key for this one call, else the SDK's cluster method.

Two deliberate departures from calling ``prof.topics.get(method="llm")``:

1. **The key is passed per call.** The SDK's ``_topics_llm`` builds
   ``LLMClient()`` with no key, so the client falls back to the
   process-wide ``ANTHROPIC_API_KEY``. Setting that per request would hand
   one user's key to every other request in flight, and the user asked for
   a key that is typed in for each run and stored nowhere. So
   :func:`_llm_topics` repeats ``_topics_llm`` -- its prompt verbatim, its
   sampling (seed 0, 10 chunks, 400 characters), its JSON parsing and its
   label-to-chunk mapping -- with ``LLMClient(api_key=...)``. A PR adding
   ``api_key`` to ``topics.get`` upstream would let this call the SDK
   directly.
2. **Topics map back to papers through their chunks.** The SDK collects a
   topic's ``paper_ids`` only from ``paper_summary`` chunks, and a lite
   profile has ``paper_abstract`` chunks, so every topic came back with no
   papers. The chunk ids the SDK returns are mapped to papers here.

The MiniLM vectors stay inside the SDK's index: RADAR scores with SPECTER2
and only the topic *text* is used (as reranker queries, see
``rerankers/medcpt.py``).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import structlog

log = structlog.get_logger("rag_lib.interest_terms")

BACKEND_SPEC = "st:all-MiniLM-L6-v2"
N_TOPICS = 8
_HASH_FILE = ".radar_seed_hash"


class TermsUnavailable(RuntimeError):
    """A package the SDK's topic extraction needs is not installed."""


class KeyRejected(RuntimeError):
    """Anthropic refused the key the user typed in."""


class LLMFailed(RuntimeError):
    """The LLM call failed for a reason other than the key."""


def _require_deps() -> None:
    missing = []
    for mod, pkg in (("sentence_transformers", "sentence-transformers"),
                     ("sqlite_vec", "sqlite-vec"), ("sklearn", "scikit-learn")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise TermsUnavailable("not installed: " + ", ".join(missing))


def seed_hash(seed_ids: list[str]) -> str:
    return hashlib.sha1("\n".join(sorted(seed_ids)).encode()).hexdigest()[:16]


def _paper_id(openalex_id: str) -> str:
    """The SDK's ``paper_id`` for a seed: the bare W id, or a slug of the local id."""
    tail = openalex_id.rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", tail)


# ---------------------------------------------------------------------------
# A. Prosopia-backed interests
# ---------------------------------------------------------------------------


def prosopia_terms(base_url: str, slug: str, *, timeout: float = 30.0) -> dict | None:
    """The profile's own topic fields, or None when it cannot be read.

    Read straight from the API's ``fields``: ``research_interests`` is not
    part of the SDK's ``ProfileDocument``. Anonymous reads see published
    profiles only; a private one returns None and the caller falls back to
    the generated path.
    """
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/v1/profiles/{slug}", timeout=timeout)
        if resp.status_code != 200:
            return None
        fields = resp.json().get("fields") or {}
    except Exception:  # noqa: BLE001 — unreadable means "use the other path"
        return None

    def labels(v: Any) -> list[str]:
        if isinstance(v, dict):
            v = v.get("top") or []
        out = []
        for item in v or []:
            text = item.get("label") if isinstance(item, dict) else item
            if text:
                out.append(str(text).replace("-", " ").strip())
        return out

    # The current API ("2026-09") lists research_interests as weighted
    # concepts, a negative weight meaning "not this" -- the shape
    # researcher_topics reads for the topic list. Read that way here too:
    # labels() found no interests in it, and would have queried the
    # unwanted ones had it found them.
    from . import researcher_topics

    ri = fields.get("research_interests")
    not_interests = labels(fields.get("not_interests"))
    if isinstance(ri, list) and any(isinstance(e, dict) and isinstance(e.get("concept"), dict) for e in ri):
        view = researcher_topics.read(fields)
        interests = [t["display_name"] for t in view["wanted"]] + view["wanted_labels"]
        not_interests = [t["display_name"] for t in view["unwanted"]] + view["unwanted_labels"]
    else:
        interests = labels(ri)
    terms = [{"label": t, "weight": None, "paper_ids": [], "kind": "research_interest"}
             for t in interests]
    terms += [{"label": t, "weight": None, "paper_ids": [], "kind": "expertise"}
              for t in labels(fields.get("expertise"))]
    terms += [{"label": t, "weight": None, "paper_ids": [], "kind": "subfield"}
              for t in labels(fields.get("subfields"))]
    # Never query what the owner said they do not want.
    unwanted = {n.lower() for n in not_interests}
    terms = [t for t in terms if t["label"].lower() not in unwanted]
    if not terms:
        return None
    return {"terms": terms, "not_interests": not_interests}


# ---------------------------------------------------------------------------
# B. The SDK pipeline over the interest's seeds
# ---------------------------------------------------------------------------


def build_lite_profile(root: Path, profile_row: sqlite3.Row, seeds: list[sqlite3.Row],
                       *, backend: Any = None):
    """Write the interest as a lite RP profile and build its index.

    Rebuilt from scratch when the seed set changed (the hash file), reused
    otherwise: the index is the slow part (~15 s with model load for 80
    seeds). Returns ``(profile, {paper_id: openalex_id})``.
    """
    from researcher_profiles import PaperRecord, ProfileDocument, ResearcherProfile

    want = seed_hash([s["openalex_id"] for s in seeds])
    ids = {_paper_id(s["openalex_id"]): s["openalex_id"] for s in seeds}
    hash_file = root / _HASH_FILE
    if root.exists() and hash_file.exists() and hash_file.read_text().strip() == want \
            and (root / ".cache" / "embeddings.sqlite").exists():
        return ResearcherProfile.from_files(root), ids

    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    slug = re.sub(r"[^a-z0-9-]+", "-", (profile_row["slug"] or "interest").lower()).strip("-") or "interest"
    rid = f"local:radar-{slug}-{hashlib.sha1(str(profile_row['id']).encode()).hexdigest()[:6]}"
    prof = ResearcherProfile.from_files(root)
    prof.save_profile(ProfileDocument(
        name=profile_row["name"] or slug, rid=rid, provenance="synthetic", level="lite",
    ))
    papers = []
    for s in seeds:
        oa = s["openalex_id"]
        # A seed with no abstract (an uploaded PDF) is described by the
        # start of its text, so it still gets a chunk in the index.
        abstract = (s["abstract"] or "").strip() or (s["body_text"] or "")[:1500].strip() or None
        papers.append(PaperRecord(
            name=s["title"] or oa,
            paper_id=_paper_id(oa),
            openalex_id=oa if oa.startswith("https://openalex.org/") else None,
            doi=s["doi"],
            abstract=abstract,
            venue=s["venue"],
            datePublished=str(s["year"]) if s["year"] else None,
        ))
    prof.save_papers(papers)
    prof = ResearcherProfile.from_files(root)
    prof.index.build(backend=backend if backend is not None else BACKEND_SPEC)
    hash_file.write_text(want)
    return prof, ids


def _chunk_sources(root: Path) -> dict[int, str]:
    """chunk id -> the SDK's ``source_id`` (our ``paper_id``)."""
    with sqlite3.connect(root / ".cache" / "embeddings.sqlite") as c:
        return {int(r[0]): r[1] for r in c.execute("SELECT id, source_id FROM chunks")}


def _papers_for(chunk_ids: list[int], sources: dict[int, str], ids: dict[str, str]) -> list[str]:
    return sorted({ids[sources[c]] for c in chunk_ids if sources.get(c) in ids})


def _llm_topics(prof, n: int, api_key: str, backend: Any) -> tuple[list[dict], dict]:
    """``researcher_profiles.profile.topics._topics_llm`` with the key passed in.

    Prompt, sampling, parsing and label mapping are the SDK's (0.3.0,
    unchanged at upstream e78fe77). Raises KeyRejected / LLMFailed instead
    of quietly falling back to clustering, so the user is told which ran.
    """
    import random

    import anthropic
    import numpy as np
    from researcher_profiles.generative.llm import LLMClient
    from researcher_profiles.profile.topics import _load_all_chunk_vectors

    mat, metas = _load_all_chunk_vectors(prof)
    if not metas:
        return [], {}
    random.seed(0)
    sample = random.sample(metas, min(10, len(metas)))
    joined = "\n\n---\n\n".join((m["text"] or "")[:400] for m in sample)
    prompt = (
        f"Read these chunks of a researcher's writing and identify the top {n} "
        "research topics. Respond with strict JSON: a list of objects with keys "
        "'label' (short topic name) and 'weight' (float in [0,1]).\n\n"
        f"{joined}"
    )
    try:
        resp = LLMClient(api_key=api_key).complete(
            system=[{"type": "text", "text": "You extract concise research topics."}],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
        )
    # ``from None`` throughout: the provider's exception (and its traceback,
    # which holds the client and its headers) is not chained onto ours.
    except anthropic.AuthenticationError:
        raise KeyRejected("Anthropic rejected the API key") from None
    except anthropic.PermissionDeniedError:
        raise KeyRejected("Anthropic refused this key for the model") from None
    except Exception as exc:  # noqa: BLE001 — never echo the provider's message
        raise LLMFailed(f"the LLM call failed ({type(exc).__name__})") from None

    m = re.search(r"\[.*\]", (resp.text or "").strip(), re.DOTALL)
    try:
        data = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, list):
        raise LLMFailed("the LLM did not return a JSON list of topics")
    data = [d for d in data if isinstance(d, dict) and d.get("label")]
    if not data:
        raise LLMFailed("the LLM returned no topic labels")

    labels = [str(d["label"]) for d in data]
    label_vecs = np.array(backend.embed(labels), dtype=np.float32)
    norms = np.linalg.norm(label_vecs, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    sims = (label_vecs / norms) @ mat.T
    out = []
    for i, d in enumerate(data[: len(labels)]):
        top_idx = np.argsort(-sims[i])[:5]
        out.append({
            "label": labels[i],
            "weight": float(d.get("weight") or 0.0),
            "chunk_ids": [int(metas[int(j)]["id"]) for j in top_idx],
        })
    usage = dict(resp.usage or {})
    usage["model"] = resp.model
    return out[:n], usage


def generate(conn: sqlite3.Connection, settings: Any, profile_row: sqlite3.Row, *,
             api_key: str | None, backend: Any = None) -> dict:
    """Generate (and return) the interest's topic record. Writes nothing to the DB.

    ``api_key`` is the user's Anthropic key for this call only; it is passed
    to the client and dropped. Without it the SDK's cluster method runs.
    """
    pid = int(profile_row["id"])
    seeds = conn.execute(
        """
        SELECT p.openalex_id, p.title, p.abstract, p.body_text, p.doi, p.year, p.venue
        FROM profile_seeds s JOIN papers p USING (openalex_id)
        WHERE s.profile_id = ?
        """,
        (pid,),
    ).fetchall()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record: dict[str, Any] = {"generated_at": now, "seed_hash": seed_hash([s["openalex_id"] for s in seeds]),
                              "not_interests": [], "edited": False}

    # A. The researcher's own Prosopia profile.
    rrow = conn.execute(
        "SELECT r.source, r.key, r.base_url FROM profiles p JOIN researchers r ON r.id = p.researcher_id WHERE p.id = ?",
        (pid,),
    ).fetchone()
    if rrow is not None and rrow["source"] == "prosopia":
        found = prosopia_terms(rrow["base_url"] or settings.RADAR_PROSOPIA_BASE_URL, rrow["key"])
        if found:
            log.info("interest_terms.prosopia", profile_id=pid, n=len(found["terms"]))
            return {**record, "source": "prosopia", "model": None, **found}

    if not seeds:
        raise ValueError("this interest has no seeds to describe")
    _require_deps()
    from researcher_profiles.embeddings.backends import get_backend

    be = backend if backend is not None else get_backend(BACKEND_SPEC)
    root = Path(getattr(settings, "RADAR_RP_PROFILES_DIR", None) or Path(settings.RADAR_VAULT_DIR).parent / "rp_profiles") / str(pid)
    t0 = time.monotonic()
    prof, ids = build_lite_profile(root, profile_row, seeds, backend=be)
    sources = _chunk_sources(root)

    if api_key:
        topics, usage = _llm_topics(prof, N_TOPICS, api_key, be)
        record.update(source="rp-sdk-llm", model=usage.pop("model", None), usage=usage)
    else:
        sdk_topics = prof.topics.get(n=N_TOPICS, method="cluster")
        topics = [{"label": t.label, "weight": t.weight, "chunk_ids": list(t.chunk_ids or [])} for t in sdk_topics]
        record.update(source="rp-sdk-cluster", model=None)
    record["terms"] = [
        {"label": t["label"], "weight": round(float(t["weight"] or 0.0), 4),
         "paper_ids": _papers_for(t["chunk_ids"], sources, ids)}
        for t in topics
    ]
    log.info("interest_terms.generated", profile_id=pid, source=record["source"],
             n=len(record["terms"]), elapsed_s=round(time.monotonic() - t0, 1),
             usage=record.get("usage"))
    return record


def load(profile_row: sqlite3.Row) -> dict | None:
    try:
        raw = profile_row["interest_terms_json"]
    except (IndexError, KeyError):
        return None
    return json.loads(raw) if raw else None


def save(conn: sqlite3.Connection, profile_id: int, record: dict | None) -> None:
    conn.execute(
        "UPDATE profiles SET interest_terms_json = ?, updated_at = datetime('now') WHERE id = ?",
        (json.dumps(record) if record is not None else None, profile_id),
    )
    conn.commit()


def query_labels(record: dict | None) -> list[str]:
    """The labels the reranker uses as queries, in stored order."""
    return [t["label"] for t in (record or {}).get("terms") or [] if t.get("label")]
