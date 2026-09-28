"""Seed a draft from a Researcher Profile document (``profile.jsonld``).

The databio Researcher Profile format (https://village.databio.org/
researcher-profiles/rp-spec/) describes one person as a ``schema:Person``
extended with ``rid`` (an ORCID or a minted ``local:`` id), ``expertise``,
``not_interests``, a summary, and a manifest of files. The wizard's
"From Profile" path accepts just that document — pasted or uploaded —
and uses what RADAR can:

* **Identity.** The ORCID (``rid`` / ``@id``) and the OpenAlex author id
  (``identifier``) feed the existing ORCID import
  (:func:`orcid_import.run_import`), which fetches the works from OpenAlex,
  cross-checks them against the ORCID registry, embeds the lead-author
  papers and aggregates the concept list. Nothing here re-implements that.
* **Signal.** ``expertise`` and ``not_interests`` are short phrases the
  researcher (or their builder) declared. Each is embedded with the same
  model and searched against the same OpenAlex topic index the UMLS
  mapper uses; a hit turns the matching concept on (expertise) or off
  (not_interests), and expertise topics the corpus never surfaced are
  appended with ``source="rp_expertise"``.
* **Display.** Summary, affiliation, field, collaborators, level and
  provenance are kept verbatim on ``profiles.rp_meta_json``.

A profile without an ORCID cannot fetch anything: the draft is created
with the metadata attached and the user uploads PDFs as seeds; the
expertise / not_interests signal is applied at Step 3 instead
(:func:`apply_rp_signals_from_meta`).

Papers named in the manifest are *not* in the document (only their
relative paths are), so nothing here reads ``papers.jsonld``.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any

import structlog

from ...db.repos import profiles as profiles_repo
from ..schemas import OrcidImportResult
from . import orcid_import

log = structlog.get_logger(__name__)

DATABIO_CONTEXT = "https://profiles.databio.org/context/v1.jsonld"
_ORCID_IN_IRI = re.compile(r"orcid\.org/(\d{4}-\d{4}-\d{4}-\d{3}[\dXx])\b")
_OPENALEX_AUTHOR = re.compile(r"^(?:https?://openalex\.org/)?(A\d+)$", re.I)


class RpProfileError(ValueError):
    """The text is not a usable Researcher Profile document (→ 422)."""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def deslug(phrase: str) -> str:
    """``"atac-seq-pipelines"`` → ``"atac seq pipelines"`` for embedding and display."""
    return re.sub(r"[-_]+", " ", (phrase or "").strip()).strip()


@dataclass
class RpProfile:
    name: str
    orcid: str | None
    rid: str | None
    openalex_author_id: str | None
    level: str | None
    provenance: str | None
    date_modified: str | None
    affiliation: str | None
    field: str | None
    summary: str | None
    expertise: list[str] = field(default_factory=list)
    not_interests: list[str] = field(default_factory=list)
    collaborators: list[str] = field(default_factory=list)
    same_as: list[str] = field(default_factory=list)
    paper_stats: dict | None = None
    warnings: list[str] = field(default_factory=list)

    def meta(self) -> dict:
        """What goes on ``profiles.rp_meta_json`` (only keys that carry data)."""
        out: dict[str, Any] = {"source": "rp_profile", "name": self.name}
        for k in ("rid", "orcid", "openalex_author_id", "level", "provenance",
                  "date_modified", "affiliation", "field", "summary", "paper_stats"):
            v = getattr(self, k)
            if v not in (None, "", {}):
                out[k] = v
        for k in ("expertise", "not_interests", "collaborators", "same_as"):
            v = getattr(self, k)
            if v:
                out[k] = list(v)
        return out


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for v in value:
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
        elif isinstance(v, dict):
            n = v.get("name")
            if isinstance(n, str) and n.strip():
                out.append(n.strip())
    return out


def parse_profile_dict(doc: Any) -> RpProfile:
    """Validate and extract a Researcher Profile from its parsed JSON."""
    if not isinstance(doc, dict):
        raise RpProfileError("profile document must be a JSON object")
    warnings: list[str] = []

    types = doc.get("@type")
    types = types if isinstance(types, list) else [types]
    if not any(isinstance(t, str) and t.split(":")[-1] == "Person" for t in types):
        raise RpProfileError('profile document must have "@type": "Person"')

    name = doc.get("name")
    if not isinstance(name, str) or not name.strip():
        raise RpProfileError('profile document has no "name"')
    name = name.strip()

    ctx = doc.get("@context")
    if ctx != DATABIO_CONTEXT and doc.get("conformsTo") != DATABIO_CONTEXT:
        warnings.append(
            f"unrecognised @context {ctx!r}; read as a databio Researcher Profile anyway"
        )

    rid = doc.get("rid") if isinstance(doc.get("rid"), str) else None
    orcid: str | None = None
    for candidate in (rid, doc.get("@id")):
        if not isinstance(candidate, str):
            continue
        m = _ORCID_IN_IRI.search(candidate)
        raw = m.group(1) if m else candidate
        try:
            orcid = orcid_import.normalize_orcid(raw)
            break
        except ValueError:
            continue
    if orcid is None and rid and rid.startswith("local:"):
        warnings.append("profile has a local: identity (no ORCID)")

    openalex_author_id: str | None = None
    for ident in doc.get("identifier") or []:
        if not isinstance(ident, dict):
            continue
        if str(ident.get("propertyID") or "").lower() == "openalex":
            m = _OPENALEX_AUTHOR.match(str(ident.get("value") or "").strip())
            if m:
                openalex_author_id = f"https://openalex.org/{m.group(1).upper()}"
                break

    def _opt_str(key: str) -> str | None:
        v = doc.get(key)
        return v.strip() if isinstance(v, str) and v.strip() else None

    same_as = doc.get("sameAs")
    same_as = [same_as] if isinstance(same_as, str) else _as_str_list(same_as)
    paper_stats = doc.get("paper_stats") if isinstance(doc.get("paper_stats"), dict) else None

    return RpProfile(
        name=name,
        orcid=orcid,
        rid=rid,
        openalex_author_id=openalex_author_id,
        level=_opt_str("level"),
        provenance=_opt_str("provenance"),
        date_modified=_opt_str("dateModified"),
        affiliation=_opt_str("affiliation"),
        field=_opt_str("field"),
        summary=_opt_str("summary"),
        expertise=_as_str_list(doc.get("expertise")),
        not_interests=_as_str_list(doc.get("not_interests")),
        collaborators=_as_str_list(doc.get("collaborators")),
        same_as=same_as,
        paper_stats=paper_stats,
        warnings=warnings,
    )


def parse_profile(text: str) -> RpProfile:
    """Parse the text of a ``profile.jsonld``; :class:`RpProfileError` when unusable."""
    if not isinstance(text, str) or not text.strip():
        raise RpProfileError("empty profile document")
    if len(text) > 2_000_000:
        raise RpProfileError("profile document is too large (2 MB limit)")
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RpProfileError(f"profile document is not valid JSON: {exc.msg} (line {exc.lineno})") from exc
    return parse_profile_dict(doc)


# ---------------------------------------------------------------------------
# Phrase → OpenAlex topic mapping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhraseHit:
    phrase: str        # the profile's phrase, as written
    topic_id: str      # https://openalex.org/T…
    display_name: str
    similarity: float
    subfield: str | None = None
    field: str | None = None
    domain: str | None = None

    def to_dict(self) -> dict:
        return {
            "phrase": self.phrase, "topic_id": self.topic_id, "display_name": self.display_name,
            "similarity": self.similarity, "subfield": self.subfield, "field": self.field,
            "domain": self.domain,
        }


def map_phrases_to_topics(
    phrases: list[str],
    *,
    settings,
    min_sim: float,
    top_k: int,
) -> tuple[list[PhraseHit], list[str]]:
    """Embed each phrase and find its nearest OpenAlex topics.

    Uses the UMLS mapper's model (``RADAR_UMLS_EMBEDDING_MODEL``, mxbai
    through ollama) and its precomputed topic index
    (``RADAR_UMLS_CACHE_DIR``), so the two signals live in one vector
    space. Best-effort: when the index or the embedder is unavailable the
    result is empty and the reason is returned as a warning, never an
    exception — the import must not fail because ollama is down.
    """
    phrases = [p for p in (phrases or []) if p and p.strip()]
    if not phrases:
        return [], []
    try:
        import numpy as np

        from ...embedders import get_embedder
        from ...umls.cache import get_topic_index

        embedder = get_embedder(settings.RADAR_UMLS_EMBEDDING_MODEL)
        index = get_topic_index(settings.RADAR_UMLS_CACHE_DIR)
    except Exception as exc:  # noqa: BLE001 — degrade, do not fail the import
        log.warning("rp_import.topic_index_unavailable", error=str(exc))
        return [], [f"expertise mapping skipped: {type(exc).__name__}: {exc}"]

    hits: list[PhraseHit] = []
    warnings: list[str] = []
    for phrase in phrases:
        try:
            vec = np.asarray(embedder(deslug(phrase)), dtype=np.float32)
            matches = index.search(vec, top_k=top_k, min_similarity=min_sim)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"could not map {phrase!r}: {type(exc).__name__}: {exc}")
            continue
        for topic, sim in matches:
            def _name(block: Any) -> str | None:
                return block.get("display_name") if isinstance(block, dict) else (block or None)
            hits.append(PhraseHit(
                phrase=phrase, topic_id=topic["id"], display_name=topic.get("display_name") or "",
                similarity=round(float(sim), 4),
                subfield=_name(topic.get("subfield")), field=_name(topic.get("field")),
                domain=_name(topic.get("domain")),
            ))
    hits.sort(key=lambda h: -h.similarity)
    return hits, warnings


def hits_from_dicts(rows: list[dict] | None) -> list[PhraseHit]:
    out: list[PhraseHit] = []
    for r in rows or []:
        try:
            out.append(PhraseHit(
                phrase=r["phrase"], topic_id=r["topic_id"], display_name=r.get("display_name") or "",
                similarity=float(r.get("similarity") or 0.0),
                subfield=r.get("subfield"), field=r.get("field"), domain=r.get("domain"),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Applying the signal to a concept list
# ---------------------------------------------------------------------------


def apply_rp_signals(
    topic_filters: dict,
    *,
    expertise_hits: list[PhraseHit],
    not_interest_hits: list[PhraseHit],
    max_added_per_phrase: int = 1,
    add_min_sim: float = 0.0,
) -> tuple[dict, dict]:
    """Switch concepts on/off from the profile's declared (non-)interests.

    Rules:

    1. A concept hit by a ``not_interests`` phrase is switched **off** and
       tagged ``rp_off_by`` — unless an ``expertise`` phrase hits the same
       concept with a *higher* cosine, in which case the stronger declared
       signal wins and the concept stays on.
    2. A concept already in the list hit by an ``expertise`` phrase is
       switched **on** (even when no seed paper carries it) and tagged
       ``rp_on_by``.
    3. An ``expertise`` hit the corpus never surfaced is **appended** with
       ``source="rp_expertise"``, ``count=0``, ``seed_papers=0``, ``on=True``,
       but only when its cosine is >= ``add_min_sim`` and at most
       ``max_added_per_phrase`` per phrase (best first). Every added concept
       becomes an OpenAlex query at gather time, hence the separate floor.

    Returns ``(new_topic_filters, summary)``; the input is not modified.
    ``summary`` is what the import result reports.
    """
    out = json.loads(json.dumps(topic_filters)) if topic_filters else {"topics": []}
    topics: list[dict] = out.setdefault("topics", [])
    # Stored lists carry OpenAlex IRIs (https://openalex.org/T…) but some
    # producers (and the test corpus) use the bare id; match on the bare form.
    def bare(tid: str) -> str:
        return str(tid).rsplit("/", 1)[-1]
    by_id: dict[str, dict] = {bare(t["id"]): t for t in topics if t.get("id")}

    # Best expertise cosine per concept, to arbitrate conflicts.
    best_expertise: dict[str, float] = {}
    for h in expertise_hits:
        k = bare(h.topic_id)
        if h.similarity > best_expertise.get(k, -1.0):
            best_expertise[k] = h.similarity

    switched_off: list[dict] = []
    overruled: list[dict] = []
    for h in not_interest_hits:
        t = by_id.get(bare(h.topic_id))
        if t is None:
            continue
        if best_expertise.get(bare(h.topic_id), -1.0) > h.similarity:
            overruled.append({"id": t["id"], "display_name": t.get("display_name"),
                              "by": h.phrase, "similarity": h.similarity,
                              "kept_by_expertise": best_expertise[bare(h.topic_id)]})
            continue
        t["on"] = False
        t.setdefault("rp_off_by", [])
        if h.phrase not in t["rp_off_by"]:
            t["rp_off_by"].append(h.phrase)
        if not any(bare(x["id"]) == bare(h.topic_id) for x in switched_off):
            switched_off.append({"id": t["id"], "display_name": t.get("display_name"),
                                 "by": h.phrase, "similarity": h.similarity})
    off_ids = {bare(x["id"]) for x in switched_off}

    switched_on: list[dict] = []
    added: list[dict] = []
    added_per_phrase: dict[str, int] = {}
    for h in expertise_hits:
        if bare(h.topic_id) in off_ids:
            continue  # not_interests win
        t = by_id.get(bare(h.topic_id))
        if t is not None:
            was_on = bool(t.get("on", True))
            t["on"] = True
            t.setdefault("rp_on_by", [])
            if h.phrase not in t["rp_on_by"]:
                t["rp_on_by"].append(h.phrase)
            if not was_on and not any(bare(x["id"]) == bare(h.topic_id) for x in switched_on):
                switched_on.append({"id": t["id"], "display_name": t.get("display_name"),
                                    "by": h.phrase, "similarity": h.similarity})
            continue
        if h.similarity < add_min_sim or added_per_phrase.get(h.phrase, 0) >= max_added_per_phrase:
            continue
        added_per_phrase[h.phrase] = added_per_phrase.get(h.phrase, 0) + 1
        entry = {
            "id": h.topic_id, "display_name": h.display_name, "count": 0, "seed_papers": 0,
            "on": True, "source": "rp_expertise", "similarity": h.similarity,
            "rp_on_by": [h.phrase],
        }
        for k in ("subfield", "field", "domain"):
            v = getattr(h, k)
            if v:
                entry[k] = v
        topics.append(entry)
        by_id[bare(h.topic_id)] = entry
        added.append({"id": h.topic_id, "display_name": h.display_name,
                      "by": h.phrase, "similarity": h.similarity})

    out["rp_signals"] = {
        "expertise_hits": [h.to_dict() for h in expertise_hits],
        "not_interest_hits": [h.to_dict() for h in not_interest_hits],
        "switched_off": switched_off, "switched_on": switched_on, "added": added,
        "overruled": overruled,
    }
    summary = {
        "n_expertise_hits": len(expertise_hits),
        "n_not_interest_hits": len(not_interest_hits),
        "switched_off": switched_off,
        "switched_on": switched_on,
        "added": added,
        "overruled": overruled,
    }
    return out, summary


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def persist_meta(conn: sqlite3.Connection, profile_id: int, meta: dict, *, researcher_name: str | None) -> None:
    conn.execute(
        """
        UPDATE profiles
           SET rp_meta_json = ?, researcher_name = COALESCE(?, researcher_name),
               updated_at = datetime('now')
         WHERE id = ?
        """,
        (json.dumps(meta, ensure_ascii=False), researcher_name, profile_id),
    )
    conn.commit()


def load_meta(conn: sqlite3.Connection, profile_id: int) -> dict | None:
    row = conn.execute("SELECT rp_meta_json FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    if row is None or not row[0]:
        return None
    try:
        data = json.loads(row[0])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _signal_hits(meta: dict, settings) -> tuple[list[PhraseHit], list[PhraseHit], list[str], bool]:
    """Expertise / not_interest hits for a profile, computed once and cached in ``meta["mapped"]``."""
    cached = meta.get("mapped")
    if isinstance(cached, dict) and "expertise" in cached and "not_interests" in cached:
        return hits_from_dicts(cached.get("expertise")), hits_from_dicts(cached.get("not_interests")), [], False
    ex, w1 = map_phrases_to_topics(
        meta.get("expertise") or [], settings=settings,
        min_sim=settings.RADAR_RP_EXPERTISE_MIN_SIM, top_k=settings.RADAR_RP_EXPERTISE_TOP_K,
    )
    # A not_interest switches off only its single best concept: with the
    # compressed mxbai cosines, the 2nd/3rd neighbours of "phylogenomics"
    # are the researcher's core genomics topics.
    ni, w2 = map_phrases_to_topics(
        meta.get("not_interests") or [], settings=settings,
        min_sim=settings.RADAR_RP_NOT_INTEREST_MIN_SIM, top_k=1,
    )
    warnings = w1 + w2
    # Cache only a successful mapping; a degraded run (ollama down) retries next time.
    ok = not warnings
    if ok:
        meta["mapped"] = {"expertise": [h.to_dict() for h in ex], "not_interests": [h.to_dict() for h in ni]}
    return ex, ni, warnings, ok


def apply_rp_signals_from_meta(
    conn: sqlite3.Connection, settings, *, profile_id: int, topic_filters: dict,
) -> dict:
    """Wizard hook: apply a stored profile's signal to a freshly aggregated concept list.

    Used for profiles seeded "From Profile" whose seeds come from PDFs (no
    ORCID import ran). No-op when the profile has no ``rp_meta_json``.
    """
    meta = load_meta(conn, profile_id)
    if not meta or not (meta.get("expertise") or meta.get("not_interests")):
        return topic_filters
    ex, ni, warnings, freshly_mapped = _signal_hits(meta, settings)
    if freshly_mapped:
        persist_meta(conn, profile_id, meta, researcher_name=None)
    if warnings:
        log.warning("rp_import.signal_degraded", profile_id=profile_id, warnings=warnings)
    out, _summary = apply_rp_signals(
        topic_filters, expertise_hits=ex, not_interest_hits=ni,
        max_added_per_phrase=1, add_min_sim=settings.RADAR_RP_EXPERTISE_ADD_MIN_SIM,
    )
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _step(reporter, name: str, **kw) -> None:
    if reporter is not None:
        reporter.step(name, **kw)


def resolve_author_for_profile(client, profile: RpProfile) -> orcid_import.AuthorInfo | None:
    """OpenAlex author for the profile: by ORCID first, then by the profile's OpenAlex id."""
    if profile.orcid is None:
        return None
    author = orcid_import.resolve_author(client, profile.orcid)
    if author is not None or not profile.openalex_author_id:
        return author
    get_author = getattr(client, "get_author", None)
    if get_author is None:
        return None
    raw = orcid_import._with_retry(lambda: get_author(profile.openalex_author_id))
    if not raw:
        return None
    insts = raw.get("last_known_institutions") or []
    inst = insts[0] if insts else {}
    return orcid_import.AuthorInfo(
        orcid=profile.orcid,
        openalex_author_id=raw.get("id") or profile.openalex_author_id,
        display_name=(raw.get("display_name") or profile.name).strip(),
        institution=inst.get("display_name") or None,
        institution_ror=inst.get("ror") or None,
        topics=list(raw.get("topics") or []),
        works_count=raw.get("works_count"),
    )


def after_topics_hook(conn: sqlite3.Connection, profile_id: int):
    """The seed phase's post-processing for a profile-seeded draft, or None.

    Returns a callable ``(conn, profile_id, topic_filters) -> topic_filters``
    that applies the stored expertise / not_interests signal, when the
    profile row carries ``rp_meta_json``; ``None`` for plain ORCID drafts.
    """
    meta = load_meta(conn, profile_id)
    if not meta or not (meta.get("expertise") or meta.get("not_interests")):
        return None

    def _apply(c: sqlite3.Connection, pid: int, topic_filters: dict) -> dict:
        from ..settings import get_settings
        return apply_rp_signals_from_meta(c, get_settings(), profile_id=pid, topic_filters=topic_filters)

    return _apply


def run_profile_fetch(
    conn: sqlite3.Connection,
    settings,
    *,
    user_id: int,
    profile_id: int,
    slug: str,
    profile: RpProfile,
    name: str,
    client,
    reporter=None,
    author: orcid_import.AuthorInfo | None = None,
) -> OrcidImportResult:
    """Phase A for a profile with an ORCID: the ORCID fetch, with the profile
    metadata already on the row. The expertise / not_interests signal is
    applied by the seed phase (:func:`after_topics_hook`).

    Raises what :func:`orcid_import.run_fetch` raises. A profile without an
    ORCID must not reach here — the route handles that case without a run.
    """
    if profile.orcid is None:
        raise ValueError("profile has no ORCID; nothing to fetch from OpenAlex")

    _step(reporter, "parse_profile", message=f"Profile for {profile.name} ({profile.level or 'unknown level'})")
    if author is None:
        author = resolve_author_for_profile(client, profile)
    if author is None:
        raise LookupError(f"ORCID {profile.orcid} not found on OpenAlex")

    result = orcid_import.run_fetch(
        conn, settings,
        user_id=user_id, profile_id=profile_id, slug=slug,
        orcid=profile.orcid, name=name, client=client, reporter=reporter, author=author,
    )
    # run_fetch stores OpenAlex's display name; the profile's own name wins.
    persist_meta(conn, profile_id, profile.meta(), researcher_name=profile.name)
    result.warnings.extend(profile.warnings)
    result.rp = {
        "level": profile.level, "provenance": profile.provenance,
        "n_expertise": len(profile.expertise), "n_not_interests": len(profile.not_interests),
    }
    return result


def rp_summary_from_topics(topic_filters: dict, meta: dict | None) -> dict:
    """The ``rp`` block of a seed-phase result, read back from the stored
    ``rp_signals`` (written by :func:`apply_rp_signals`)."""
    sig = (topic_filters or {}).get("rp_signals") or {}
    meta = meta or {}
    return {
        "level": meta.get("level"), "provenance": meta.get("provenance"),
        "n_expertise": len(meta.get("expertise") or []),
        "n_not_interests": len(meta.get("not_interests") or []),
        "n_expertise_hits": len(sig.get("expertise_hits") or []),
        "n_not_interest_hits": len(sig.get("not_interest_hits") or []),
        "switched_off": sig.get("switched_off") or [],
        "switched_on": sig.get("switched_on") or [],
        "added": sig.get("added") or [],
        "overruled": sig.get("overruled") or [],
    }


def run_profile_import(
    conn: sqlite3.Connection,
    settings,
    *,
    user_id: int,
    profile_id: int,
    slug: str,
    profile: RpProfile,
    name: str,
    client,
    reporter=None,
    author: orcid_import.AuthorInfo | None = None,
    selection: str | list[str] = "default",
) -> OrcidImportResult:
    """One-shot fetch + seed for a profile with an ORCID (CLI path)."""
    fetch = run_profile_fetch(
        conn, settings, user_id=user_id, profile_id=profile_id, slug=slug,
        profile=profile, name=name, client=client, reporter=reporter, author=author,
    )
    if isinstance(selection, list):
        ids = selection
    else:
        records = orcid_import.load_work_records(conn, profile_id)
        if selection == "all":
            ids = [r.paper.openalex_id for r in records if r.seed_eligible]
        elif selection == "lead":
            defaults, _dup, _w = orcid_import.seed_defaults(records, 0)
            ids = [r.paper.openalex_id for r in defaults]
        else:
            from ...db.repos import orcid_works as works_repo
            ids = works_repo.default_ids(conn, profile_id)
    seed = orcid_import.run_seed(
        conn, settings, user_id=user_id, profile_id=profile_id, slug=slug,
        selected_ids=ids, reporter=reporter, after_topics=after_topics_hook(conn, profile_id),
    )
    seed.n_fetched = fetch.n_fetched
    seed.n_kept = fetch.n_kept
    seed.author = fetch.author
    seed.warnings = fetch.warnings + seed.warnings
    seed.report = {**fetch.report, **seed.report}
    seed.rp = rp_summary_from_topics(profiles_repo.topic_filters(conn, profile_id), load_meta(conn, profile_id))
    return seed


__all__ = [
    "PhraseHit",
    "RpProfile",
    "RpProfileError",
    "apply_rp_signals",
    "apply_rp_signals_from_meta",
    "deslug",
    "load_meta",
    "map_phrases_to_topics",
    "parse_profile",
    "parse_profile_dict",
    "persist_meta",
    "after_topics_hook",
    "resolve_author_for_profile",
    "rp_summary_from_topics",
    "run_profile_fetch",
    "run_profile_import",
]
