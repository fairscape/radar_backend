"""Profile/Paper ↔ SQLite bridge.

Two write paths through this module:

  store_profile_from_object — write a Profile and its seed papers (with
    embeddings + topics) to the DB; idempotent on (user_id, name).
  dedup_and_insert_candidates — write a list of (score, Paper) results
    from a gather run, dedup against profile_candidates so the user's
    prior triage state survives a re-fetch.

Both call into ``rag_lib.db.repos.*`` which is the only direct user of
sqlite3. Higher layers (CLI, API services) never construct SQL.

Papers without an ``openalex_id`` are skipped — the schema is
content-addressed by that key. In practice this only matters for CSV
seeds that never resolved on OpenAlex (paywalled, not indexed); the
runtime warning surfaces it.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import structlog

import sqlite3
import sys

from ..db import encode_vector
from ..db.repos import (
    candidates as candidates_repo,
    embeddings as embeddings_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    schedules as schedules_repo,
    users as users_repo,
    versions as versions_repo,
)
from ..paper import Paper, paper_keys, title_key
from ..profile import Profile


class _BatchConn:
    """Thin proxy around a sqlite3.Connection that suppresses commit().

    Repo functions call ``conn.commit()`` after every single row write.
    For bulk inserts (500+ papers) this causes 1000+ individual fsync
    operations on NFS, stalling the event loop for minutes.  Wrapping
    the connection in ``_BatchConn`` lets the caller do a single
    ``conn.commit()`` on the real connection after the batch finishes.
    """

    __slots__ = ("_conn",)

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def execute(self, *args, **kwargs):
        return self._conn.execute(*args, **kwargs)

    def executemany(self, *args, **kwargs):
        return self._conn.executemany(*args, **kwargs)

    def commit(self) -> None:
        pass  # suppressed — caller commits once at the end

    def rollback(self) -> None:
        self._conn.rollback()

    @property
    def row_factory(self):
        return self._conn.row_factory

    @row_factory.setter
    def row_factory(self, value):
        self._conn.row_factory = value


# How many row-writes to accumulate before a real commit. Small enough
# that the SQLite write lock is never held for more than a fraction of a
# second (concurrent API writes only get a 5s busy timeout), large enough
# that we aren't paying an fsync per row on NFS.
_COMMIT_EVERY = 100

log = structlog.get_logger("rag_lib.persistence.db_store")


def resolve_user_id(conn: sqlite3.Connection, email: str) -> int:
    """Get-or-create a user by email; return its id."""
    return int(users_repo.upsert(conn, email)["id"])


def _paper_dict_for_repo(paper: Paper) -> dict:
    d = paper.to_dict()
    return {
        "openalex_id": d.get("openalex_id"),
        "doi": d.get("doi"),
        "title": d.get("title") or "",
        "abstract": d.get("abstract"),
        "year": d.get("year"),
        "venue": d.get("venue"),
        "publication_date": d.get("publication_date"),
        # A list; papers.upsert encodes it, and writes nothing for an empty
        # one, so a gather never blanks a byline an import already stored.
        "authors": d.get("authors") or [],
        "primary_topic": d.get("primary_topic"),
        "topics": d.get("topics") or [],
        "source": d.get("source") or "unknown",
        "local_path": d.get("local_path"),
        "body_text": d.get("body_text"),
        "pdf_url": d.get("pdf_url"),
        "oa_status": d.get("oa_status"),
    }


def store_papers(
    conn: sqlite3.Connection,
    papers: list[Paper],
    *,
    skip_re_embed: bool = True,
) -> int:
    """Upsert each paper and any embeddings it carries.

    With ``skip_re_embed=True`` (default), an embedding is written only
    if no row exists for ``(openalex_id, model)``. This is the path the
    gather CLI uses to avoid recomputing SPECTER2 vectors for papers we
    have already seen.

    Returns the count of papers written (excluding skipped ones).
    """
    written = 0
    for p in papers:
        if not p.openalex_id:
            print(f"warn: skipping paper without openalex_id: {p.title!r}",
                  file=sys.stderr)
            continue
        papers_repo.upsert(conn, _paper_dict_for_repo(p))
        for model, vec in (p.embeddings or {}).items():
            if skip_re_embed and embeddings_repo.has(conn, p.openalex_id, model):
                continue
            embeddings_repo.upsert(conn, p.openalex_id, model, vec)
        written += 1
    return written


def _as_private_versions(conn: sqlite3.Connection, user_id: int, papers: list[Paper]) -> list[tuple[Paper, str, str | None]]:
    """Give each seed read from the user's own PDF (it has body text) its
    private version id, as an upload gets, and store the public record's
    metadata without that text. Returns ``(paper, file_hash, canonical)``
    for those, to register once the papers are stored. The vectors the
    paper carries were made from its text and go with it.
    """
    out = []
    for paper in papers:
        if (not paper.openalex_id or versions_repo.is_version(paper.openalex_id)
                or not (paper.body_text or "").strip()):
            continue
        path = Path(paper.local_path) if paper.local_path else None
        if path is not None and path.is_file():
            from ..vault import compute_file_hash
            digest = compute_file_hash(path)
        else:
            digest = hashlib.sha256(paper.body_text.encode("utf-8")).hexdigest()
        canonical = None if paper.openalex_id.startswith("local:") else paper.openalex_id
        if canonical is not None:
            papers_repo.upsert(conn, versions_repo.public_fields(_paper_dict_for_repo(paper)))
        paper.openalex_id = versions_repo.version_id(user_id, digest)
        paper.source = "user_pdf_version"
        out.append((paper, digest, canonical))
    return out


def store_profile_from_object(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    profile: Profile,
) -> int:
    """Persist a Profile and its seed corpus. Returns ``profile_id``.

    The fitted centroid (when present in ``profile.selector_config``) is
    stored as a float32 BLOB on ``profiles.centroid``; the full
    selector_config still goes into ``selector_config_json`` so nothing
    is lost. Coherence diagnostics, when surfaced by the selector,
    populate the dedicated columns for fast read-side queries.

    A seed with body text (``cli.ingest`` of a PDF) is the user's upload:
    it is stored as their private version, like a vault upload, and the
    fit and topic lists name it by that id.
    """
    # Before anything is written: a profile file naming another user's
    # upload would otherwise overwrite its text, and only the seed
    # trigger, after the commit, would refuse it. A version id is the
    # user's only once it is registered as theirs; an unregistered one
    # would be a version with no owner.
    foreign = [p.openalex_id for p in profile.papers
               if versions_repo.is_version(p.openalex_id) and versions_repo.owner(conn, p.openalex_id) != int(user_id)]
    if foreign:
        raise ValueError(f"{foreign[0]} is not one of your uploads; it cannot be imported")
    before = {id(p): p.openalex_id for p in profile.papers}
    private = _as_private_versions(conn, user_id, profile.papers)
    renames = {before[id(p)]: p.openalex_id for p, _, _ in private}
    if renames:
        profile.selector_config = versions_repo.rename_ids(profile.selector_config, renames)
        profile.topic_filters = versions_repo.rename_ids(profile.topic_filters, renames)
    sel_cfg = profile.selector_config or {}

    centroid_blob: bytes | None = None
    centroid_list = sel_cfg.get("centroid") if isinstance(sel_cfg, dict) else None
    if centroid_list:
        centroid_blob = encode_vector(centroid_list)

    diag = sel_cfg.get("diagnostics_snapshot") if isinstance(sel_cfg, dict) else None
    diag = diag or {}
    coh_med = diag.get("coherence_median")
    coh_iqr = diag.get("coherence_iqr")
    coh_bm = diag.get("coherence_bimodal")

    profile_id = profiles_repo.upsert(
        conn,
        user_id=user_id,
        name=profile.name,
        embedding_model=profile.embedding_model,
        n_seed=len(profile.papers),
        topic_filters=profile.topic_filters,
        centroid=centroid_blob,
        threshold=profile.threshold,
        selector_config=sel_cfg or None,
        gatherer_config=profile.gatherer_config or None,
        coherence_median=coh_med,
        coherence_iqr=coh_iqr,
        coherence_bimodal=coh_bm,
    )

    store_papers(conn, profile.papers)
    for paper, digest, canonical in private:
        conn.execute(
            "UPDATE papers SET file_hash = ?, uploaded_by_user_id = ? WHERE openalex_id = ?",
            (digest, user_id, paper.openalex_id),
        )
        versions_repo.register(conn, user_id=user_id, paper_id=paper.openalex_id,
                               canonical_id=canonical, file_hash=digest)
        papers_repo.record_upload(conn, user_id=user_id, openalex_id=paper.openalex_id, file_hash=digest,
                                  local_path=paper.local_path, n_pages=None)
    for paper in profile.papers:
        if not paper.openalex_id:
            continue
        profiles_repo.attach_seed(conn, profile_id, paper.openalex_id)

    # Ensure every profile has a default daily schedule. Idempotent: the
    # 0004 migration backfilled existing rows; this covers profiles
    # created post-migration (CLI gather, Phase 11 wizard) so the
    # scheduler picks them up on the next boot without manual setup.
    if schedules_repo.get(conn, profile_id) is None:
        schedules_repo.upsert(conn, profile_id=profile_id)

    return profile_id


def dedup_and_insert_candidates(
    conn: sqlite3.Connection,
    *,
    profile_id: int,
    gather_run_id: int | None,
    ranked: list[tuple],
    tier_used: str | None = None,
    source_topics: dict[str, str] | None = None,
    n_counted: int | None = None,
) -> tuple[int, int]:
    """Persist ranked gather results and dedup against prior candidates.

    ``n_counted``: only the first that many entries count toward the
    returned ``(n_new, n_redup)`` -- the gather stores the papers below the
    threshold after the passing ones, and those never reach the feed.

    ``ranked`` accepts either the legacy 2-tuple shape ``(score, Paper)``
    or the Phase 3 3-tuple ``(score, Paper, breakdown)`` where
    ``breakdown`` may carry ``score_raw``, ``score_max_seed``, and
    ``score_pct`` floats. Missing breakdown fields write ``NULL`` to
    their respective columns.

    Each entry triggers (in order):
      1. ``papers.upsert`` (idempotent; later sources don't blank fields).
      2. ``embeddings.upsert`` for every embedding the paper carries
         (skipped when an entry for that model already exists).
      3. ``candidates.insert_dedup`` — upsert that refreshes the score
         columns while leaving ``shown_at``/``saved_at``/``dismissed_at``
         untouched, so prior triage survives a resurface.

    Returns ``(n_new, n_redup)``.
    """
    # Title-level dedup: OpenAlex occasionally assigns different IDs to
    # the same paper (e.g. Zenodo versioned DOIs).  Keep only the
    # highest-scored entry per normalised title within this batch.
    #
    # Uses the same key as the gatherer's ``_Deduper``. This used to
    # normalise with ``strip().lower()`` alone, which differs in both
    # directions: it kept copies that differ only in punctuation, and it
    # collapsed unrelated papers sharing a short generic title.
    seen_titles: dict[str, int] = {}   # normalised title -> index in ranked
    deduped_indices: set[int] = set()
    for i, entry in enumerate(ranked):
        _, paper, _ = _unpack_entry(entry)
        key = title_key(paper.title)
        if not key:
            continue
        if key in seen_titles:
            deduped_indices.add(i)
        else:
            seen_titles[key] = i

    # Chunked batch-commit. The repo functions commit after every single
    # row, which is 1000+ fsyncs for 500 papers on NFS. Wrapping the whole
    # batch in ONE transaction fixes that but breaks the other way: it
    # holds the SQLite write lock for the entire gather, so concurrent API
    # writes (uploads, triage, threshold edits) blow past the 5s busy
    # timeout and fail with "database is locked". Committing every
    # ``_COMMIT_EVERY`` rows bounds both the fsync count and the lock hold.
    batch = _BatchConn(conn)

    # The same paper already stored under another id (an earlier scan
    # brought the other copy): by DOI or title, as the seeds are kept out
    # (profiles_repo.seed_keys). Deduped within the batch only, it came
    # back as a second card -- and, the first saved, as one still to review.
    known_ids: set[str] = set()
    owner_of: dict[str, str] = {}   # "doi:<doi>" / "title:<key>" -> stored id
    for row in candidates_repo.stored_paper_keys(conn, profile_id):
        known_ids.add(row["openalex_id"])
        for k in paper_keys(row["doi"], row["title"], row["year"], papers_repo.decode_authors(row)):
            owner_of.setdefault(k, row["openalex_id"])

    n_new = 0
    n_redup = 0
    try:
        papers = [_paper_from_entry(e) for e in ranked]
        for start in range(0, len(papers), _COMMIT_EVERY):
            store_papers(batch, papers[start:start + _COMMIT_EVERY])
            conn.commit()

        pending = 0
        for i, entry in enumerate(ranked):
            if i in deduped_indices:
                continue
            score, paper, breakdown = _unpack_entry(entry)
            if not paper.openalex_id:
                continue
            if paper.openalex_id not in known_ids:
                keys = paper_keys(paper.doi, paper.title, paper.year, paper.authors)
                if any(owner_of.get(k, paper.openalex_id) != paper.openalex_id for k in keys):
                    # Another id of a paper the interest has: a resurfacing.
                    if n_counted is None or i < n_counted:
                        n_redup += 1
                    continue
                for k in keys:
                    owner_of.setdefault(k, paper.openalex_id)
                known_ids.add(paper.openalex_id)
            if not math.isfinite(float(score)):
                # SQLite stores NaN as NULL and the column is NOT NULL: one
                # such score (a reranker or embedder glitch) failed the
                # whole chunk, and with it the scan.
                log.warning("db_store.non_finite_score_skipped", openalex_id=paper.openalex_id, profile_id=profile_id)
                continue
            added = candidates_repo.insert_dedup(
                batch,
                profile_id=profile_id,
                openalex_id=paper.openalex_id,
                score=float(score),
                tier_used=tier_used,
                gather_run_id=gather_run_id,
                score_raw=_breakdown_float(breakdown, "score_raw"),
                score_max_seed=_breakdown_float(breakdown, "score_max_seed"),
                score_pct=_breakdown_float(breakdown, "score_pct"),
                score_reranker_raw=_breakdown_float(breakdown, "score_reranker_raw"),
                score_reranker_norm=_breakdown_float(breakdown, "score_reranker_norm"),
                score_blended=_breakdown_float(breakdown, "score_blended"),
                sourced_by_topic_id=(source_topics or {}).get(paper.openalex_id),
                reranker_query_key=(breakdown or {}).get("reranker_query_key"),
            )
            if n_counted is not None and i >= n_counted:
                pass
            elif added:
                n_new += 1
            else:
                n_redup += 1
            pending += 1
            if pending >= _COMMIT_EVERY:
                conn.commit()
                pending = 0
        conn.commit()
    except Exception:
        # Don't leave a half-open transaction on a connection the caller
        # keeps using: jobs.py's error handler writes the audit row on
        # this same connection, and its commit would otherwise persist a
        # partial batch instead of discarding it.
        conn.rollback()
        raise

    return n_new, n_redup


def _unpack_entry(entry) -> tuple[float, Paper, dict]:
    """Accept either 2-tuple or 3-tuple ranked entries; return canonical 3-tuple."""
    if len(entry) == 3:
        score, paper, breakdown = entry
        return score, paper, dict(breakdown or {})
    score, paper = entry
    return score, paper, {}


def _paper_from_entry(entry) -> Paper:
    return entry[1]


def _breakdown_float(breakdown: dict, key: str) -> float | None:
    v = breakdown.get(key)
    return None if v is None else float(v)
