"""Wizard service — Phase 11.

Backs the four-step profile-build wizard's ``/api/profiles/draft/...``
endpoints. A "draft" is a normal ``profiles`` row carrying
``is_draft=1``; the wizard mutates the same row across steps and the
commit endpoint flips the flag, so seeds attached during Step 1 keep
their FK on commit without copy.

  Step 1 — seeds: handled by the existing vault upload endpoint
            (``POST /api/vault/upload`` with ``profile_slug=<draft>``).
            ``profile_seeds`` rows accumulate against the draft's id.
  Step 2 — coherence: ``compute_draft_coherence`` reads the seeds'
            embeddings and returns histogram + median + IQR + bimodality.
  Step 3 — topics: ``aggregate_draft_topics`` aggregates OpenAlex topic
            filters across the draft's seeds and writes the full set
            back to ``topic_filters_json`` so Step 4 sees them.
  Step 4 — calibrate: ``dry_run_draft`` runs ``radar.dry_run`` over the
            last ``days`` of OpenAlex output and returns a sweep + a
            10-card preview of the top hits.
  Commit  — ``commit_draft`` flips ``is_draft`` to 0, persists the
            chosen threshold + the pruned topic_filters, fits and
            stores whichever selector the draft picked (registry
            dispatch — when it's the centroid selector, the fitted
            centroid lands on ``profiles.centroid`` for fast restart),
            and registers a daily schedule.
"""

from __future__ import annotations

import json
import random
import sqlite3
import time
from typing import Any

import numpy as np
import structlog

from ... import calibration
from ...coherence import coherence as coherence_compute
from ...db import encode_vector
from ...db.repos import (
    draft_owned as draft_owned_repo,
    embeddings as embeddings_repo,
    gather_runs as gather_runs_repo,
    papers as papers_repo,
    profiles as profiles_repo,
    researchers as researchers_repo,
    users as users_repo,
    schedules as schedules_repo,
)
from ... import interest_terms, llm_topics, researcher_topics
from ...gatherers.openalex import OpenAlexGatherer
from ...openalex_tiers import enabled_topic_ids
from ...paper import Paper, Topic
from ...profile import Profile
from ...selectors import get_selector
from ..mappers import _read_minutes, bucket_for_similarity, floor4
from ..schemas import (
    MAX_NAME,
    Card,
    DraftCoherence,
    DraftDryRun,
    DryRunPaper,
    LeastSimilarPair,
    SeedSimilarity,
    SweepRow,
)

log = structlog.get_logger("rag_lib.api.services.wizard")


DEFAULT_THRESHOLDS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
PREVIEW_CARDS = 10
MAX_DRY_RUN_DAYS = 30
# The trial takes what a scan takes (each topic's most cited in the
# window, FUNCTIONAL_SPEC S.1), so the threshold is set on what the feed
# will see. No total cap beyond RADAR_GATHER_MAX_CANDIDATES.
DRY_RUN_FETCH_LIMIT: int | None = None


def capped_fetch_limit(settings: Any, requested: int | None) -> int | None:
    """``requested`` bounded by ``RADAR_GATHER_MAX_CANDIDATES`` when set
    (None = no cap)."""
    cap = getattr(settings, "RADAR_GATHER_MAX_CANDIDATES", None)
    if cap is None or int(cap) <= 0:
        return requested
    return int(cap) if requested is None else min(int(requested), int(cap))


# ---------------------------------------------------------------------------
# Step 0 — create draft
# ---------------------------------------------------------------------------


def create_draft(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    name: str,
    embedding_model: str = "placeholder-v1",
    selector: str = "centroid",
) -> dict:
    """Insert a draft profiles row. Returns ``{slug, name}``.

    The slug is ``slugify(name)`` with a numeric suffix on collision
    with one of the user's own slugs. ``profiles.name`` is unique per user
    (migration 0021) so we mirror the same suffix into the name when
    needed; the user sees the suffix and can rename later.

    ``embedding_model`` is stored on the profile so coherence joins and
    selector fits use the same vectors the upload path produced.

    ``selector`` is the registry key (``"centroid"``, ``"max_seed"``, or
    a plugin-registered name). Stored as ``selector_config_json={"type":
    <key>}`` on the draft so the dry-run and commit steps know which
    selector to fit. Both knobs are validated up-front; an unknown key
    raises ``ValueError`` before any DB write.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("name must not be empty")
    from ...embedders import get_embedder
    get_embedder(embedding_model)  # validate; raises ValueError if unknown
    get_selector(selector)         # validate; raises ValueError if unknown

    base_slug = profiles_repo.slugify(name)
    for _attempt in range(5):
        slug, suffix = _unique_slug(conn, user_id, base_slug, name)
        final_name = name if suffix == 0 else _suffixed(name, suffix)
        try:
            profile_id = profiles_repo.create_draft(
                conn,
                user_id=user_id,
                name=final_name,
                slug=slug,
                embedding_model=embedding_model,
            )
            break
        except sqlite3.IntegrityError as exc:
            # Another request took the name between the check and the
            # insert (a double-clicked Create): the next free one. Any
            # other constraint is a real error.
            conn.rollback()
            if "UNIQUE" not in str(exc):
                raise
    else:
        raise ValueError(f"could not find a free name for {name!r}")
    # Stash the chosen selector type so dry-run + commit pick it up.
    # No fitted state yet — that lands on commit.
    conn.execute(
        "UPDATE profiles SET selector_config_json = ? WHERE id = ?",
        (json.dumps({"type": selector}), profile_id),
    )
    conn.commit()
    return {"slug": slug, "name": final_name}


def _unique_slug(
    conn: sqlite3.Connection, user_id: int, base: str, name: str | None = None,
) -> tuple[str, int]:
    """Return ``(slug, suffix_n)`` where suffix_n is 0 for the bare slug.

    Walks ``base``, ``base-2``, ``base-3``, ... until neither the slug nor
    (when given) ``name (n)`` is one of *this user's* interests. Both are
    unique per user (migration 0021); checking the whole table, as this
    did, renamed an interest because another user had one of that name,
    and so told the user it existed.
    """
    def _taken(col: str, value: str) -> bool:
        return conn.execute(
            f"SELECT 1 FROM profiles WHERE user_id = ? AND {col} = ? LIMIT 1", (user_id, value),
        ).fetchone() is not None

    # The slug and the name are suffixed independently: two different names
    # that slugify alike (slugs are cut at 80 characters) used to get the
    # second renamed "<name> (2)" although that name was free.
    n = 1
    while _taken("slug", base if n == 1 else f"{base}-{n}"):
        n += 1
    slug = base if n == 1 else f"{base}-{n}"
    if name is None:
        return slug, 0
    m = 1
    while _taken("name", name if m == 1 else _suffixed(name, m)):
        m += 1
    return slug, 0 if m == 1 else m


def _suffixed(name: str, n: int) -> str:
    """``name (n)`` within MAX_NAME: the suffix shortens the name, it does
    not push a 120-character one to 124."""
    tail = f" ({n})"
    return f"{name[:MAX_NAME - len(tail)].rstrip()}{tail}"


# ---------------------------------------------------------------------------
# Step 2 — coherence
# ---------------------------------------------------------------------------


class SavingInProgress(RuntimeError):
    """The draft is being saved: its first scan is running, and it goes
    live when that succeeds. It cannot be discarded meanwhile."""

    def __init__(self, slug: str):
        super().__init__(f"draft '{slug}' is being saved; its first scan is running")
        self.slug = slug


class ImportInProgress(RuntimeError):
    """The draft's seeds are still being imported.

    Distinct from "this draft has no seeds": the wizard should show the
    import's progress and poll, not offer the manual pick step.
    """

    def __init__(self, slug: str):
        super().__init__(f"an import is still filling draft '{slug}'")
        self.slug = slug


def compute_draft_coherence(
    conn: sqlite3.Connection, *, user_id: int, slug: str
) -> DraftCoherence:
    row = _require_profile(conn, user_id, slug)
    profile_id = int(row["id"])
    embedding_model = row["embedding_model"]

    # An import attaches its seeds one at a time, so mid-import the draft
    # holds part of them: coherence (and from it topics, the trial and the
    # fit at commit) would describe a set the user never chose. With none
    # attached yet it also looked like a draft with no seeds at all, which
    # invited adding 82 papers by hand while they were being embedded.
    if gather_runs_repo.has_unfinished_import(conn, profile_id):
        raise ImportInProgress(slug)
    ids, vecs = _seed_embeddings_with_ids(conn, profile_id, embedding_model)
    bins = profiles_repo.coherence_bins(conn, profile_id)
    if not vecs:
        conn.execute(
            """
            UPDATE profiles SET
              coherence_median  = NULL,
              coherence_iqr     = NULL,
              coherence_bimodal = NULL,
              seed_sim_min      = NULL,
              seed_sim_median   = NULL,
              seed_sim_max      = NULL,
              n_seed            = (SELECT COUNT(*) FROM profile_seeds s WHERE s.profile_id = profiles.id),
              updated_at        = datetime('now')
            WHERE id = ?
            """,
            (profile_id,),
        )
        conn.commit()
        return DraftCoherence(
            bins=bins, median=0.0, iqr=0.0, bimodal=False, n=0,
            label="none", summary=calibration.describe_coherence("none", 0, None),
        )
    metrics = coherence_compute(vecs)
    median = _finite(metrics["median"])
    iqr = _finite(metrics["iqr"])
    bimodal = bool(metrics["bimodal"])
    n = int(metrics["n"])
    band = calibration.seed_similarity_band(vecs)
    label = calibration.coherence_label(median if n >= 2 else None, iqr, n)
    agreement = calibration.agreement_score(median) if n >= 2 else None
    least = None
    pair = calibration.least_similar_pair(vecs) if n >= 2 else None
    if pair is not None:
        i, j, cos = pair
        from ...db.repos import versions as versions_repo

        titles = _titles_for(conn, [ids[i], ids[j]])
        least = LeastSimilarPair(
            a_id=ids[i], a_title=titles.get(ids[i], ""),
            b_id=ids[j], b_title=titles.get(ids[j], ""),
            cosine=round(cos, 4),
            a_canonical_id=versions_repo.canonical_id(conn, ids[i]),
            b_canonical_id=versions_repo.canonical_id(conn, ids[j]),
        )
    # Persist now so the profile row reflects what we just showed
    # instead of waiting for commit_draft. Lets the profile detail page
    # render the histogram + median without re-running the compute.
    conn.execute(
        """
        UPDATE profiles SET
          coherence_median  = ?,
          coherence_iqr     = ?,
          coherence_bimodal = ?,
          seed_sim_min      = ?,
          seed_sim_median   = ?,
          seed_sim_max      = ?,
          n_seed            = (SELECT COUNT(*) FROM profile_seeds WHERE profile_id = ?),
          updated_at        = datetime('now')
        WHERE id = ?
        """,
        (
            median, iqr, 1 if bimodal else 0,
            band["min"] if band else None,
            band["median"] if band else None,
            band["max"] if band else None,
            profile_id, profile_id,
        ),
    )
    conn.commit()
    return DraftCoherence(
        bins=bins, median=median, iqr=iqr, bimodal=bimodal, n=n,
        label=label, agreement=agreement,
        summary=calibration.describe_coherence(label, n, agreement),
        seed_similarity=_seed_similarity_schema(band),
        least_similar=least,
    )


def _seed_similarity_schema(band: dict | None) -> SeedSimilarity | None:
    if not band:
        return None
    return SeedSimilarity(
        min=round(band["min"], 4), median=round(band["median"], 4), max=round(band["max"], 4),
    )


def _titles_for(conn: sqlite3.Connection, openalex_ids: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for oa in openalex_ids:
        prow = papers_repo.get_by_openalex_id(conn, oa)
        if prow is not None:
            out[oa] = prow["title"] or ""
    return out


# ---------------------------------------------------------------------------
# Step 3 — aggregated topics
# ---------------------------------------------------------------------------


def aggregate_draft_topics(
    conn: sqlite3.Connection, *, user_id: int, slug: str
) -> dict:
    """Aggregate OpenAlex topics across the profile's seeds.

    Works on drafts AND committed profiles, so the profile detail page
    can re-run topic aggregation after the user uploads more seeds.
    Persists the full aggregated dict back to ``topic_filters_json``.

    When UMLS is enabled, also merges UMLS-mapped topics (stored on
    ``papers.umls_mapped_topics_json`` at upload time) into the filters
    to improve coarse-filter recall.

    Any on/off choices the user already made survive: re-aggregation
    rebuilds the list from scratch, so without this the "RECOMPUTE"
    button would silently switch every pruned topic back on.
    """
    row = _require_profile(conn, user_id, slug)
    profile_id = int(row["id"])
    # Part of the seeds would give part of the topics (step 3 by Back/Forward).
    if gather_runs_repo.has_unfinished_import(conn, profile_id):
        raise ImportInProgress(slug)
    # Topics we've never seen start on -- but in a copy being edited they
    # join a saved interest's list, and start off as on its edit page.
    new_on = not is_edit_copy(row)
    return _write_rebuilt_topics(
        conn, profile_id, lambda: _aggregate_topics(conn, profile_id, new_on=new_on),
    )


def _is_draft(conn: sqlite3.Connection, profile_id: int) -> bool | None:
    got = conn.execute("SELECT is_draft FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    return bool(got[0]) if got is not None else None


def is_edit_copy(row) -> bool:
    """A copy of a saved interest made by Edit interest (FUNCTIONAL_SPEC E):
    a draft row pointing at the interest it copies."""
    return row is not None and bool(row["is_draft"]) and row["edit_of"] is not None


def is_new_draft(row) -> bool:
    """A draft of a new interest (the wizard's): its topic list follows its
    seeds, a new topic starts on. A copy being edited is a draft too -- not
    listed, scanned or scored -- but its topics are a saved interest's."""
    return row is not None and bool(row["is_draft"]) and row["edit_of"] is None


def _follows_seeds(conn: sqlite3.Connection, profile_id: int) -> bool | None:
    """``is_new_draft`` of the row as stored now; None when it is gone."""
    row = conn.execute("SELECT is_draft, edit_of FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    return is_new_draft(row) if row is not None else None


def _raw_topic_filters(conn: sqlite3.Connection, profile_id: int) -> str | None:
    got = conn.execute("SELECT topic_filters_json FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    return got[0] if got is not None else None


def _write_rebuilt_topics(conn: sqlite3.Connection, profile_id: int, build) -> dict:
    """Store the list ``build()`` makes from the seeds and the stored list,
    unless the stored list changed while it was being built: then build it
    again from the new one, and store that.

    The build can take seconds (UMLS extraction, label mapping), and the
    step-3 list is re-read on focus and by other tabs, so a read that
    started before "Use profile topics" used to write its older list over
    the apply -- flag, added topics and marks gone, with the toast saying
    they were applied. Comparing only the flag still let it write over a
    not-wanted choice or a switch made meanwhile, so the whole stored list
    is compared, under the write lock; the build itself runs without it, so
    the lock is held for one UPDATE.
    """
    seen = _raw_topic_filters(conn, profile_id)
    was_draft = _is_draft(conn, profile_id)
    topic_filters = build()
    for _ in range(3):
        if conn.in_transaction:
            conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        try:
            # The stored list, not the built one: a build that found the
            # researcher gone drops the flag on purpose and must not loop.
            now = _raw_topic_filters(conn, profile_id)
            if _is_draft(conn, profile_id) != was_draft:
                # Committed (or gone) while it was built: a draft's rebuild
                # (new topics on) must not land on a saved list. The caller
                # tries again and takes the saved interest's path.
                conn.rollback()
                raise TopicsChanging(profile_id)
            if now == seen:
                conn.execute(
                    "UPDATE profiles SET topic_filters_json = ?, updated_at = datetime('now') WHERE id = ?",
                    (json.dumps(_without_unmatched_marks(topic_filters)), profile_id),
                )
                conn.commit()
                return topic_filters
            conn.rollback()
        except BaseException:
            conn.rollback()
            raise
        log.info("wizard.topics_rebuilt_after_profile_change", profile_id=profile_id)
        seen = now
        topic_filters = build()
    # Three rebuilds, each overtaken by another write: writing the last one
    # anyway could still undo one. The caller tries again.
    raise TopicsChanging(profile_id)


class TopicsChanging(RuntimeError):
    """The stored topic list changed while it was being rebuilt, every time
    it was tried."""

    def __init__(self, profile_id: int) -> None:
        super().__init__(f"profile {profile_id}: topics changed while the list was being rebuilt")
        self.profile_id = profile_id


def _aggregate_topics(
    conn: sqlite3.Connection, profile_id: int, *, new_on: bool, profile_topics: bool | None = None,
) -> dict:
    """The seeds' aggregated topics, carrying forward prior on/off states.

    Writes nothing to the profile. ``new_on`` is the state of a topic not
    seen before. (A seed OpenAlex does not have may get its content topics
    stored on its paper row here; see ``_content_topics``.)

    ``profile_topics`` says whether the researcher's profile topics are folded
    in: None reads the stored flag, True/False overrides it (the apply and
    remove paths, which then write what this returns in one go).
    """
    prior_on = _prior_topic_states(conn, profile_id)
    stored = _stored_topic_filters(conn, profile_id)
    from_content = _content_topics(conn, profile_id)
    papers = _load_seed_papers(conn, profile_id)
    topic_filters = Profile.aggregate_topic_filters(papers)
    _keep_content_topics(topic_filters, from_content)
    _keep_adopted_topics(stored, topic_filters)
    llm_topics.keep_stored(stored, topic_filters)

    # The researcher's own profile topics, once the user has applied them
    # (``set_researcher_topics``): the list is rebuilt from the seeds every
    # time, so they are folded in again here, from the profile as stored now.
    applied = researcher_topics.is_applied(stored) if profile_topics is None else bool(profile_topics)
    from_profile = _researcher_topics_for(conn, profile_id) if applied else None

    # Best-effort UMLS topic merge: read the per-paper mapped topics
    # that vault.upload() stored and merge novel ones into the filters.
    # A topic the profile wants is the interest's already: not a UMLS one.
    topic_filters = _try_merge_umls_topics(
        conn, profile_id, topic_filters, also_have=(from_profile or {}).get("wanted") or [])
    # Before the profile is applied: it marks these as any other topic, and
    # a mark it has on one is not taken for a mark that vanished.
    topic_filters = _keep_unmatched_seed_topics(conn, profile_id, stored, topic_filters)
    if from_profile is not None:
        topic_filters[researcher_topics.CHOICES] = researcher_topics.stored_choices(stored, from_profile)
        # A topic only a removed seed brought still counts as the seeds'
        # (E.8: removing a seed moves no switch, so neither the not-wanted
        # default), as in ``_overlap_basis``: the same default either way.
        topic_filters = researcher_topics.apply(topic_filters, from_profile, basis=topic_filters)
        if from_profile.get("labels_unmapped"):
            # Part of the profile (its text labels) could not be mapped just
            # now: the topics they gave, their marks and states, are kept
            # as the last full read stored them rather than dropped for the
            # duration -- a scan in between would search without them and
            # the user's switches on them would be lost.
            topic_filters = _keep_stored_profile(stored, topic_filters)
        else:
            _settle_vanished_marks(stored, topic_filters, prior_on)
        _carry_before_states(stored, topic_filters)
    elif applied and _has_researcher(conn, profile_id):
        # The profile lists no topics Radar can read just now (re-imported
        # without them, or only text labels and the mapper is down): still
        # applied. What the last readable aggregation stored -- the
        # profile-only topics, the marks, the remembered states -- is kept
        # as it was; "Remove profile topics" is the way out.
        topic_filters = _keep_stored_profile(stored, topic_filters)
    elif applied:
        # The researcher was forgotten: the profile goes with them, and the
        # marked topics go back as they were before it was applied.
        topic_filters[researcher_topics.FLAG] = False
        _restore_or_adopt(stored, topic_filters, prior_on)
    elif isinstance(stored.get(researcher_topics.CHOICES), dict) and _has_researcher(conn, profile_id):
        # Not applied (taken off, say): the not-wanted choices wait for the
        # next apply, as ``strip`` leaves them -- while the researcher is
        # there; forgetting them forgets the choices too.
        topic_filters[researcher_topics.CHOICES] = dict(stored[researcher_topics.CHOICES])

    stored_marks = {researcher_topics.code(t["id"]): t.get("profile")
                    for t in stored.get("topics") or [] if t.get("id")}
    for t in topic_filters.get("topics") or []:
        if t.get("id"):
            t["on"] = prior_on.get(t["id"], _new_topic_on(t, new_on))
            # A topic the profile has an opinion on starts as the profile
            # says when the user has not seen it, or has not seen the
            # profile's opinion on it: a mark new since the stored list (a
            # re-import that now rejects a seed topic) is the profile's say,
            # as on the first apply, not "a choice the user made since". A
            # state set under the same mark is the user's and is kept.
            # Either way the state it would have had is remembered, so
            # removing the profile can put it back (``set_researcher_topics``).
            if from_profile is not None and t.get("profile"):
                t.setdefault(researcher_topics.BEFORE, prior_on.get(t["id"], _new_topic_on(t, new_on)))
                if t["id"] not in prior_on or stored_marks.get(researcher_topics.code(t["id"])) != t["profile"]:
                    t["on"] = t["profile"] == researcher_topics.INTEREST
    return topic_filters


# A saved topic no seed brings any more (FUNCTIONAL_SPEC T.2-2).
UNMATCHED = "unmatched"
# The topics a seed brings: OpenAlex's own (no source), read from an
# unmatched PDF's text, mapped from its UMLS concepts. A researcher's
# profile or Claude-generated topics are not the seeds'.
_SEED_SOURCES = (None, "pdf_content", "umls")


def _keep_unmatched_seed_topics(conn: sqlite3.Connection, profile_id: int, stored: dict,
                                topic_filters: dict) -> dict:
    """``topic_filters`` plus the saved interest's seed topics the rebuild
    left out (FUNCTIONAL_SPEC T.2-2, T.2-3).

    The list is rebuilt from the seeds, so removing a seed used to take its
    topics out of the edit page's list, and the next save took them out of
    the scans with it: removing a paper and changing what the interest
    searches are separate choices. A topic no current seed brings stays as
    saved, on or off, marked ``unmatched``, with no seed counted; the user
    switches it off. One a seed still brings that this rebuild left out
    (OpenAlex's beyond the list's length, PDF-text topics switched off)
    stays too, unmarked -- except a UMLS one, which the merge's own rules
    (its cap, a surer concept) settle. A new interest's draft still follows
    its seeds: nothing searches by it yet. A copy being edited is the saved
    interest's list (FUNCTIONAL_SPEC E).
    """
    topics = list(topic_filters.get("topics") or [])
    for t in topics:
        t.pop(UNMATCHED, None)
    if _follows_seeds(conn, profile_id) is not False:
        return {**topic_filters, "topics": topics} if topics else topic_filters
    code = researcher_topics.code
    have_ids = {code(t.get("id")) for t in topics if t.get("id")}
    have_names = {(t.get("display_name") or "").lower() for t in topics}
    brought = _seed_topic_codes(conn, profile_id)
    # Already in the rebuild but no seed's: a stored copy put back as it was
    # (``_keep_stored_umls`` after a merge that failed).
    for i, t in enumerate(topics):
        if t.get("id") and t.get("source") in _SEED_SOURCES and code(t["id"]) not in brought:
            topics[i] = {**t, "count": 0, "papers": 0, UNMATCHED: True}
    kept = []
    for old in stored.get("topics") or []:
        name = (old.get("display_name") or "").lower()
        if (not old.get("id") or old.get("source") not in _SEED_SOURCES
                or code(old["id"]) in have_ids or (name and name in have_names)):
            continue
        still = code(old["id"]) in brought
        if still and old.get("source") == "umls":
            continue
        # The researcher's marks are put on by ``researcher_topics.apply``,
        # which runs on this list next (as on the rest of it).
        t = {k: v for k, v in old.items() if k not in ("profile", researcher_topics.BEFORE)}
        if not still:
            t.update(count=0, papers=0)
            t[UNMATCHED] = True
        kept.append(t)
        have_ids.add(code(t["id"]))
        have_names.add(name)
    return {**topic_filters, "topics": [*topics, *kept]}


def _seed_topic_codes(conn: sqlite3.Connection, profile_id: int) -> set[str]:
    """Every topic a current seed brings, as ``researcher_topics.code``: its
    OpenAlex or PDF-text topics (``topics_json``) and the ones its UMLS
    concepts map to (``umls_mapped_topics_json``) -- read from the seeds,
    not from what a rebuild kept."""
    out: set[str] = set()
    for oa_id in profiles_repo.list_seed_openalex_ids(conn, profile_id):
        row = conn.execute("SELECT topics_json, umls_mapped_topics_json FROM papers WHERE openalex_id = ?",
                           (oa_id,)).fetchone()
        if row is None:
            continue
        for raw, read in ((row[0], lambda d: (d.get("topics") or []) + [d.get("primary_topic")]),
                          (row[1], lambda d: d)):
            try:
                got = read(json.loads(raw)) if raw else []
            except (ValueError, TypeError, AttributeError):
                continue
            for t in got if isinstance(got, list) else []:
                tid = (t.get("id") or t.get("topic_id")) if isinstance(t, dict) else None
                if tid:
                    out.add(researcher_topics.code(tid))
    return out


def _without_unmatched_marks(topic_filters: dict) -> dict:
    """What is stored: ``unmatched`` is worked out on each aggregation from
    the seeds of the time, so a stored one would go stale (a seed added back)."""
    topics = topic_filters.get("topics")
    if not topics or not any(UNMATCHED in t for t in topics):
        return topic_filters
    return {**topic_filters, "topics": [{k: v for k, v in t.items() if k != UNMATCHED} for t in topics]}


def _new_topic_on(t: dict, new_on: bool) -> bool:
    """The state of a topic the list did not have before. A UMLS topic
    starts off whatever the others do: matched on the words alone, it is
    the user's to switch on (FUNCTIONAL_SPEC U.2-8)."""
    return new_on and t.get("source") != "umls"


def _keep_stored_profile(stored: dict, topic_filters: dict) -> dict:
    """``topic_filters`` plus what the stored list says of the profile: its
    own topics (source ``researcher``), and each topic's mark and remembered
    state; the flag stays set. For an aggregation that cannot read the
    profile right now."""
    out = dict(topic_filters)
    topics = [dict(t) for t in out.get("topics") or []]
    have = {t.get("id") for t in topics}
    by_id = {t.get("id"): t for t in stored.get("topics") or [] if t.get("id")}
    for t in topics:
        old = by_id.get(t.get("id"))
        if old is None:
            continue
        for key in ("profile", researcher_topics.BEFORE):
            if key in old:
                t[key] = old[key]
    for tid, old in by_id.items():
        if old.get("source") == researcher_topics.SOURCE and tid not in have:
            topics.append(dict(old))
    out["topics"] = topics
    # The stored choices, with any this aggregation was given on top.
    out[researcher_topics.CHOICES] = {**researcher_topics.stored_choices(stored),
                                      **(topic_filters.get(researcher_topics.CHOICES) or {})}
    out[researcher_topics.FLAG] = True
    return out


def _stored_topic_filters(conn: sqlite3.Connection, profile_id: int) -> dict:
    try:
        return profiles_repo.topic_filters(conn, profile_id) or {}
    except (ValueError, TypeError):  # unreadable JSON; a database error is not "nothing stored"
        return {}


def _has_researcher(conn: sqlite3.Connection, profile_id: int) -> bool:
    """The interest has a researcher's profile to read: the copy it took
    when it applied the topics, or the researcher it was built from."""
    if _researcher_snapshot(conn, profile_id) is not None:
        return True
    row = conn.execute(
        "SELECT 1 FROM profiles p JOIN researchers r ON r.id = p.researcher_id WHERE p.id = ?",
        (profile_id,),
    ).fetchone()
    return row is not None


def _settle_vanished_marks(stored: dict, topic_filters: dict, prior_on: dict[str, bool]) -> None:
    """A topic the profile had an opinion on last time and has none on now
    (its document was re-read without it): the state the profile dictated
    goes back to what it was before, unless the user changed it since. Its
    remembered state is not carried on (``_carry_before_states`` keeps only
    marked topics'), so nothing is left behind to restore later."""
    marked_now = {researcher_topics.code(t.get("id")) for t in topic_filters.get("topics") or [] if t.get("profile")}
    for old in stored.get("topics") or []:
        if not old.get("id") or not old.get("profile") or researcher_topics.code(old["id"]) in marked_now:
            continue
        dictated = old["profile"] == researcher_topics.INTEREST
        if researcher_topics.BEFORE in old and bool(old.get("on")) == dictated:
            prior_on[old["id"]] = bool(old[researcher_topics.BEFORE])


def _restore_or_adopt(stored: dict, topic_filters: dict, prior_on: dict[str, bool]) -> None:
    """The researcher is gone. The marked topics go back as they were before
    the profile was applied -- unless that would leave nothing on while the
    list had something on: then every state stays and the profile's own
    topics are kept as ordinary ones (as ``drop_profile_topics`` does)."""
    stored_topics = [t for t in stored.get("topics") or [] if t.get("id")]
    restored = {t["id"]: bool(t[researcher_topics.BEFORE]) for t in stored_topics if researcher_topics.BEFORE in t}
    own = [t for t in stored_topics if t.get("source") != researcher_topics.SOURCE]
    if (any(restored.get(t["id"], bool(t.get("on"))) for t in own)
            or not any(t.get("on") for t in stored_topics)):
        prior_on.update(restored)
        return
    have = {researcher_topics.code(t.get("id")) for t in topic_filters.get("topics") or [] if t.get("id")}
    for t in stored_topics:
        if t.get("source") == researcher_topics.SOURCE and researcher_topics.code(t["id"]) not in have:
            kept = {k: v for k, v in t.items() if k not in ("profile", researcher_topics.BEFORE)}
            kept["source"] = researcher_topics.ADOPTED
            topic_filters.setdefault("topics", []).append(kept)


def _restore_before(t: dict) -> None:
    """Take the profile's say off one marked topic: back to its remembered
    pre-apply state -- only while it is still as the profile dictated. A
    switch the user made since (a wanted topic turned off on the edit page)
    is theirs and stays; restoring it regardless turned that topic back on
    for every later scan. As ``_settle_vanished_marks``."""
    if researcher_topics.BEFORE not in t:
        return
    mark = t.get("profile")
    if mark and bool(t.get("on", True)) != (mark == researcher_topics.INTEREST):
        return
    t["on"] = bool(t[researcher_topics.BEFORE])


def _carry_before_states(stored: dict, topic_filters: dict) -> None:
    """Keep each marked topic's recorded pre-apply state across a rebuild."""
    before = {t.get("id"): t[researcher_topics.BEFORE] for t in stored.get("topics") or []
              if t.get("id") and researcher_topics.BEFORE in t}
    for t in topic_filters.get("topics") or []:
        if t.get("profile") and t.get("id") in before:
            t[researcher_topics.BEFORE] = before[t["id"]]


def _researcher_snapshot(conn: sqlite3.Connection, profile_id: int) -> dict | None:
    """The researcher's profile as the interest took it when it applied the
    topics (``{"researcher_id", "name", "document"}``), or None.

    A new interest's draft has it only while the researcher is there: the
    draft is still being built from them, and deleting them takes their
    topics off it (E.7). Every reader of the copy asks here, so none of them
    keeps a deleted researcher's topics on a draft."""
    got = conn.execute(
        "SELECT p.researcher_doc_json, p.is_draft, p.edit_of, r.id AS rid FROM profiles p"
        " LEFT JOIN researchers r ON r.id = p.researcher_id WHERE p.id = ?",
        (profile_id,),
    ).fetchone()
    if got is None or not got["researcher_doc_json"]:
        return None
    if got["is_draft"] and got["edit_of"] is None and got["rid"] is None:
        return None
    got = (got["researcher_doc_json"],)
    try:
        snap = json.loads(got[0])
    except (ValueError, TypeError):
        return None
    return snap if isinstance(snap, dict) else None


def _take_researcher_snapshot(conn: sqlite3.Connection, profile_id: int, rrow: sqlite3.Row) -> None:
    """Copy the researcher's profile onto the interest, as applying the
    topics takes it: from then on the interest reads this copy, so a
    re-import or a delete of the researcher does not change its topics
    (FUNCTIONAL_SPEC E.7). Committed."""
    try:
        document = json.loads(rrow["document_json"]) if rrow["document_json"] else None
    except (ValueError, TypeError):
        document = None
    conn.execute(
        "UPDATE profiles SET researcher_doc_json = ? WHERE id = ?",
        (json.dumps({"researcher_id": int(rrow["id"]), "name": rrow["name"], "document": document}), profile_id),
    )
    conn.commit()


def _researcher_topics_for(conn: sqlite3.Connection, profile_id: int, *, live: bool = False) -> dict | None:
    """The profile topics of the researcher this interest was built from, or
    None when there is none or their profile listed none.

    Read from the copy the interest took when it applied them; ``live`` (or
    no copy yet) reads the researcher as stored now -- what applying them
    would take."""
    snap = None if live else _researcher_snapshot(conn, profile_id)
    if snap is not None:
        document = snap.get("document")
        base = researcher_topics.read(document if isinstance(document, dict) else None)
    else:
        row = conn.execute(
            "SELECT r.* FROM researchers r JOIN profiles p ON p.researcher_id = r.id WHERE p.id = ?",
            (profile_id,),
        ).fetchone()
        if row is None:
            return None
        base = researcher_topics.from_row(row)
    from ..settings import get_settings
    view = researcher_topics.with_labels(base, settings=get_settings())
    return view if researcher_topics.available(view) else None


class SavedInterest(RuntimeError):
    """A saved interest's seeds change only through its Edit interest copy
    (FUNCTIONAL_SPEC E.6 R12). Refused in the services, not only by the
    routes: a route without the check (``POST /draft/{slug}/seeds``) added
    seeds to a saved interest, refitted and rescored it."""

    def __init__(self, slug: str):
        super().__init__(f"'{slug}' is a saved interest: change its seeds in Edit interest")
        self.slug = slug


class NoTopicOn(ValueError):
    """Applying the profile's topics would leave none switched on."""


def drop_profile_topics(conn: sqlite3.Connection, profile_id: int) -> bool:
    """Take an applied profile off the stored list without rebuilding it:
    marked topics back as they were, profile-only topics gone, flag off.
    Returns whether anything was applied. For forgetting the researcher:
    the not-wanted choices go too, applied or not (``strip`` keeps them for
    a re-apply, which there will not be). Only a new interest's draft: a
    saved interest, or one saved meanwhile, keeps the topics it was saved
    with (E.7).

    Read and written under one write lock, so a switch made in another tab
    meanwhile is not undone."""
    if conn.in_transaction:
        conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        # Read under the lock: a commit since the caller listed the drafts
        # makes it a saved interest, which keeps the topics it was saved
        # with -- its copy of the profile stays (E.7).
        row = conn.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
        live = row is None or not is_new_draft(row)
        applied = False if live else _drop_profile_topics_locked(conn, profile_id)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    if applied:
        log.info("wizard.researcher_topics_dropped", profile_id=profile_id)
    return applied


def _drop_profile_topics_locked(conn: sqlite3.Connection, profile_id: int) -> bool:
    stored = _stored_topic_filters(conn, profile_id)
    if not researcher_topics.is_applied(stored):
        if researcher_topics.CHOICES in stored:
            out = dict(stored)
            out.pop(researcher_topics.CHOICES)
            conn.execute(
                "UPDATE profiles SET topic_filters_json = ?, updated_at = datetime('now') WHERE id = ?",
                (json.dumps(out), profile_id),
            )
        return False
    topics = [dict(t) for t in stored.get("topics") or []]
    for t in topics:
        _restore_before(t)
    out = researcher_topics.strip({**stored, "topics": topics})
    out.pop(researcher_topics.CHOICES, None)
    conn.execute(
        "UPDATE profiles SET topic_filters_json = ?, updated_at = datetime('now') WHERE id = ?",
        (json.dumps(out), profile_id),
    )
    return True


def researcher_topics_view(conn: sqlite3.Connection, *, user_id: int, slug: str) -> dict:
    """What ``GET /researcher-topics`` shows: the researcher, their profile
    topics, and whether the interest has them applied. Draft or saved.

    ``applied`` is the stored flag while the researcher is still there: it
    stays true through a moment the profile cannot be read (the label
    mapper down), when ``topics.available`` is false."""
    row = _require_profile(conn, user_id, slug)
    profile_id = int(row["id"])
    stored = _stored_topic_filters(conn, profile_id)
    rrow = None
    if row["researcher_id"] is not None:
        rrow = researchers_repo.get_for_user(conn, user_id, int(row["researcher_id"]))
    applied = researcher_topics.is_applied(stored)
    # Applied: the copy the interest took, which stays when the researcher
    # is re-imported or deleted. Not applied: the researcher as stored now,
    # which is what applying would take.
    snap = _researcher_snapshot(conn, profile_id) if applied else None
    if snap is not None:
        view = _researcher_topics_for(conn, profile_id)
        name = rrow["name"] if rrow is not None else snap.get("name")
    else:
        view = _researcher_topics_for(conn, profile_id, live=True) if rrow is not None else None
        name = rrow["name"] if rrow is not None else None
    present = rrow is not None or snap is not None
    return {
        "researcher_id": int(rrow["id"]) if rrow is not None else None,
        "researcher_name": name,
        "applied": bool(present and applied),
        "topics": {"available": view is not None,
                   **researcher_topics.public(view or researcher_topics.read(None)),
                   "not_interest": _not_interest_rows(conn, row, view, stored) if view is not None else []},
    }


def _not_interest_rows(conn: sqlite3.Connection, row: sqlite3.Row, view: dict, stored: dict) -> list[dict]:
    """The not-wanted topics with their choices. A default is counted against
    the list rebuilt from the seeds, as every path that records one counts
    it (the step-3 rebuild, a topics save, an apply on a saved interest)."""
    basis = _overlap_basis(conn, int(row["id"]))
    return researcher_topics.not_interest_rows(view, stored, basis=basis)


def _overlap_basis(conn: sqlite3.Connection, profile_id: int) -> dict:
    """The list rebuilt from the seeds without the profile: what a
    not-wanted topic's default is counted against, as every rebuild
    counts it -- so the default shown is the one recorded. (A read-only
    copy of the rebuild would have to repeat its UMLS, PDF-text, Claude
    and adopted rules and drift from them.) Should the rebuild fail, the
    stored list's topics stand in rather than the view failing."""
    # The UMLS merge records how it went for step 3's notice; this read is
    # not the list the user sees, so it leaves that record as it was.
    had, kept = profile_id in _UMLS_MERGE_STATE, _UMLS_MERGE_STATE.get(profile_id)
    try:
        return _aggregate_topics(conn, profile_id, new_on=False, profile_topics=False)
    except Exception as exc:  # noqa: BLE001 -- a view, not a write
        log.warning("wizard.overlap_basis_from_stored", profile_id=profile_id, reason=type(exc).__name__)
        return {"topics": [t for t in _stored_topic_filters(conn, profile_id).get("topics") or []
                           if t.get("source") != researcher_topics.SOURCE]}
    finally:
        if had:
            _UMLS_MERGE_STATE[profile_id] = kept
        else:
            _UMLS_MERGE_STATE.pop(profile_id, None)


def set_researcher_topics(
    conn: sqlite3.Connection, *, user_id: int, slug: str, enabled: bool,
    choice: tuple[str, str] | None = None,
) -> dict:
    """Apply or drop the researcher's profile topics on an interest's list.

    Applying folds the wanted topics in switched on and switches the
    not-wanted ones off -- an explicit choice, so it overrides what the
    user had set on those topics before, which is remembered on each
    marked topic -- and records the flag so every later re-aggregation does
    the same. Dropping puts every marked topic back as it was before the
    profile was applied, removes the topics only the profile brought, and
    clears the flag. One write either way. The profile's topics are the
    reranker's queries in topic mode, so a saved interest's rerank is
    refreshed as after ``save_topics``.

    ``choice`` is ``(topic id, filter | keep | add)``: one not-wanted
    topic's choice set on an interest the profile is applied to, the rest
    of the list re-applied as it stands (see :func:`set_not_interest_choice`).

    Raises ``ValueError`` when the interest has no researcher with profile
    topics, :class:`NoTopicOn` (a ValueError) when applying would switch
    every topic off -- a scan searches by them, as ``save_topics`` --
    and ``ImportInProgress`` while an import is filling the draft.
    """
    row = _require_profile(conn, user_id, slug)
    profile_id = int(row["id"])
    if gather_runs_repo.has_unfinished_import(conn, profile_id):
        raise ImportInProgress(slug)
    if enabled:
        applied = researcher_topics.is_applied(_stored_topic_filters(conn, profile_id))
        if applied and _researcher_snapshot(conn, profile_id) is not None:
            # Applied already (a not-wanted choice, or the same apply again):
            # the copy it took stands, whatever the researcher is now.
            if _researcher_topics_for(conn, profile_id) is None:
                raise ValueError(
                    "this interest was not built from a researcher whose profile lists topics"
                )
        else:
            rrow = None
            if row["researcher_id"] is not None:
                rrow = researchers_repo.get_for_user(conn, user_id, int(row["researcher_id"]))
            if rrow is None or _researcher_topics_for(conn, profile_id, live=True) is None:
                raise ValueError(
                    "this interest was not built from a researcher whose profile lists topics"
                )
            # Applying takes the profile as it is now; the interest keeps that.
            _take_researcher_snapshot(conn, profile_id, rrow)
    for _attempt in range(3):
        # The row again each time: a commit in between makes a draft a saved
        # interest, whose list must not be rebuilt from the seeds.
        row = conn.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
        if row is None:
            raise LookupError(f"interest '{slug}' not found")  # deleted in another tab
        raw = row["topic_filters_json"]
        out = _set_researcher_topics_once(conn, row, profile_id, enabled=enabled, raw=raw, choice=choice)
        if out is not None:
            return out
    raise TopicsChanging(profile_id)


def _set_researcher_topics_once(
    conn: sqlite3.Connection, row: sqlite3.Row, profile_id: int, *, enabled: bool, raw: str | None,
    choice: tuple[str, str] | None = None,
) -> dict | None:
    """One go of ``set_researcher_topics`` against the stored list as read
    (``raw``). None when the list changed before the write (a topics save
    from another tab while the profile's labels were mapped): the write
    would have undone it, so the caller reads again and retries."""
    stored = _stored_topic_filters(conn, profile_id)
    if not enabled and not researcher_topics.is_applied(stored):
        # Nothing applied, nothing to take off: the stored list stays as it
        # is (a rebuild here would bring in topics the user has not seen).
        return stored
    if choice is not None:
        if not researcher_topics.is_applied(stored):
            raise ValueError("apply the profile's topics before choosing what to do with the ones it does not want")
        view = _researcher_topics_for(conn, profile_id)
        if view is None:
            raise ValueError("this interest was not built from a researcher whose profile lists topics")
        # Draft or saved, the one topic moves on the stored list and nothing
        # else does: no rebuild, so neither the label mapper being down
        # (which keeps the stored marks) nor a re-import that changed other
        # marks can undo or widen the choice. The next rebuild settles the rest.
        c = researcher_topics.code(choice[0])
        choices = {**researcher_topics.stored_choices(stored, view), c: choice[1]}
        new = researcher_topics.chosen_topic(stored, view, c, choice[1], basis=_overlap_basis(conn, profile_id))
        full = _with_one_topic_from(stored, new, c, choices)
    elif is_new_draft(row):
        full = _researcher_topics_rebuilt(conn, profile_id, enabled=enabled)
    else:
        full = _researcher_topics_on_stored(conn, profile_id, stored, enabled=enabled)
    if enabled:
        topics = full.get("topics") or []
        if topics and not any(t.get("on", True) for t in topics):
            raise NoTopicOn("the profile would switch every topic off: a scan searches by them")
    else:
        kept = full.get("topics") or []
        # On a saved interest the list is what the next scan searches by; a
        # draft's is settled on step 3, which says so when nothing is ticked.
        if not is_new_draft(row) and kept and not any(t.get("on", True) for t in kept):
            raise NoTopicOn("removing the profile's topics would leave none on: switch one on first")
    cur = conn.execute(
        "UPDATE profiles SET topic_filters_json = ?, updated_at = datetime('now')"
        " WHERE id = ? AND topic_filters_json IS ? AND is_draft = ?",
        (json.dumps(_without_unmatched_marks(full)), profile_id, raw, row["is_draft"]),
    )
    conn.commit()
    if cur.rowcount == 0:
        log.info("wizard.researcher_topics_retry", profile_id=profile_id)
        return None
    if not row["is_draft"]:
        from rag_lib.scheduler.jobs import request_rerank_refresh
        request_rerank_refresh(profile_id)
    log.info("wizard.researcher_topics", profile_id=profile_id, enabled=enabled,
             n_on=sum(1 for t in full.get("topics") or [] if t.get("on", True)))
    return full


def _researcher_topics_rebuilt(
    conn: sqlite3.Connection, profile_id: int, *, enabled: bool,
) -> dict:
    """A draft's list with the profile applied or taken off: rebuilt from
    the seeds, as step 3 shows it anyway (a new topic starts on there).
    With ``choices`` (a not-wanted choice being set) the aggregation settles
    the states itself: only a topic whose mark changed moves, so the
    user's other switches stay."""
    prior_on = _prior_topic_states(conn, profile_id)
    full = _aggregate_topics(conn, profile_id, new_on=True, profile_topics=True)
    topics = full.get("topics") or []
    if enabled:
        for t in topics:
            if t.get("profile"):
                # The state before the profile had its say, kept across re-applies.
                t.setdefault(researcher_topics.BEFORE, prior_on.get(t["id"], bool(t.get("on", True))))
                t["on"] = t["profile"] == researcher_topics.INTEREST
        return full
    for t in topics:
        _restore_before(t)
    return researcher_topics.strip(full)


def _researcher_topics_on_stored(
    conn: sqlite3.Connection, profile_id: int, stored: dict, *, enabled: bool,
) -> dict:
    """A saved interest's list with the profile applied or taken off,
    worked on the stored list itself.

    The stored list is what the user saved and what every scan searches
    by; rebuilding it from the seeds here (as a draft's is) replaced it
    with the recomputed one -- topics the seeds no longer give were
    dropped and topics they gained started off -- so applying the profile
    and taking it off again left a different interest (17 topics on, then
    8) with no warning. The edit page keeps showing the recomputed list;
    it replaces the stored one only when the user saves it there.

    Taking the profile off restores every marked topic to its remembered
    state and removes the topics only the profile brought. Applying starts
    from that restored list (so re-applying after the profile changed does
    not keep stale marks) and folds the profile in as the aggregation
    does: wanted topics on, not-wanted ones off, the prior state remembered.

    """
    topics = [dict(t) for t in stored.get("topics") or []]
    for t in topics:
        _restore_before(t)
    plain = researcher_topics.strip({**stored, "topics": topics})
    view = _researcher_topics_for(conn, profile_id)
    plain[researcher_topics.CHOICES] = researcher_topics.stored_choices(stored, view)
    if not enabled:
        return plain
    if view is None:  # pragma: no cover -- set_researcher_topics checked
        raise ValueError("this interest was not built from a researcher whose profile lists topics")
    if view.get("labels_unmapped") and researcher_topics.is_applied(stored):
        # Part of the profile (its text labels) cannot be read just now:
        # the list as last applied in full is better than one missing them.
        return stored
    before = {t["id"]: bool(t.get("on", True)) for t in plain.get("topics") or [] if t.get("id")}
    full = researcher_topics.apply(plain, view, basis=_overlap_basis(conn, profile_id))
    for t in full.get("topics") or []:
        if t.get("profile"):
            # A topic only the profile brings was not in the list: off is
            # the state it had, and the one a topic new to a saved interest
            # starts in -- should a seed bring it later, removing the
            # profile then leaves it off rather than widening the scan.
            t[researcher_topics.BEFORE] = before.get(t["id"], False)
            t["on"] = t["profile"] == researcher_topics.INTEREST
    return full


def _with_one_topic_from(stored: dict, new: dict | None, only: str, choices: dict[str, str]) -> dict:
    """``stored`` with the topic coded ``only`` as ``new`` has it (None: not
    in the list), and ``choices``. The topic keeps its stored state when
    its mark is the same; one that gains a mark is set as the mark says,
    its state before remembered; one that loses it goes back to that state
    (unless the user switched it since)."""
    code = researcher_topics.code
    out = dict(stored)
    topics, seen = [], False
    for old in stored.get("topics") or []:
        if not old.get("id") or code(old["id"]) != only:
            topics.append(old)
            continue
        seen = True
        if new is None:
            continue
        # The topic as ``new`` lists it (its source, should a seed carry it
        # now); its state and mark are settled below.
        t = {**old, **{k: v for k, v in new.items() if k not in ("on", "profile", researcher_topics.BEFORE)}}
        if "source" not in new:
            t.pop("source", None)
        mark = new.get("profile")
        if mark == old.get("profile"):
            topics.append(t)
            continue
        if old.get("profile"):
            _restore_before(t)
        if mark:
            t["profile"] = mark
            t.setdefault(researcher_topics.BEFORE, bool(t.get("on", True)))
            t["on"] = mark == researcher_topics.INTEREST
        else:
            t.pop("profile", None)
            t.pop(researcher_topics.BEFORE, None)
        topics.append(t)
    if not seen and new is not None:
        topics.append(dict(new))
    out["topics"] = topics
    out[researcher_topics.CHOICES] = dict(choices)
    out[researcher_topics.FLAG] = True
    return out


def _content_topics(conn: sqlite3.Connection, profile_id: int) -> dict[str, dict]:
    """Topics read from the text of the seeds OpenAlex does not have.

    Returns ``{topic_id: topic}`` for the topics no OpenAlex-matched seed
    has. A ``local:`` seed with no topics yet (uploaded before this
    existed, or while the embedder was down) is read now and stored on its
    paper row; there are only ever a few, so this stays cheap.
    """
    from ... import content_topics
    from ...db.repos import versions as versions_repo
    from ..settings import get_settings

    rows = []
    for oa_id in profiles_repo.list_seed_openalex_ids(conn, profile_id):
        prow = papers_repo.get_by_openalex_id(conn, oa_id)
        if prow is None:
            continue
        raw = prow["topics_json"]
        if versions_repo.is_unmatched(conn, oa_id) and not raw:
            raw = content_topics.try_topics_json(prow["title"], prow["abstract"], prow["body_text"],
                                                 settings=get_settings())
            if raw is not None:
                conn.execute("UPDATE papers SET topics_json = ? WHERE openalex_id = ?", (raw, oa_id))
                conn.commit()
        rows.append(raw)
    matched: set[str] = set()
    found: dict[str, dict] = {}
    for raw in rows:
        ids = content_topics.ids_from_content(raw)
        if ids:
            for t in json.loads(raw).get("topics") or []:
                if t.get("id") in ids:
                    found.setdefault(t["id"], t)
            continue
        d = json.loads(raw) if raw else {}
        for t in (d.get("topics") or []) + [d.get("primary_topic")]:
            if isinstance(t, dict) and t.get("id"):
                matched.add(t["id"])
    return {tid: t for tid, t in found.items() if tid not in matched}


def _keep_adopted_topics(stored: dict, topic_filters: dict) -> None:
    """Carry over the topics kept from a forgotten researcher's profile
    (``source: adopted``): no seed brings them back, and a rebuild from the
    seeds dropped them on the first save after the forget."""
    topics = topic_filters.setdefault("topics", [])
    have = {researcher_topics.code(t.get("id")) for t in topics if t.get("id")}
    for t in stored.get("topics") or []:
        if t.get("id") and t.get("source") == researcher_topics.ADOPTED and researcher_topics.code(t["id"]) not in have:
            # Without the marks, as ``_keep_stored_umls``: ``apply`` re-marks
            # the ones a live profile still has an opinion on. A stale
            # ``on_before_profile`` on an unmarked copy made "Remove profile
            # topics" override the user's later on/off choice.
            topics.append({k: v for k, v in t.items() if k not in ("profile", researcher_topics.BEFORE)})
            have.add(researcher_topics.code(t["id"]))


def _keep_content_topics(topic_filters: dict, from_content: dict[str, dict]) -> None:
    """Mark the content topics, and add back any the top-8 cut dropped.

    ``aggregate_topic_filters`` keeps the 8 most common topics; one read
    from a single PDF has a count of 1 and lost to the matched seeds' --
    the same squeeze that kept that paper out of the search before.
    """
    from ... import content_topics

    topics = topic_filters.setdefault("topics", [])
    have = {t.get("id") for t in topics}
    for t in topics:
        if t.get("id") in from_content:
            t["source"] = content_topics.SOURCE
    for tid, t in from_content.items():
        if tid not in have:
            topics.append({"id": tid, "display_name": t.get("display_name") or "", "count": 1, "papers": 1,
                           "source": content_topics.SOURCE})


def preview_topics(conn: sqlite3.Connection, *, user_id: int, slug: str) -> tuple[dict, set[str]]:
    """Topics for editing a saved interest, without changing it.

    ``aggregate_draft_topics`` (behind ``recompute-topics``) writes as it
    reads and switches every newly found topic *on* -- right for a draft,
    but on a live interest it widened the next scan before the user had
    seen the list. Here a new topic starts off, and nothing is written
    until ``save_topics``. Returns the filters and the ids that are new.

    On an Edit interest copy "new" is against the saved interest, not the
    copy: the first tick writes the whole list to the copy, and every other
    new topic lost its mark.
    """
    row = _require_profile(conn, user_id, slug)
    profile_id = int(row["id"])
    saved_id = int(row["edit_of"]) if row["edit_of"] is not None else profile_id
    prior = set(_prior_topic_states(conn, saved_id))
    topic_filters = _aggregate_topics(conn, profile_id, new_on=False)
    new = {t["id"] for t in topic_filters.get("topics") or [] if t.get("id") and t["id"] not in prior}
    return topic_filters, new


def save_topics(
    conn: sqlite3.Connection, *, user_id: int, slug: str, selected_topic_ids: list[str],
) -> dict:
    """Switch an interest's topics on or off: exactly ``selected_topic_ids`` on.

    Re-aggregates first, so a topic the seeds gained since the last
    aggregation can be switched on. Raises ``ValueError`` when topics exist
    but none would be on -- a scan needs at least one. Papers already found
    through a topic switched off stay; it only narrows the next scans.
    """
    row = _require_profile(conn, user_id, slug)
    profile_id = int(row["id"])

    def build() -> dict:
        full = _aggregate_topics(conn, profile_id, new_on=False)
        ids = {t["id"] for t in full.get("topics") or [] if t.get("id")}
        selected = set(selected_topic_ids) & ids
        if ids and not selected:
            raise ValueError("switch on at least one topic: a scan searches by them")
        return _prune_topic_filters(full, selected)

    pruned = _write_rebuilt_topics(conn, profile_id, build)
    # Topic names are the reranker's queries in topic mode.
    from rag_lib.scheduler.jobs import request_rerank_refresh
    request_rerank_refresh(profile_id)
    return pruned


def scan_outdated(conn: sqlite3.Connection, *, user_id: int, slug: str) -> dict:
    """Whether the recommendations no longer match the interest: the topics
    switched on, or the seeds, are not the ones its last successful scan
    searched and scored with (FUNCTIONAL_SPEC T.2-4, E.2-5). For a copy
    being edited, its topics and seeds against the saved interest's last
    scan, or against the saved interest itself when that scan recorded no
    seeds or topics (E.9). ``{"topics": bool, "seeds": bool}``; for a saved
    interest, both False with no scan that recorded them."""
    row = _require_profile(conn, user_id, slug)
    return outdated(conn, int(row["id"]))


def outdated(conn: sqlite3.Connection, profile_id: int) -> dict:
    """``scan_outdated`` by id."""
    row = conn.execute("SELECT id, edit_of FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    if row is None:
        return {"topics": False, "seeds": False}
    scanned = int(row["edit_of"]) if row["edit_of"] is not None else profile_id
    last = gather_runs_repo.last_scan(conn, scanned)
    topics, seeds = last if last is not None else (None, None)
    if row["edit_of"] is not None:
        # A copy whose interest's last scan recorded no seeds or topics (it
        # ran before migration 0032): the interest as saved stands in, so a
        # seed removed in this edit is still said (FUNCTIONAL_SPEC E.9).
        if topics is None:
            topics = enabled_topic_ids(_stored_topic_filters(conn, scanned))
        if seeds is None:
            seeds = profiles_repo.list_seed_openalex_ids(conn, scanned)
    now_topics = set(enabled_topic_ids(_stored_topic_filters(conn, profile_id)))
    now_seeds = set(profiles_repo.list_seed_openalex_ids(conn, profile_id))
    return {"topics": topics is not None and now_topics != set(topics),
            "seeds": seeds is not None and now_seeds != set(seeds)}


class NoLlmTopics(ValueError):
    """The interest has no saved research topics an LLM generated."""


def map_llm_topics(conn: sqlite3.Connection, settings: Any, *, user_id: int, slug: str) -> dict:
    """Add the OpenAlex topics nearest the interest's saved LLM research
    topics to its topic list, switched off (``llm_topics``).

    Only research topics an LLM generated are mapped (Prosopia's own, the
    cluster method's and hand-entered lists are not), as saved: an unsaved
    preview is not read. Returns ``{"topic_filters", "added", "n_labels"}``.

    The labels are embedded first, with no lock held; the stored list is
    then read, merged and written under one write lock, so a topic switched
    in another tab meanwhile is not undone.
    """
    row = _require_profile(conn, user_id, slug)
    profile_id = int(row["id"])
    if gather_runs_repo.has_unfinished_import(conn, profile_id):
        raise ImportInProgress(slug)
    record = interest_terms.load(row)
    if not record or record.get("source") not in llm_topics.LLM_RECORD_SOURCES or not record.get("saved", True):
        raise NoLlmTopics("generate research topics with Claude and save them first")
    labels = [t.get("label") for t in record.get("terms") or [] if t.get("label")]
    if not labels:
        raise NoLlmTopics("the saved research topics are empty")
    matches = llm_topics.map_labels(
        labels,
        embedding_model=settings.RADAR_UMLS_EMBEDDING_MODEL,
        cache_dir=str(settings.RADAR_UMLS_CACHE_DIR),
        min_similarity=settings.RADAR_LLM_TOPIC_MIN_SIMILARITY,
        extra_similarity=settings.RADAR_LLM_TOPIC_EXTRA_SIMILARITY,
    )
    if conn.in_transaction:
        conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        if profiles_repo.get_by_slug(conn, user_id, slug) is None:
            raise LookupError(f"interest '{slug}' not found")
        merged, added = llm_topics.merge(
            _stored_topic_filters(conn, profile_id), matches,
            max_additions=settings.RADAR_LLM_MAX_TOPIC_ADDITIONS,
            cache_dir=str(settings.RADAR_UMLS_CACHE_DIR),
        )
        conn.execute(
            "UPDATE profiles SET topic_filters_json = ?, updated_at = datetime('now') WHERE id = ?",
            (json.dumps(merged), profile_id),
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    log.info("wizard.llm_topics_mapped", profile_id=profile_id, n_labels=len(labels),
             n_matches=len(matches), n_added=len(added))
    return {"topic_filters": merged, "added": added, "n_labels": len(labels)}


# ---------------------------------------------------------------------------
# Step 4 — dry-run sweep + preview
# ---------------------------------------------------------------------------


def dry_run_draft(
    conn: sqlite3.Connection,
    settings: Any,
    *,
    user_id: int,
    slug: str,
    days: int = MAX_DRY_RUN_DAYS,
    thresholds: list[float] | None = None,
    gatherer: Any | None = None,
    reporter: Any | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    selected_topic_ids: list[str] | None = None,
) -> DraftDryRun:
    """Resolve the draft + invoke the dry-run compute.

    Thin wrapper that does the slug→draft lookup and hands off to
    :func:`compute_dry_run_for_profile`. The async dry-run job calls
    the compute function directly (it already has the profile_id from
    the kickoff request), bypassing this slug lookup.
    """
    row = _require_draft(conn, user_id, slug)
    return compute_dry_run_for_profile(
        conn,
        settings,
        profile_row=row,
        slug=slug,
        days=days,
        thresholds=thresholds,
        gatherer=gatherer,
        reporter=reporter,
        year_from=year_from,
        year_to=year_to,
        selected_topic_ids=selected_topic_ids,
    )


def compute_dry_run_for_profile(
    conn: sqlite3.Connection,
    settings: Any,
    *,
    profile_row: sqlite3.Row,
    slug: str,
    days: int = MAX_DRY_RUN_DAYS,
    thresholds: list[float] | None = None,
    gatherer: Any | None = None,
    reporter: Any | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    selected_topic_ids: list[str] | None = None,
) -> DraftDryRun:
    """Run a dry-run sweep + preview against the last ``days`` of OpenAlex.

    Returns a SweepRow per threshold plus a 10-card preview of the
    top-scoring candidates. Tests inject ``gatherer``; in production
    we build an ``OpenAlexGatherer`` from settings.

    ``reporter`` is an optional ``_ProgressReporter`` (or duck) the
    caller hands in when running under the scheduler — it stages step
    transitions through ``gather_runs.set_step``. ``embed_progress``
    is the caller's responsibility (it has to wrap ``select()``); pass
    ``None`` for the inline / test path which doesn't need progress
    reporting.
    """
    profile_id = int(profile_row["id"])

    days = max(1, min(int(days), MAX_DRY_RUN_DAYS))
    thresholds = sorted(thresholds or DEFAULT_THRESHOLDS)

    if reporter is not None:
        reporter.step("loading_profile", message="Loading draft seeds")
    profile = _load_profile(conn, profile_row)
    if not profile.papers:
        return DraftDryRun(sweep=[], preview=[])
    if selected_topic_ids is not None:
        # Step 3's ticks as sent, not only as stored: the route saved them,
        # but a topic-list GET running alongside rewrites the stored list
        # from what it read before that save.
        profile.topic_filters = _prune_topic_filters(profile.topic_filters or {}, set(selected_topic_ids))

    selector = _new_selector_for_draft(profile_row, profile)
    selector.fit(profile)

    gatherer = gatherer or OpenAlexGatherer(
        mailto=users_repo.contact_mailto(conn, profile_row["user_id"], settings.RADAR_DEFAULT_MAILTO))
    since = _days_ago(days)
    fetch_limit = capped_fetch_limit(settings, DRY_RUN_FETCH_LIMIT)

    log.info(
        "dry_run.openalex_query",
        slug=slug, days=days, since=since,
        limit=fetch_limit,
    )

    if reporter is not None:
        reporter.step("fetching", message="Querying OpenAlex")
    if year_from is not None:
        # A range of years instead of the last ``days``: each year sampled,
        # most cited first, so every year in the range is represented.
        from rag_lib import years as years_rules
        start, end = years_rules.fetch_span(year_from, year_to)
        per_year = years_rules.trial_per_year(end - start + 1)
        log.info("dry_run.year_range", slug=slug, start=start, end=end, per_year=per_year)
        candidates = gatherer.fetch_years(profile, start, end, per_year=per_year)
    else:
        candidates = gatherer.fetch(profile, since=since, limit=fetch_limit)
    # The seeds score ~1.0 against themselves; left in, they would set the
    # top of the distribution the threshold is suggested from.
    candidates = profiles_repo.without_seeds(
        candidates, profiles_repo.seed_keys(conn, profile_id))
    log.info(
        "dry_run.fetched",
        slug=slug, n=len(candidates),
        tier_used=getattr(gatherer, "last_tier_used", None),
        filter=getattr(gatherer, "last_filter_str", None),
        cost=dict(gatherer.cost()),
    )

    sample = random.sample(candidates, min(5, len(candidates)))
    for i, p in enumerate(sample):
        log.info(
            "dry_run.random_sample",
            slug=slug, idx=i,
            title=(p.title or "")[:160],
            abstract=(p.abstract or "")[:240],
        )

    if reporter is not None:
        reporter.step(
            "embedding",
            total=len(candidates),
            message=f"Embedding {len(candidates)} candidates",
        )
    # Caller wraps the select() call in embed_progress() when it wants
    # per-paper ticks — this function stays embedder-agnostic so the
    # inline / test path doesn't need a thread-local hook.
    ranked = selector.select(candidates, profile, threshold=None)

    if ranked:
        raws = [r[2].get("score_raw", r[0]) for r in ranked]
        arr = np.asarray(raws, dtype=float)
        log.info(
            "dry_run.cosine_distribution",
            slug=slug, n=len(arr),
            min=float(arr.min()), p50=float(np.percentile(arr, 50)),
            p90=float(np.percentile(arr, 90)), max=float(arr.max()),
            mean=float(arr.mean()), std=float(arr.std()),
        )

        def _log_top(label: str, key: str) -> None:
            top = sorted(ranked, key=lambda r: r[2].get(key, 0.0), reverse=True)[:5]
            for i, entry in enumerate(top):
                paper, b = entry[1], entry[2]
                log.info(
                    f"dry_run.top_by_{label}",
                    slug=slug, rank=i + 1,
                    centroid_cos=round(float(b.get("score_raw", 0.0)), 4),
                    max_seed_cos=round(float(b.get("score_max_seed", 0.0)), 4),
                    median_seed_cos=round(float(b.get("score_median_seed", 0.0)), 4),
                    pct=round(float(b.get("score_pct", 0.0)), 4),
                    title=(paper.title or "")[:160],
                )

        _log_top("centroid", "score_raw")
        _log_top("max_seed", "score_max_seed")
        _log_top("median_seed", "score_median_seed")

    sweep: list[SweepRow] = []
    for thr in thresholds:
        passing = [e for e in ranked if e[0] >= thr]
        top_title = passing[0][1].title if passing else ""
        sweep.append(SweepRow(
            th=round(float(thr), 4), n=len(passing), top=top_title or "",
        ))

    preview = _ranked_to_preview(
        ranked,
        slug=slug,
        active_topic_ids=_topic_ids(profile.topic_filters),
    )

    # Raw selector cosines for every fetched candidate. The wizard
    # renders a slider over these so the user picks θ against a real
    # histogram instead of a coarse grid of bucket cards. ``e[0]`` is the
    # selector's cosine here — the dry run does not rerank — but read
    # ``score_raw`` when present so this stays true if it ever does.
    # Rounded down as the Threshold tab does: 0.89996 shown as 0.9 would
    # pass a 0.9 bar here that the feed does not let it pass.
    scores = [floor4(e[2].get("score_raw", e[0])) for e in ranked]
    papers = sorted(
        (
            DryRunPaper(
                id=e[1].openalex_id or "", title=e[1].title or "",
                venue=e[1].venue or "", year=e[1].year, score=sc,
            )
            for e, sc in zip(ranked, scores)
        ),
        key=lambda p: p.score, reverse=True,
    )

    band = None
    try:
        band = calibration.seed_similarity_band(
            profile.seed_embeddings(profile.embedding_model)
        )
    except ValueError:
        band = None
    suggested = calibration.suggest_threshold(band["min"] if band else None, scores)
    rng = calibration.score_range(scores, band)

    return DraftDryRun(
        sweep=sweep, preview=preview, scores=scores, papers=papers,
        suggested_threshold=suggested,
        seed_similarity=_seed_similarity_schema(band),
        score_range=list(rng) if rng else None,
    )


# ---------------------------------------------------------------------------
# Commit — flip is_draft and register schedule
# ---------------------------------------------------------------------------


def commit_draft(
    conn: sqlite3.Connection,
    *,
    user_id: int,
    slug: str,
    threshold: float,
    selected_topic_ids: list[str],
    cron: str | None = None,
    tz: str | None = None,
) -> dict:
    """Get the draft ready to go live: fitted, its threshold and topics set.

    It stays a draft. It goes live (``go_live``) only when its first scan
    has succeeded (FUNCTIONAL_SPEC D.2-5, F2.3 Q1): ``POST /api/profiles``
    runs that scan next, and a scan that fails leaves the draft a draft.
    ``already_live`` in the result: it went live in another tab meanwhile.

    1. Prune ``topic_filters_json`` to the user's chosen topic ids
       (other levels — subfields/fields/domains — are preserved as
       gatherer hints; the user toggles topics specifically).
    2. Re-fit the chosen selector. When the fitted state includes a
       centroid (the centroid selector's case), store it as a BLOB on
       ``profiles.centroid`` for fast read-side access; the full
       selector state still goes into ``selector_config_json``.
    3. Update n_seed from the actual ``profile_seeds`` count.

    ``cron`` / ``tz`` are written by ``go_live``.
    """
    row = _require_draft(conn, user_id, slug)
    profile_id = int(row["id"])
    # Committing mid-import fitted the interest on the seeds attached so far;
    # the rest arrived on a live interest that nothing re-fitted.
    if gather_runs_repo.has_unfinished_import(conn, profile_id):
        raise ImportInProgress(slug)

    # The list is pruned from the stored one as read, and written only if
    # that is still what is stored: the fit takes seconds, and a profile-topic
    # choice or an apply made meanwhile in another tab was written over.
    first: dict | None = None
    for _attempt in range(3):
        raw = _raw_topic_filters(conn, profile_id)
        try:
            full_topics = json.loads(raw) if raw else {}
        except ValueError:
            full_topics = {}
        # The ticks are the user's for the list they saw; a topic changed
        # since by a profile-topic choice or an apply in another tab keeps
        # the state that set it.
        if first is None:
            first = full_topics
            selected = set(selected_topic_ids)
        else:
            selected = _ticks_over(first, full_topics, set(selected_topic_ids))
        # As save_topics: with every topic off, each scan gathers nothing.
        ids = {t["id"] for t in (full_topics or {}).get("topics") or [] if t.get("id")}
        if ids and not (selected & ids):
            raise ValueError("switch on at least one topic: a scan searches by them")
        pruned = _prune_topic_filters(full_topics, selected)

        def write(profile, sel_cfg, diag, centroid_blob, band, pruned=pruned, raw=raw):
            return profiles_repo.commit_draft(
                conn,
                profile_id,
                threshold=threshold,
                topic_filters=pruned,
                centroid=centroid_blob,
                selector_config=sel_cfg,
                coherence_median=diag.get("coherence_median"),
                coherence_iqr=diag.get("coherence_iqr"),
                coherence_bimodal=diag.get("coherence_bimodal"),
                fit_seeds=sel_cfg[FIT_SEEDS],
                band=band,
                topic_filters_was=raw,
                go_live=False,
            )
        try:
            flipped, _profile = _fit_and_write(
                conn, lambda: _require_draft(conn, user_id, slug),
                lambda r, prof: _new_selector_for_draft(r, prof, threshold=threshold),
                write, topic_filters=pruned,
            )
        except LookupError:
            flipped = 0                  # no longer a draft: committed elsewhere, or gone (below)
        if flipped:
            break
        now = profiles_repo.get_by_slug(conn, user_id, slug)
        if now is None or int(now["id"]) != profile_id:
            raise LookupError(f"draft '{slug}' not found")   # discarded in another tab while fitting
        if not now["is_draft"]:
            return {"slug": slug, "id": profile_id, "already_live": True}   # saved there first
        log.info("wizard.commit_after_topics_change", profile_id=profile_id)
    else:
        raise TopicsChanging(profile_id)
    return {"slug": slug, "id": profile_id, "already_live": False}


def go_live(conn: sqlite3.Connection, profile_id: int, *, cron: str | None = None, tz: str | None = None) -> None:
    """The draft's first scan succeeded: it is a live interest now, with its
    schedule, and what it brought in (``draft_owned``) is no longer the
    draft's. One transaction. ``LookupError`` when it is no longer a draft
    (deleted while the scan ran, or live already)."""
    try:
        if not profiles_repo.go_live(conn, profile_id):
            conn.rollback()
            raise LookupError(f"profile {profile_id} is no longer a draft")
        draft_owned_repo.clear(conn, profile_id)
        schedules_repo.upsert(conn, profile_id=profile_id, cron=cron, tz=tz)   # commits
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise


# ---------------------------------------------------------------------------
# Abandon
# ---------------------------------------------------------------------------


def delete_draft(
    conn: sqlite3.Connection, *, user_id: int, slug: str
) -> bool:
    row = profiles_repo.get_by_slug(conn, user_id, slug)
    if row is None or not row["is_draft"]:
        return False
    # Deleting cascades the import's run row: the job then failed on its
    # next seed and the researcher kept a partial paper list.
    if gather_runs_repo.has_unfinished_import(conn, int(row["id"])):
        raise ImportInProgress(slug)
    # Being saved: its first scan is running, and goes live when it succeeds.
    if gather_runs_repo.running_scan(conn, int(row["id"])) is not None:
        raise SavingInProgress(slug)
    # With what the draft brought in and nothing else uses: its uploads,
    # the researcher its import created and that import's papers (spec D).
    draft_owned_repo.discard(conn, int(row["id"]))
    return True


def remove_draft_seed(
    conn: sqlite3.Connection, *, user_id: int, slug: str, openalex_id: str
) -> bool:
    """Drop one seed from a draft, e.g. the off-topic half of the least-alike pair.

    Raises ``LookupError`` when the draft doesn't exist; returns False
    when the paper was not one of its seeds. Kept for the draft-only
    route; ``remove_seed`` is the general form.
    """
    _require_draft(conn, user_id, slug)
    return remove_seed(conn, user_id=user_id, slug=slug, openalex_id=openalex_id)


# ---------------------------------------------------------------------------
# Seeds of any profile, and re-fitting a live one
# ---------------------------------------------------------------------------


def _as_seed_id(conn: sqlite3.Connection, profile_id: int, openalex_id: str) -> str | None:
    """The stored form of ``openalex_id`` among this profile's seeds, or None.

    Seeds are stored as https://openalex.org/W… (or a synthetic
    ``prosopia:``/``local:`` id), and a caller may send the bare W-id.
    Both name the same seed.
    """
    seeds = set(profiles_repo.list_seed_openalex_ids(conn, profile_id))
    oa = (openalex_id or "").strip()
    if oa in seeds:
        return oa
    if oa.startswith("W") and f"https://openalex.org/{oa}" in seeds:
        return f"https://openalex.org/{oa}"
    return None


def embed_missing(
    conn: sqlite3.Connection, embedding_model: str, openalex_ids: list[str],
) -> int:
    """Embed those of ``openalex_ids`` that have no vector under this model.

    A seed without a vector is silently left out of the fit -- the
    selector only sees papers it can place -- so anything that adds a
    seed has to make sure it has one. Uploads and imports embed as they
    store; a paper attached from the feed does not, because candidates
    are scored without their vectors being kept (449 of 449 had none on
    2026-10-01). Same input text as every other path. Returns how many
    were embedded.
    """
    from ...embed import build_embedding_input
    from ...embedders import get_embedder

    embedder = None
    n = 0
    for oa_id in openalex_ids:
        if embeddings_repo.has(conn, oa_id, embedding_model):
            continue
        prow = papers_repo.get_by_openalex_id(conn, oa_id)
        if prow is None:
            continue
        if embedder is None:
            embedder = get_embedder(embedding_model)
        paper = Paper(
            doi=prow["doi"], openalex_id=prow["openalex_id"],
            title=prow["title"] or "", abstract=prow["abstract"] or "",
            year=prow["year"], venue=prow["venue"], body_text=prow["body_text"],
        )
        embeddings_repo.upsert(conn, oa_id, embedding_model,
                               embedder(build_embedding_input(paper)))
        n += 1
    if n:
        log.info("seeds.embedded_missing", model=embedding_model, n=n)
    return n


class SeedsChanging(RuntimeError):
    """Seeds were added or removed in another tab under every fit tried."""


def _fit_and_write(conn, reread, new_selector, write, *, topic_filters):
    """Fit the profile on its seeds and ``write`` the result, which
    raises ``profiles_repo.SeedsChanged`` when the seeds moved meanwhile:
    then fit again on the new ones, up to three times.

    ``reread()`` returns the profile row (and raises if it is gone);
    ``topic_filters`` None means the stored ones. Returns ``write``'s result
    and the profile fitted.
    """
    for _ in range(3):
        row = reread()
        profile_id = int(row["id"])
        tf = topic_filters if topic_filters is not None else profiles_repo.topic_filters(conn, profile_id)
        profile = _load_profile(conn, row, topic_filters=tf)
        selector = new_selector(row, profile)
        selector.fit(profile)
        sel_cfg = selector.config()
        # The seeds the fit was built from, not a second read: a seed
        # attached in between was listed as fitted, so no later scan
        # refitted for it.
        sel_cfg[FIT_SEEDS] = sorted(p.openalex_id for p in profile.papers if p.openalex_id)
        diag = sel_cfg.get("diagnostics_snapshot") or {}
        centroid_list = sel_cfg.get("centroid")
        try:
            band = calibration.seed_similarity_band(
                profile.seed_embeddings(profile.embedding_model)
            )
        except ValueError:
            band = None
        try:
            return write(profile, sel_cfg, diag,
                         encode_vector(centroid_list) if centroid_list else None, band), profile
        except profiles_repo.SeedsChanged:
            log.info("wizard.refit_after_seed_change", profile_id=profile_id)
    raise SeedsChanging(profile_id)


class ScanRunning(RuntimeError):
    """A scan of the interest is still running; a refit now would score part
    of the pool against one centroid and part against another."""


# The seed ids a live interest's fit was made from, stored inside
# ``selector_config_json``. The scan compares the seeds it has against this
# (not against the seeds at its own start): a seed attached under a scan
# whose end-of-run refit did not happen -- the scan failed, another run
# opened as it closed, the seed arrived between "Scan now" and the job --
# stayed unfitted for good otherwise.
FIT_SEEDS = "fit_seed_ids"


def fit_seed_ids(row) -> list[str] | None:
    """The seeds the stored fit was made from; None for a fit from before
    this was recorded."""
    sel_json = row["selector_config_json"] if row is not None else None
    if not sel_json:
        return None
    try:
        cfg = json.loads(sel_json)
    except (ValueError, TypeError):
        return None
    ids = cfg.get(FIT_SEEDS) if isinstance(cfg, dict) else None
    return sorted(str(i) for i in ids) if isinstance(ids, list) else None


def refit_profile(conn: sqlite3.Connection, *, user_id: int, slug: str,
                  own_run_id: int | None = None) -> dict:
    """Re-fit a live profile's selector on the seeds it has now.

    ``own_run_id``: the scan calling this holds the interest's scan lock and
    its own run row is open; that row does not count as "a scan running".

    The fitted half of ``commit_draft``. Until this existed the scheduler
    kept scoring with the centroid fitted at commit -- it reuses the
    persisted selector -- so seeds added to or removed from a live
    interest changed nothing in its feed, and ``POST /{key}/refit``
    returned a fixed placeholder string. Threshold and topics are left
    alone: they are the user's choices, not a function of the seeds.

    Raises ``LookupError`` for an unknown profile and ``ValueError`` for
    a draft (fitted at commit) or a profile with no embedded seed.
    """
    row = profiles_repo.get_by_slug(conn, user_id, slug)
    if row is None:
        raise LookupError(f"profile '{slug}' not found for current user")
    if row["is_draft"]:
        raise ValueError(f"'{slug}' is a draft; it is fitted when it is committed")
    t0 = time.monotonic()
    profile_id = int(row["id"])
    running = gather_runs_repo.running_scan(conn, profile_id)
    if running is not None and (own_run_id is None or int(running["id"]) != int(own_run_id)):
        raise ScanRunning(f"a scan of '{slug}' is still running; refit when it has finished")
    def reread():
        r = profiles_repo.get_by_slug(conn, user_id, slug)
        if r is None or int(r["id"]) != profile_id:
            raise LookupError(f"profile '{slug}' not found for current user")
        embed_missing(conn, r["embedding_model"],
                      profiles_repo.list_seed_openalex_ids(conn, profile_id))
        return r

    def selector_for(r, profile):
        if not any(profile.embedding_model in p.embeddings for p in profile.papers):
            raise ValueError(f"'{slug}' has no embedded seed papers to fit")
        return _new_selector_for_draft(r, profile, threshold=r["threshold"])

    def write(profile, sel_cfg, diag, centroid_blob, band):
        return profiles_repo.update_fit(
            conn,
            profile_id,
            centroid=centroid_blob,
            selector_config=sel_cfg,
            coherence_median=diag.get("coherence_median"),
            coherence_iqr=diag.get("coherence_iqr"),
            coherence_bimodal=diag.get("coherence_bimodal"),
            fit_seeds=sel_cfg[FIT_SEEDS],
            band=band,
        )
    _n, profile = _fit_and_write(conn, reread, selector_for, write, topic_filters=None)
    embedded = [p for p in profile.papers if profile.embedding_model in p.embeddings]
    # The stored candidates were scored against the old seeds; without this
    # a new seed changed nothing in the feed until the next morning's scan.
    from rag_lib.scoring.rescore import rescore_candidates
    rescored = rescore_candidates(conn, profile_id)
    elapsed = round(time.monotonic() - t0, 2)
    log.info("profile.refit", slug=slug, n_seeds=len(profile.papers),
             n_embedded=len(embedded), elapsed_s=elapsed, **rescored)
    return {"n_seeds": len(profile.papers), "n_embedded": len(embedded),
            "elapsed_s": elapsed, **rescored}


def remove_seed(
    conn: sqlite3.Connection, *, user_id: int, slug: str, openalex_id: str
) -> bool:
    """Drop one seed from a draft: a new interest's, or the copy Edit
    interest changes. A saved interest's seeds change only through that
    copy (``SavedInterest``, FUNCTIONAL_SPEC E.6 R12); nothing is refitted
    here -- the copy's save does not refit either, the next scan does.

    Raises ``LookupError`` for an unknown profile and ``ValueError`` when
    it would leave a copy with no seed (it becomes the interest, which
    needs one). Returns False when the paper was not one of its seeds.
    The topics stay as they were, on or off (FUNCTIONAL_SPEC E.2-3 as
    revised): the ones no seed left brings are only marked, worked out from
    the seeds when the list is read (``_keep_unmatched_seed_topics``).
    """
    row = profiles_repo.get_by_slug(conn, user_id, slug)
    if row is None:
        raise LookupError(f"profile '{slug}' not found for current user")
    if not row["is_draft"]:
        raise SavedInterest(slug)
    profile_id = int(row["id"])
    if gather_runs_repo.has_unfinished_import(conn, profile_id):
        raise ImportInProgress(slug)
    stored = _as_seed_id(conn, profile_id, openalex_id)
    if stored is None:
        return False
    copy = is_edit_copy(row)
    if copy and not [s for s in profiles_repo.list_seed_openalex_ids(conn, profile_id) if s != stored]:
        raise ValueError("an interest needs at least one seed; add another before removing this one")
    if not profiles_repo.detach_seed(conn, profile_id, stored, keep_embedded=None):
        if _as_seed_id(conn, profile_id, openalex_id) is None:
            return False  # removed meanwhile, in another tab
        raise ValueError("an interest needs at least one seed that can be placed; add another before removing this one")
    return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_draft(
    conn: sqlite3.Connection, user_id: int, slug: str
) -> sqlite3.Row:
    """A new interest's draft: what the wizard's own routes (coherence,
    topics, dry-run, seeds, save) take. An Edit interest copy is a draft
    too, but not one of these: saving it as a new interest made a second
    "X (editing)" interest that kept the original locked (spec E). It is
    changed through the interest routes and saved with save-edit."""
    row = profiles_repo.get_by_slug(conn, user_id, slug)
    if row is None or not is_new_draft(row):
        raise LookupError(f"draft '{slug}' not found")
    return row


def _new_selector_for_draft(
    row: sqlite3.Row, profile: Profile, *, threshold: float | None = None,
):
    """Instantiate the selector type the wizard recorded on the draft.

    Reads ``selector_config_json["type"]`` and looks the class up in the
    selectors registry. Drafts created before the selector field landed
    (or rows where the JSON is null/malformed) fall back to the
    centroid default so existing profiles keep working.
    """
    sel_json = row["selector_config_json"]
    sel_type = "centroid"
    if sel_json:
        try:
            cfg = json.loads(sel_json)
            if isinstance(cfg, dict) and isinstance(cfg.get("type"), str):
                sel_type = cfg["type"]
        except (TypeError, ValueError):
            pass
    cls = get_selector(sel_type)
    return cls(
        embedding_model=profile.embedding_model,
        threshold=threshold,
    )


def _require_profile(
    conn: sqlite3.Connection, user_id: int, slug: str
) -> sqlite3.Row:
    """Find a profile by slug — draft OR committed.

    Used by recompute paths that need to work after commit so the
    profile detail page can re-aggregate topics / re-fit coherence
    when the user uploads more seeds.
    """
    row = profiles_repo.get_by_slug(conn, user_id, slug)
    if row is None:
        raise LookupError(f"profile '{slug}' not found")
    return row


def _seed_embeddings_with_ids(
    conn: sqlite3.Connection, profile_id: int, model: str
) -> tuple[list[str], list[list[float]]]:
    rows = conn.execute(
        """
        SELECT ps.openalex_id AS oa, pe.vector
        FROM profile_seeds ps
        JOIN paper_embeddings pe ON pe.openalex_id = ps.openalex_id
        WHERE ps.profile_id = ? AND pe.embedding_model = ?
        """,
        (profile_id, model),
    ).fetchall()
    if not rows:
        return [], []
    from ...db.codec import decode_vector
    return [r["oa"] for r in rows], [decode_vector(r["vector"]).tolist() for r in rows]


def _seed_embeddings(
    conn: sqlite3.Connection, profile_id: int, model: str
) -> list[list[float]]:
    rows = conn.execute(
        """
        SELECT pe.vector
        FROM profile_seeds ps
        JOIN paper_embeddings pe USING (openalex_id)
        WHERE ps.profile_id = ? AND pe.embedding_model = ?
        """,
        (profile_id, model),
    ).fetchall()
    if not rows:
        return []
    from ...db.codec import decode_vector
    return [decode_vector(r["vector"]).tolist() for r in rows]


def _load_seed_papers(
    conn: sqlite3.Connection, profile_id: int
) -> list[Paper]:
    seed_ids = profiles_repo.list_seed_openalex_ids(conn, profile_id)
    out: list[Paper] = []
    for oa_id in seed_ids:
        prow = papers_repo.get_by_openalex_id(conn, oa_id)
        if prow is None:
            continue
        topics = papers_repo.decode_topics(prow)
        out.append(Paper.from_dict({
            "doi": prow["doi"],
            "openalex_id": prow["openalex_id"],
            "title": prow["title"],
            "abstract": prow["abstract"],
            "year": prow["year"],
            "venue": prow["venue"],
            "primary_topic": topics.get("primary_topic"),
            "topics": topics.get("topics") or [],
            "local_path": prow["local_path"],
            "body_text": prow["body_text"],
            "source": prow["source"],
        }))
    return out


def _load_profile(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    *,
    topic_filters: dict | None = None,
) -> Profile:
    profile_id = int(row["id"])
    embedding_model = row["embedding_model"]
    papers = _load_seed_papers(conn, profile_id)
    for paper in papers:
        vec = embeddings_repo.get(conn, paper.openalex_id, embedding_model)
        if vec is not None:
            paper.embeddings[embedding_model] = vec.tolist()
    if topic_filters is None:
        topic_filters = profiles_repo.topic_filters(conn, profile_id)
    return Profile(
        name=row["name"],
        papers=papers,
        topic_filters=topic_filters,
        embedding_model=embedding_model,
        threshold=row["threshold"],
    )


def _topic_ids(topic_filters: dict | None) -> list[str]:
    """Topics the dry-run preview should highlight a match against.

    The dry-run's gather goes through ``enabled_topic_ids``, so the
    preview has to use the same set or it credits a result to a topic
    that took no part in fetching it.
    """
    return enabled_topic_ids(topic_filters)


def _prior_topic_states(
    conn: sqlite3.Connection, profile_id: int
) -> dict[str, bool]:
    """Existing ``{topic_id: on}`` for a profile, empty if never aggregated."""
    try:
        existing = profiles_repo.topic_filters(conn, profile_id) or {}
    except (ValueError, TypeError):
        return {}
    return {
        t["id"]: bool(t.get("on", True))
        for t in (existing.get("topics") or [])
        if t.get("id")
    }


def _ticks_over(first: dict, now: dict, selected: set[str]) -> set[str]:
    """``selected`` (ticks made on the list ``first``) carried to the list
    ``now``: a topic whose profile mark or not-wanted choice changed in
    between, or that is new, keeps its state in ``now``."""
    code = researcher_topics.code
    before = {t["id"]: t.get("profile") for t in first.get("topics") or [] if t.get("id")}
    choices_then = researcher_topics.stored_choices(first)
    choices_now = researcher_topics.stored_choices(now)
    out = set(selected)
    for t in now.get("topics") or []:
        tid = t.get("id")
        if not tid:
            continue
        if (tid not in before or before[tid] != t.get("profile")
                or choices_then.get(code(tid)) != choices_now.get(code(tid))):
            (out.add if t.get("on", True) else out.discard)(tid)
    return out


def _prune_topic_filters(
    full: dict, selected_ids: set[str]
) -> dict:
    """Mark topics outside ``selected_ids`` as off; keep other levels intact.

    Deselection used to *delete* the entry, which threw away the one bit
    we need later: a topic missing from the list could equally mean "the
    user switched it off" or "it didn't exist last time we aggregated".
    Re-aggregating (``aggregate_draft_topics``) therefore resurrected
    everything the user had pruned. Flagging instead of deleting keeps
    the distinction, and the gatherer reads the flag via
    ``openalex_tiers.is_enabled``.
    """
    out = dict(full or {})
    out["topics"] = [
        {**t, "on": t.get("id") in selected_ids}
        for t in (full.get("topics") or [])
    ]
    return out


def _ranked_to_preview(
    ranked: list[tuple],
    *,
    slug: str,
    active_topic_ids: list[str],
) -> list[Card]:
    """Build preview Cards directly from the selector's ranked output."""
    out: list[Card] = []
    for entry in ranked[:PREVIEW_CARDS]:
        score, paper, breakdown = _unpack(entry)
        pct = (breakdown or {}).get("score_pct")
        out.append(_paper_to_card(
            paper, score, slug, active_topic_ids,
            percentile=float(pct) if pct is not None else None,
        ))
    return out


def _unpack(entry) -> tuple[float, Paper, dict]:
    if len(entry) == 3:
        return entry[0], entry[1], entry[2] or {}
    return entry[0], entry[1], {}


def _paper_to_card(
    paper: Paper,
    score: float,
    slug: str,
    active_topic_ids: list[str],
    *,
    percentile: float | None = None,
) -> Card:
    """Construct a ``Card`` from an in-memory ``Paper`` (preview only).

    Mirrors ``mappers.candidate_row_to_card`` but reads from a Paper
    dataclass instead of a sqlite row, since wizard previews don't
    persist candidates.
    """
    score = floor4(score)
    terms = _paper_terms(paper)
    matched = [t.display_name for t in _matched_topics(paper, set(active_topic_ids))]
    abstract = paper.abstract or ""
    pub_date = ""
    if paper.year is not None:
        pub_date = f"{int(paper.year)}-01-01"
    return Card(
        id=paper.openalex_id or "",
        title=paper.title or "",
        authors=[],
        venue=paper.venue or "",
        date=pub_date,
        doi=paper.doi,
        openalex=paper.openalex_id or "",
        profile=slug,
        score=score,
        similarity=score,
        bucket=bucket_for_similarity(
            score, threshold=None, seed_sim_min=None, percentile=percentile,
        ),
        abstract=abstract,
        mesh=list(paper.mesh or []),
        terms=terms,
        matched=matched,
        topicMatch=_topic_match(paper, set(active_topic_ids)),
        centroidCos=score,
        noveltyDelta=0.0,
        mins=_read_minutes(abstract),
    )


def _paper_terms(paper: Paper) -> list[str]:
    out: list[str] = []
    if paper.primary_topic and paper.primary_topic.display_name:
        out.append(paper.primary_topic.display_name)
    for t in paper.topics or []:
        if t.display_name and t.display_name not in out:
            out.append(t.display_name)
    return out


def _matched_topics(paper: Paper, active: set[str]) -> list[Topic]:
    if not active:
        return []
    out: list[Topic] = []
    if paper.primary_topic and paper.primary_topic.id in active:
        out.append(paper.primary_topic)
    for t in paper.topics or []:
        if t.id in active and t not in out:
            out.append(t)
    return out


def _topic_match(paper: Paper, active: set[str]) -> float:
    if not active:
        return 0.0
    ids: set[str] = set()
    if paper.primary_topic and paper.primary_topic.id:
        ids.add(paper.primary_topic.id)
    for t in paper.topics or []:
        if t.id:
            ids.add(t.id)
    if not ids:
        return 0.0
    return round(len(ids & active) / len(ids), 3)


def _finite(v: float) -> float:
    f = float(v)
    if not np.isfinite(f):
        return 0.0
    return f


def _days_ago(days: int) -> str:
    import datetime
    d = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    return d.strftime("%Y-%m-%d")


# How long the step-3 backfill may spend running the linker. Extraction
# is ~0.1s per paper warm and the mapping a few embedding calls, so this
# covers tens of papers; it exists for the case where every seed is
# pending at once.
_UMLS_BACKFILL_BUDGET_S = 30.0


def _extract_missing_umls(conn, settings, seed_ids: list[str]) -> None:
    """Fill in UMLS data for seeds that have none yet.

    Delegates to the same ``_try_extract_umls`` the upload path uses, so
    there is one implementation of what gets extracted and how it is
    stored — and one place where a failure is swallowed. Papers that
    already have concepts are left alone; re-running the linker over
    them would cost seconds each and change nothing.

    Stops starting new extractions once ``_UMLS_BACKFILL_BUDGET_S`` has
    elapsed. A paper it does not reach keeps its NULL column and is
    picked up the next time step 3 is opened, which is a better failure
    than a request that never returns. The budget is checked before each
    paper, so one pathologically slow paper can overrun it — the point is
    to bound the count, not to interrupt work in progress.
    """
    from .vault import _recently_failed, _try_extract_umls, _umls_input_text

    pending = []
    for oa_id in seed_ids:
        row = papers_repo.get_by_openalex_id(conn, oa_id)
        # A recent failure waits out its back-off here too: step 3 is
        # loaded again and again, and each time it re-ran the linker on
        # every failing seed for up to the whole budget.
        if row is not None and not row["umls_concepts_json"] and not _recently_failed(oa_id):
            pending.append(row)
    if not pending:
        return

    log.info("wizard.umls_extract_on_demand", n=len(pending))
    deadline = time.monotonic() + _UMLS_BACKFILL_BUDGET_S
    done = 0
    for row in pending:
        if time.monotonic() >= deadline:
            log.warning(
                "wizard.umls_extract_budget_exhausted",
                done=done, remaining=len(pending) - done,
            )
            break
        _try_extract_umls(
            conn,
            settings,
            row["openalex_id"],
            _umls_input_text(row["title"], row["abstract"], row["body_text"]),
        )
        done += 1


# Per interest, how the last UMLS merge went, for the topic pages to warn
# when UMLS topics could not be (fully) brought in. In memory: it describes
# the list the user is looking at, which was built moments ago.
_UMLS_MERGE_STATE: dict[int, dict] = {}


def umls_topics_state(profile_id: int) -> dict:
    """``{"available", "reason", "n"}`` of the interest's last topic aggregation.

    ``reason``: ``busy`` (database locked), ``error`` (anything else
    failed), ``bad_data`` (``n`` seeds' stored UMLS data unreadable; they
    are sent back to extraction), ``pending`` (``n`` seeds not extracted
    yet) or ``failed`` (``n`` seeds whose extraction failed; retried after
    a back-off). In each case the stored UMLS topics of those seeds are kept.
    """
    return dict(_UMLS_MERGE_STATE.get(profile_id) or {"available": True, "reason": None, "n": 0})


def _keep_stored_umls(conn: sqlite3.Connection, profile_id: int, topic_filters: dict,
                      unread: set[str] | None = None) -> dict:
    """``topic_filters`` plus the UMLS topics the interest already had.

    The list is rebuilt from the seeds on every aggregation and then
    written back, so a merge that could not run (or could not read every
    seed) used to drop every UMLS topic -- and the on/off the user had set
    on it. Kept as stored; the next complete merge settles them.

    ``unread``: the seeds that could not be read this time. Only topics
    that came from one of them are kept (each UMLS topic records its
    ``seeds``), so a topic of a seed the user removed does not linger
    while some other seed is unreadable. None (the whole merge failed):
    every stored UMLS topic is kept.
    """
    try:
        stored = profiles_repo.topic_filters(conn, profile_id) or {}
    except Exception:  # noqa: BLE001 -- the database may be what failed
        return topic_filters
    have = {t.get("id") for t in topic_filters.get("topics") or []}
    keep = [dict(t) for t in stored.get("topics") or []
            if t.get("source") == "umls" and t.get("id") and t["id"] not in have
            # Topics stored before ``seeds`` was recorded are kept as before.
            and (unread is None or not t.get("seeds") or unread & set(t["seeds"]))]
    # Within the same bounds as a merge: a name the list has is not added
    # twice, and the interest keeps at most MAX_TOPIC_ADDITIONS UMLS topics
    # (the stored ones are in the merge's order, surest first).
    from ..settings import get_settings

    names = {(t.get("display_name") or "").lower() for t in topic_filters.get("topics") or []}
    room = get_settings().RADAR_UMLS_MAX_TOPIC_ADDITIONS - sum(
        t.get("source") == "umls" for t in topic_filters.get("topics") or [])
    keep = [t for t in keep if (t.get("display_name") or "").lower() not in names][:max(room, 0)]
    if not keep:
        return topic_filters
    # The researcher's marks are put on afresh by ``_aggregate_topics`` when
    # the profile is applied; a kept copy must not carry them from a time it was.
    for t in keep:
        t.pop("profile", None)
        t.pop(researcher_topics.BEFORE, None)
    return {**topic_filters, "topics": [*(topic_filters.get("topics") or []), *keep]}


def _try_merge_umls_topics(
    conn: sqlite3.Connection,
    profile_id: int,
    topic_filters: dict,
    also_have: list[dict] | None = None,
) -> dict:
    """``_merge_umls_topics``, never raising; records how it went (``umls_topics_state``)."""
    state = {"available": True, "reason": None, "n": 0}
    try:
        merged = _merge_umls_topics(conn, profile_id, topic_filters, state, also_have)
    except Exception as exc:  # noqa: BLE001
        locked = isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower()
        state.update(available=False, reason="busy" if locked else "error", n=0)
        log.warning(
            "wizard.umls_merge_skipped",
            profile_id=profile_id,
            reason=type(exc).__name__,
            detail=str(exc)[:200],
        )
        merged = _keep_stored_umls(conn, profile_id, topic_filters)
    _UMLS_MERGE_STATE[profile_id] = state
    return merged


def forget_stale_umls_mappings(conn: sqlite3.Connection, seed_ids: list[str]) -> None:
    """Send back to extraction the seeds whose stored topic mapping was made
    by the earlier rules (``topic_mapper.stale_mapping``): the topics are now
    chosen on the concepts' confidence, which those did not record."""
    from ...umls.topic_mapper import stale_mapping

    stale = []
    for oa_id in seed_ids:
        row = conn.execute("SELECT umls_concepts_json, umls_mapped_topics_json FROM papers WHERE openalex_id = ?",
                           (oa_id,)).fetchone()
        if row is None or row[0] is None:
            continue      # not extracted yet
        try:
            concepts = json.loads(row[0])
            entries = json.loads(row[1]) if row[1] else []
        except ValueError:
            continue      # unreadable: the merge marks it as bad data
        # Concepts and no mapping at all: the earlier rules found no topic
        # near enough to any of them (now each concept has its nearest).
        if (row[1] is None and concepts) or (
                isinstance(entries, list) and stale_mapping([e for e in entries if isinstance(e, dict)])):
            stale.append(oa_id)
    if stale:
        conn.executemany(
            "UPDATE papers SET umls_concepts_json = NULL, umls_mapped_topics_json = NULL "
            "WHERE openalex_id = ?", [(oa,) for oa in stale])
        conn.commit()
        log.info("wizard.umls_stale_mappings", n=len(stale))


def _merge_umls_topics(
    conn: sqlite3.Connection,
    profile_id: int,
    topic_filters: dict,
    state: dict,
    also_have: list[dict] | None = None,
) -> dict:
    """Best-effort merge of UMLS-mapped topics into topic_filters.

    Reads ``umls_mapped_topics_json`` from each seed paper and calls
    ``merge_umls_topics()`` to append novel topics. Raises on failure;
    ``_try_merge_umls_topics`` keeps the stored UMLS topics then. Seeds
    whose UMLS data is missing or unreadable are recorded in ``state``,
    and the stored UMLS topics are kept alongside the merge.

    A seed whose extraction has not landed yet is computed here rather
    than skipped. This began as a race: ``upload()`` ran extraction on a
    daemon thread and returned immediately, while the wizard reached this
    point seconds later — the user uploads in step 1 and clicks through
    to step 3 — so the column was still NULL and step 3 showed only the
    OpenAlex topics. Upload is synchronous now, so the race is gone and
    the usual case is nothing pending. What is left is the genuine
    backfill: papers whose extraction failed into the blanket ``except``
    in ``_try_extract_umls``, and papers uploaded while the feature was
    switched off.

    Per paper the cost is a tenth of a second once the models are warm,
    but nothing bounded how many papers that covered, and this runs
    inside a request. ``_extract_missing_umls`` therefore works to a wall
    clock rather than to the seed count; whatever it does not reach stays
    pending for the next call.
    """
    from ..settings import get_settings
    settings = get_settings()
    if not settings.RADAR_UMLS_ENABLED:
        return topic_filters

    from ...umls.topic_mapper import MappedTopic, merge_umls_topics

    seed_ids = profiles_repo.list_seed_openalex_ids(conn, profile_id)
    if not seed_ids:
        return topic_filters

    forget_stale_umls_mappings(conn, seed_ids)
    _extract_missing_umls(conn, settings, seed_ids)

    from .vault import _recently_failed

    umls_per_paper: list[list[MappedTopic]] = []
    paper_ids: list[str] = []     # the seed each umls_per_paper entry came from
    bad: list[str] = []
    pending: list[str] = []
    for oa_id in seed_ids:
        row = papers_repo.get_by_openalex_id(conn, oa_id)
        if row is None:
            continue
        if row["umls_concepts_json"] is None:
            pending.append(oa_id)     # not extracted yet, or its extraction failed
            continue
        raw = row["umls_mapped_topics_json"]
        if not raw:
            continue
        try:
            umls_per_paper.append([
                MappedTopic(
                    topic_id=e["topic_id"],
                    display_name=e["display_name"],
                    similarity=e["similarity"],
                    source_cui=e["source_cui"],
                    source_name=e["source_name"],
                    confidence=e["confidence"],
                )
                for e in json.loads(raw)
            ])
            paper_ids.append(oa_id)
        except (ValueError, KeyError, TypeError):
            bad.append(oa_id)     # one unreadable seed no longer sinks the rest

    if bad:
        # Back to extraction: NULL is what every extraction path picks up.
        conn.executemany(
            "UPDATE papers SET umls_concepts_json = NULL, umls_mapped_topics_json = NULL "
            "WHERE openalex_id = ?", [(oa,) for oa in bad])
        conn.commit()
        log.warning("wizard.umls_bad_data", profile_id=profile_id, n=len(bad))
        state.update(available=False, reason="bad_data", n=len(bad))
    elif pending:
        # "failed": every one is waiting out a failed extraction's back-off,
        # so a reload will not help for hours; "pending": some are just
        # not read yet.
        failed = all(_recently_failed(oa) for oa in pending)
        state.update(available=False, reason="failed" if failed else "pending", n=len(pending))
    unread = set(bad) | set(pending)

    if not umls_per_paper:
        return _keep_stored_umls(conn, profile_id, topic_filters, unread) if unread else topic_filters

    _merged = merge_umls_topics(
        topic_filters,
        umls_per_paper,
        min_confidence=settings.RADAR_UMLS_TOPIC_MIN_CONFIDENCE,
        per_paper=settings.RADAR_UMLS_TOPICS_PER_PAPER,
        max_additions=settings.RADAR_UMLS_MAX_TOPIC_ADDITIONS,
        cache_dir=str(settings.RADAR_UMLS_CACHE_DIR),
        also_have=also_have,
    )

    # Which seeds chose each UMLS topic, so that a later merge that cannot
    # read some seeds keeps only their topics (_keep_stored_umls).
    for t in _merged.get("topics") or []:
        if t.get("source") == "umls" and "papers" in t:
            t["seeds"] = [paper_ids[i] for i in t.pop("papers")]
    return _keep_stored_umls(conn, profile_id, _merged, unread) if unread else _merged


def set_not_interest_choice(
    conn: sqlite3.Connection, *, user_id: int, slug: str, topic_id: str, choice: str,
) -> dict:
    """Set what one of the profile's not-wanted topics does on this
    interest: ``filter`` it out, ``keep`` it as the other sources have it,
    or ``add`` it as a topic of the interest's own. The profile must be
    applied, and ``topic_id`` one of the topics it does not want. Raises as
    :func:`set_researcher_topics`."""
    if choice not in researcher_topics.CHOICE_VALUES:
        raise ValueError(f"choice must be one of {list(researcher_topics.CHOICE_VALUES)}")
    row = _require_profile(conn, user_id, slug)
    view = _researcher_topics_for(conn, int(row["id"])) if row["researcher_id"] is not None else None
    codes = {researcher_topics.code(t["id"]) for t in (view or {}).get("unwanted") or []}
    if researcher_topics.code(topic_id) not in codes:
        raise ValueError(f"{topic_id} is not a topic the researcher's profile does not want")
    return set_researcher_topics(conn, user_id=user_id, slug=slug, enabled=True,
                                 choice=(topic_id, choice))
