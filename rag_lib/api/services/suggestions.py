"""Suggested interests — two or three topics found in a researcher's papers.

A researcher who publishes on two things has two interests waiting in
their profile. The papers are already embedded (that is what importing
a researcher does), so finding those groups is a read: pairwise
cosines, an agglomerative tree, and a cut chosen on the calibrated
scale in ``rag_lib.calibration``.

The design choices, and why:

- **Average-linkage agglomerative clustering** over cosine distance,
  in plain numpy. A profile has tens to a few hundred papers, which is
  small enough for the O(n²) matrix and the O(n³) worst case; the
  method needs no k up front and no extra dependency.
- **At most three suggestions, and not every paper has to be in one.**
  More than three is a list of papers, not a set of interests, and the
  wizard's own coherence check is the place to refine one. The tree is
  cut progressively deeper; at each depth the groups that are big
  enough (``MIN_PAPERS``) and tight enough (median pairwise cosine at
  the calibrated "focused" band) are the candidates, and the first
  depth that yields three such groups wins, largest first. Papers that
  fall outside those groups — the one-off reviews, the early work in
  another field — are simply left out, which is what a researcher
  would do by hand. If the whole set is already focused, the single
  suggestion is all of it.
- **Loose papers are kept, unticked.** A paper whose cosine to its
  group's centroid sits below the random-same-field floor (0.86) is
  reported separately so the UI can leave it out by default. Nothing is
  hidden; the user sees it and can tick it back.
- **Names come from OpenAlex topics** already on the paper rows: the
  most common primary topic within the group, preferring one that is
  distinctive for it. A group of synthetic (unresolved) papers has no
  topics and gets a name from its most typical paper's title.

All of this is a suggestion, not a verdict. The wizard's coherence
check runs on whichever set the user accepts.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from ... import calibration
from ...db import decode_vector
from ...db.repos import researchers as researchers_repo


log = structlog.get_logger("rag_lib.api.services.suggestions")


MAX_SUGGESTIONS = 3
# A group smaller than this is a couple of papers, not an interest.
MIN_PAPERS = 3
# Fewer embedded papers than this and the only honest suggestion is
# "all of them"; there is nothing to split.
MIN_TO_SPLIT = 8
# A group counts as an interest when its median pairwise cosine reaches
# the calibrated "focused" band (docs/CALIBRATION.md); if no depth of
# the tree yields one, the bar drops to "broad" before giving up.
GROUP_TIGHT = calibration.COHERENCE_FOCUSED
GROUP_LOOSE = calibration.COHERENCE_BROAD
# How deep to cut the tree while looking for groups.
MAX_DEPTH = 12
# Below this cosine to the group's centroid a paper is as related as a
# random paper from the same field; it stays listed but unticked.
LOOSE_FLOOR = 0.86
# Do not let a pathological ORCID import (a thousand works) turn the
# O(n³) tree into a minute of CPU: cluster the most recent N.
MAX_PAPERS = 400


@dataclass
class Suggestion:
    name: str
    paper_ids: list[str]
    loose_ids: list[str]
    topics: list[dict[str, Any]]
    coherence_median: float | None
    agreement: int | None
    label: str
    seed_titles: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------


def _unit(vecs: np.ndarray) -> np.ndarray:
    arr = np.asarray(vecs, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def average_linkage_tree(sim: np.ndarray) -> list[tuple[int, int]]:
    """Merge order for average-linkage clustering on a similarity matrix.

    Returns ``n - 1`` merges as ``(a, b)`` pairs of cluster indices,
    where new clusters take indices ``n, n + 1, …`` in merge order (the
    same convention as scipy's linkage matrix). Similarity rather than
    distance so the Lance–Williams update is a size-weighted mean, which
    is what "average linkage" means here.
    """
    n = sim.shape[0]
    if n < 2:
        return []
    s = sim.astype(np.float64).copy()
    np.fill_diagonal(s, -np.inf)
    size = np.ones(n, dtype=np.float64)
    alive = np.ones(n, dtype=bool)
    ids = list(range(n))
    merges: list[tuple[int, int]] = []
    next_id = n
    for _ in range(n - 1):
        masked = np.where(alive[:, None] & alive[None, :], s, -np.inf)
        flat = int(np.argmax(masked))
        i, j = divmod(flat, n)
        if i > j:
            i, j = j, i
        merges.append((ids[i], ids[j]))
        # Cluster i absorbs j; its similarity to every other live cluster
        # is the size-weighted mean of the two.
        wi, wj = size[i], size[j]
        s[i, :] = (wi * s[i, :] + wj * s[j, :]) / (wi + wj)
        s[:, i] = s[i, :]
        s[i, i] = -np.inf
        size[i] = wi + wj
        alive[j] = False
        ids[i] = next_id
        next_id += 1
    return merges


def cut_tree(merges: list[tuple[int, int]], n: int, k: int) -> list[list[int]]:
    """The ``k`` clusters left after undoing the last ``k - 1`` merges."""
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    stop = max(0, len(merges) - (k - 1))
    for step, (a, b) in enumerate(merges):
        if step >= stop:
            break
        members[n + step] = members.pop(a) + members.pop(b)
    return sorted(members.values(), key=lambda m: (-len(m), m[0]))


def _median_pairwise(sim: np.ndarray, idx: list[int]) -> float | None:
    if len(idx) < 2:
        return None
    sub = sim[np.ix_(idx, idx)]
    iu = np.triu_indices(len(idx), k=1)
    return float(np.median(sub[iu]))


def choose_groups(
    sim: np.ndarray,
    merges: list[tuple[int, int]],
    *,
    max_k: int = MAX_SUGGESTIONS,
    min_papers: int = MIN_PAPERS,
) -> tuple[list[list[int]], str | None]:
    """Up to ``max_k`` tight, sizeable groups; papers outside them are left out.

    Returns the groups (largest first) and a note when the answer is
    degenerate: the whole set is one focused topic, there are too few
    papers to split, or no group met even the relaxed bar.
    """
    n = sim.shape[0]
    everything = [list(range(n))]
    whole = _median_pairwise(sim, list(range(n)))
    if n < MIN_TO_SPLIT:
        return everything, f"Fewer than {MIN_TO_SPLIT} papers, so the one suggestion is all of them."
    if whole is not None and whole >= GROUP_TIGHT:
        return everything, "These papers already read as one focused topic."

    # Deeper cuts for bigger profiles: 250 papers need more than a
    # dozen clusters before a tight 30-paper topic separates out.
    depth = min(max(MAX_DEPTH, n // 10), n // min_papers)

    def search(bar: float) -> list[list[int]]:
        best: list[list[int]] = []
        for k in range(2, depth + 1):
            groups = cut_tree(merges, n, k)
            good = [
                g for g in groups
                if len(g) >= min_papers and (_median_pairwise(sim, g) or 0.0) >= bar
            ]
            good.sort(key=lambda g: (-len(g), g[0]))
            if len(good) >= max_k:
                return good[:max_k]
            if len(good) > len(best) or (
                len(good) == len(best) and sum(map(len, good)) > sum(map(len, best))
            ):
                best = good
        return best

    # Tight groups first; but two or three broad groups beat one tight
    # sliver, so the relaxed bar is consulted whenever the strict one
    # did not find a full set.
    tight = search(GROUP_TIGHT)
    if len(tight) >= max_k:
        return tight, None
    loose = search(GROUP_LOOSE)
    chosen = tight
    if len(loose) > len(tight) or (
        len(loose) == len(tight) and sum(map(len, loose)) > sum(map(len, tight))
    ):
        chosen = loose
    if chosen:
        return chosen, None
    return everything, "No group of these papers stood out; the one suggestion is all of them."


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def _topics_of(row: sqlite3.Row) -> tuple[dict | None, list[dict]]:
    raw = row["topics_json"] if row["topics_json"] else None
    if not raw:
        return None, []
    try:
        data = json.loads(raw)
    except ValueError:
        return None, []
    return data.get("primary_topic"), list(data.get("topics") or [])


def _name_for(
    rows: list[sqlite3.Row], members: list[int], others: list[int]
) -> tuple[str, list[dict[str, Any]]]:
    """A name and the top topics for a group.

    Counts primary topics inside the group and outside it; the score is
    inside share minus outside share, so a topic every paper of the
    researcher carries does not name every group the same way.
    """
    inside: Counter[str] = Counter()
    outside: Counter[str] = Counter()
    names: dict[str, str] = {}
    for i in members:
        primary, topics = _topics_of(rows[i])
        for t in ([primary] if primary else []) + topics[:3]:
            tid = t.get("id") or t.get("display_name")
            if not tid:
                continue
            names[tid] = t.get("display_name") or str(tid)
            inside[tid] += 2 if t is primary else 1
    for i in others:
        primary, topics = _topics_of(rows[i])
        for t in ([primary] if primary else []) + topics[:3]:
            tid = t.get("id") or t.get("display_name")
            if tid:
                outside[tid] += 2 if t is primary else 1
    if not inside:
        title = (rows[members[0]]["title"] or "").strip()
        return (title[:60] or "Untitled group"), []
    n_in = max(1, len(members))
    n_out = max(1, len(others))
    scored = sorted(
        inside,
        key=lambda tid: (-(inside[tid] / n_in - 0.5 * outside[tid] / n_out), -inside[tid], names[tid]),
    )
    top = [{"id": tid, "name": names[tid], "count": inside[tid]} for tid in scored[:3]]
    return names[scored[0]], top


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def suggest_interests(
    conn: sqlite3.Connection,
    *,
    researcher_id: int,
    embedding_model: str,
) -> dict[str, Any]:
    """Two or three interests worth building from this researcher, or one.

    Returns ``{embedding_model, n_papers, n_embedded, suggestions, note}``.
    ``note`` explains a degenerate answer (too few papers, nothing
    embedded under the model, or no split worth making).
    """
    rows = researchers_repo.list_papers(conn, researcher_id)
    ids = [r["openalex_id"] for r in rows]
    vecs: dict[str, np.ndarray] = {}
    if ids:
        marks = ",".join("?" for _ in ids)
        for r in conn.execute(
            f"SELECT openalex_id, vector FROM paper_embeddings WHERE embedding_model = ? AND openalex_id IN ({marks})",
            (embedding_model, *ids),
        ):
            vecs[r["openalex_id"]] = np.asarray(decode_vector(r["vector"]), dtype=np.float64)

    kept = [r for r in rows if r["openalex_id"] in vecs]
    note: str | None = None
    if len(kept) > MAX_PAPERS:
        kept = kept[:MAX_PAPERS]  # list_papers is newest first
        note = f"Grouped the {MAX_PAPERS} most recent papers."
    n = len(kept)
    result: dict[str, Any] = {
        "embedding_model": embedding_model,
        "n_papers": len(rows),
        "n_embedded": len(vecs),
        "n_grouped": 0,
        "suggestions": [],
        "note": note,
    }
    if n == 0:
        result["note"] = "No papers are embedded under this model yet." if rows else "This profile has no papers."
        return result

    arr = _unit(np.stack([vecs[r["openalex_id"]] for r in kept]))
    sim = arr @ arr.T
    merges = average_linkage_tree(sim)
    groups, why = choose_groups(sim, merges)
    if why and result["note"] is None:
        result["note"] = why
    covered = sum(len(g) for g in groups)
    result["n_grouped"] = covered

    suggestions: list[Suggestion] = []
    for g in groups:
        others = [i for i in range(n) if i not in set(g)]
        centroid = arr[g].mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-12)
        to_centroid = {i: float(arr[i] @ centroid) for i in g}
        ordered = sorted(g, key=lambda i: -to_centroid[i])
        tight = [i for i in ordered if to_centroid[i] >= LOOSE_FLOOR]
        loose = [i for i in ordered if to_centroid[i] < LOOSE_FLOOR]
        if len(tight) < 1:
            tight, loose = ordered, []
        median = _median_pairwise(sim, tight)
        label = calibration.coherence_label(median, None, len(tight))
        name, topics = _name_for(kept, tight, others)
        suggestions.append(Suggestion(
            name=name,
            paper_ids=[kept[i]["openalex_id"] for i in tight],
            loose_ids=[kept[i]["openalex_id"] for i in loose],
            topics=topics,
            coherence_median=round(median, 4) if median is not None else None,
            agreement=calibration.agreement_score(median) if median is not None and len(tight) >= 2 else None,
            label=label,
            seed_titles=[(kept[i]["title"] or "")[:120] for i in tight[:3]],
        ))

    # Distinct names: two groups sharing a top topic get their second one.
    seen: Counter[str] = Counter(s.name for s in suggestions)
    for s in suggestions:
        if seen[s.name] > 1 and len(s.topics) > 1:
            s.name = f"{s.topics[0]['name']} · {s.topics[1]['name']}"

    result["suggestions"] = [s.__dict__ for s in suggestions]
    log.info(
        "suggestions.done", researcher_id=researcher_id, n=n,
        k=len(suggestions), sizes=[len(s.paper_ids) for s in suggestions],
    )
    return result
