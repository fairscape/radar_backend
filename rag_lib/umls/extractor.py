"""SciSpacy NER + UMLS linker for biomedical concept extraction.

Lazy-loads the spaCy model + UMLS linker on first call (same pattern as
``rag_lib.embedders._load_specter2``). Subsequent calls reuse the cached
pipeline.

The ``SCISPACY_CACHE`` environment variable controls where the ~1 GB
UMLS knowledge base is stored on disk. Set it before importing this
module to avoid writing to ``~/.scispacy/``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, asdict
from threading import Lock

from .semantic_types import is_relevant, get_type_name

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UmlsConcept:
    """A single extracted UMLS concept."""

    cui: str  # e.g. "C0011849"
    name: str  # preferred name, e.g. "Diabetes Mellitus, Type 2"
    tui: str  # semantic type unique identifier, e.g. "T047"
    semantic_type: str  # human-readable, e.g. "Disease or Syndrome"
    confidence: float  # linker similarity score in [0, 1]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------

_nlp = None
_nlp_lock = Lock()


def _get_nlp(spacy_model: str, cache_dir: str | None, threshold: float):
    """Load spaCy model + UMLS linker, cached globally."""
    global _nlp

    if _nlp is not None:
        return _nlp

    with _nlp_lock:
        if _nlp is not None:
            return _nlp

        # Set cache dir before importing scispacy
        if cache_dir:
            os.environ["SCISPACY_CACHE"] = str(cache_dir)

        import spacy
        import scispacy  # noqa: F401
        from scispacy.abbreviation import AbbreviationDetector  # noqa: F401 — registers 'abbreviation_detector'
        from scispacy.linking import EntityLinker  # noqa: F401 — registers 'scispacy_linker' factory

        log.info("Loading SciSpacy model: %s", spacy_model)
        nlp = spacy.load(spacy_model)

        # Must come before the linker. The linker's own
        # ``resolve_abbreviations`` is gated on
        # ``Doc.has_extension("abbreviations")`` (scispacy linking.py), which
        # only this pipe registers — without it the flag is silently a no-op
        # and every acronym is linked on its surface form. That is not a
        # marginal loss of quality: an exact string match against a UMLS
        # alias scores ~1.0, higher than any real multi-word concept, so
        # unexpanded acronyms took the top of the confidence ranking and the
        # ``max_concepts`` truncation kept them. Measured on the 11 seed
        # papers in the deployment DB, the linker was resolving "eGFR" to
        # Epidermal Growth Factor Receptor (0.999) on a chronic-kidney-
        # disease trial, "BG" to O(6)-benzylguanine (0.999) instead of blood
        # glucose, "SE" to selenium (0.999) and "RF" to the UMLS finding
        # "RF" (1.000) instead of random forest.
        nlp.add_pipe("abbreviation_detector")

        nlp.add_pipe("scispacy_linker", config={
            "linker_name": "umls",
            "resolve_abbreviations": True,
            "threshold": threshold,
            "no_definition_threshold": 0.95,
            "filter_for_definitions": True,
            "max_entities_per_mention": 1,
            "k": 30,
        })
        log.info("SciSpacy + UMLS linker loaded")

        _nlp = nlp
        return _nlp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_umls_concepts(
    text: str,
    *,
    min_confidence: float = 0.7,
    spacy_model: str = "en_core_sci_lg",
    cache_dir: str | None = None,
    max_concepts: int = 30,
) -> list[UmlsConcept]:
    """Extract UMLS concepts from biomedical text.

    Parameters
    ----------
    text:
        Input text (abstract preferred, body_text as fallback).
    min_confidence:
        Minimum linker similarity score to keep a concept.
    spacy_model:
        SciSpacy model name (must be installed).
    cache_dir:
        Directory for UMLS KB cache. If None, uses ~/.scispacy/.
    max_concepts:
        Maximum number of concepts to return (sorted by confidence desc).

    Returns
    -------
    List of UmlsConcept, deduplicated by CUI, filtered by relevant
    semantic types and min_confidence, sorted by confidence descending.
    """
    if not text or not text.strip():
        return []

    nlp = _get_nlp(spacy_model, cache_dir, min_confidence)
    linker = nlp.get_pipe("scispacy_linker")

    doc = nlp(text)

    # Collect concepts, dedup by CUI (keep highest confidence)
    seen: dict[str, UmlsConcept] = {}

    # [umls-probe] 临时诊断 — 确认后整块删除（本文件两处 + vault.py + wizard.py）
    from collections import Counter as _Counter

    _p_drop = _Counter()          # 每个过滤器丢了多少 mention
    _p_freq = _Counter()          # 存活概念被提及几次
    _p_surf: dict[str, set] = {}  # 它们的原文写法
    _p_head = text[:200].lower()  # 标题大致落在这里

    for ent in doc.ents:
        if not ent._.kb_ents:
            _p_drop["linker 没给出候选"] += 1
            continue

        # Take the top match only (max_entities_per_mention=1)
        cui, score = ent._.kb_ents[0]

        if score < min_confidence:
            _p_drop[f"分数 < {min_confidence}"] += 1
            continue

        # Look up full entity from KB
        kb_entry = linker.kb.cui_to_entity.get(cui)
        if kb_entry is None:
            _p_drop["CUI 不在 KB 里"] += 1
            continue

        # Filter by relevant semantic types
        tuis = kb_entry.types  # list of TUI strings
        relevant_tui = None
        for t in tuis:
            if is_relevant(t):
                relevant_tui = t
                break

        if relevant_tui is None:
            _p_drop["语义类型不相关"] += 1
            continue

        _p_freq[cui] += 1
        _p_surf.setdefault(cui, set()).add(ent.text)

        # Dedup: keep highest confidence per CUI
        if cui in seen and seen[cui].confidence >= score:
            continue

        seen[cui] = UmlsConcept(
            cui=cui,
            name=kb_entry.canonical_name,
            tui=relevant_tui,
            semantic_type=get_type_name(relevant_tui) or "",
            confidence=round(score, 4),
        )

    # Sort by confidence descending, truncate
    results = sorted(seen.values(), key=lambda c: c.confidence, reverse=True)

    # [umls-probe] 漏斗 + 截断线。截断是整条链上唯一看不见的损失：被
    # max_concepts 切掉的概念和"从没被抽出来"在下游长得一模一样。
    print(f"\n[umls] 输入 {len(text):,} 字符 -> NER {len(doc.ents)} 个实体 "
          f"-> {len(seen)} 个概念通过全部过滤器 -> 返回 {min(len(results), max_concepts)} "
          f"(cap {max_concepts})", flush=True)
    for _why, _n in _p_drop.most_common():
        print(f"[umls]   丢弃 {_n:5d}  {_why}", flush=True)
    print(f"[umls] 保留的（排序键=linker 字符串匹配分, T=出现在开头200字符内）:",
          flush=True)
    for _i, _c in enumerate(results[:max_concepts], 1):
        _t = "T" if any(m.lower() in _p_head for m in _p_surf.get(_c.cui, ())) else " "
        print(f"[umls]   {_i:3d} {_c.confidence:.4f} x{_p_freq[_c.cui]:<3d} {_t} "
              f"{_c.name[:38]:40s}{sorted(_p_surf.get(_c.cui, []))[:3]}", flush=True)
    if len(results) > max_concepts:
        _cut = sorted(results[max_concepts:], key=lambda c: -_p_freq[c.cui])[:10]
        print(f"[umls] 被 cap 切掉 {len(results) - max_concepts} 个，"
              f"其中提及次数最多的 10 个:", flush=True)
        for _c in _cut:
            _r = results.index(_c) + 1
            print(f"[umls]   x{_p_freq[_c.cui]:<3d} {_c.confidence:.4f} "
                  f"(分数排名 {_r:3d})  {_c.name[:44]}", flush=True)

    return results[:max_concepts]


# ---------------------------------------------------------------------------
# Ad-hoc check: re-extract a profile's seeds and diff against what is stored
# ---------------------------------------------------------------------------
#
#     python -m rag_lib.umls.extractor [profile-slug]
#
# Answers "would this text give the same concepts again, and would they
# still land on the same OpenAlex topics". Runs the real pipeline with the
# deployment's own settings, so the first call pays the model load; per
# paper after that is a fraction of a second.
#
# Both layers are checked, because only the second one is visible to the
# user. Concepts live on the paper row; what step 3 offers to switch on
# are the topics they map to, and the merged set that reached the profile.
#
# The imports live in here rather than at module scope on purpose: this
# package sits below the API and the database layer, and nothing about
# extraction should depend on either.


def _main() -> int:
    import json
    import sys
    import time

    from ..api.settings import get_settings
    from ..db import connect
    from ..paper import Paper
    from ..profile import Profile
    from .topic_mapper import map_concepts_to_topics, merge_umls_topics

    slug = sys.argv[1] if len(sys.argv) > 1 else "type-1-diabetes"
    settings = get_settings()
    conn = connect(settings.RADAR_DB_PATH)

    prow = conn.execute(
        "SELECT id FROM profiles WHERE slug = ?", (slug,),
    ).fetchone()
    if prow is None:
        print(f"no profile {slug!r}")
        return 1

    # topics_json comes along because the profile's OpenAlex topic set is
    # rebuilt here too. Only source columns are read: the paper's text and
    # the topics OpenAlex assigned it. The two ``umls_*`` columns are read
    # as the comparison target and are never printed.
    rows = conn.execute(
        """
        SELECT p.openalex_id, p.title, p.abstract, p.body_text, p.topics_json,
               p.umls_concepts_json, p.umls_mapped_topics_json
          FROM profile_seeds ps
          JOIN papers p   ON p.openalex_id = ps.openalex_id
         WHERE ps.profile_id = ?
         ORDER BY p.title
        """,
        (prow["id"],),
    ).fetchall()
    if not rows:
        print(f"no seeds for profile {slug!r}")
        return 1

    print(f"profile      : {slug}  ({len(rows)} seeds)")
    print(f"spacy model  : {settings.RADAR_UMLS_SPACY_MODEL}")
    print(f"min conf     : {settings.RADAR_UMLS_MIN_CONFIDENCE}")
    print(f"max concepts : {settings.RADAR_UMLS_MAX_CONCEPTS}")
    print(f"embed model  : {settings.RADAR_UMLS_EMBEDDING_MODEL}")
    print(f"min topic sim: {settings.RADAR_UMLS_MIN_TOPIC_SIMILARITY}")
    print(f"max additions: {settings.RADAR_UMLS_MAX_TOPIC_ADDITIONS}")
    print(f"cache dir    : {settings.RADAR_UMLS_CACHE_DIR}\n")

    identical = 0
    topics_identical = 0
    per_paper: list[list] = []
    seed_papers: list[Paper] = []
    for row in rows:
        # Same text the upload path picks: the title leads, because an
        # abstract that only ever writes "T2DM" gives the linker nothing
        # to expand.
        head = (row["title"] or "").strip()
        body = (row["abstract"] or "").strip() or (row["body_text"] or "")[:20_000]
        text = f"{head}\n\n{body}" if head and body else (head or body)

        t0 = time.monotonic()
        fresh = extract_umls_concepts(
            text,
            min_confidence=settings.RADAR_UMLS_MIN_CONFIDENCE,
            spacy_model=settings.RADAR_UMLS_SPACY_MODEL,
            cache_dir=str(settings.RADAR_UMLS_CACHE_DIR),
            max_concepts=settings.RADAR_UMLS_MAX_CONCEPTS,
        )
        elapsed = time.monotonic() - t0

        stored = {
            c["cui"]: c for c in json.loads(row["umls_concepts_json"] or "[]")
        }
        now = {c.cui: c for c in fresh}

        print("=" * 78)
        print((row["title"] or "")[:76])
        print(f"  {len(text):,} chars in  ->  {len(fresh)} concepts, "
              f"{len(stored)} stored   ({elapsed:.1f}s)")

        gone = sorted(set(stored) - set(now))
        new = sorted(set(now) - set(stored))
        moved = sorted(
            cui for cui in set(stored) & set(now)
            if abs(stored[cui]["confidence"] - now[cui].confidence) > 1e-4
        )

        if not stored:
            print("  nothing stored to compare against")
        elif not (gone or new or moved):
            identical += 1
            print("  identical to the stored extraction")
        else:
            for cui in gone:
                print(f"  - {cui}  {stored[cui]['name'][:50]}")
            for cui in new:
                print(f"  + {cui}  {now[cui].name[:50]}")
            for cui in moved:
                print(f"  ~ {cui}  {stored[cui]['confidence']:.4f} -> "
                      f"{now[cui].confidence:.4f}  {now[cui].name[:40]}")

        for c in fresh:
            print(f"      {c.confidence:.4f}  {c.cui}  {c.name[:42]:44s}"
                  f"[{c.semantic_type}]")

        # The layer the user actually sees: concept names embedded against
        # the OpenAlex topic index. Same call the upload path makes.
        t1 = time.monotonic()
        mapped = map_concepts_to_topics(
            fresh,
            min_similarity=settings.RADAR_UMLS_MIN_TOPIC_SIMILARITY,
            embedding_model=settings.RADAR_UMLS_EMBEDDING_MODEL,
            cache_dir=str(settings.RADAR_UMLS_CACHE_DIR),
        )
        map_elapsed = time.monotonic() - t1

        stored_t = {
            m["topic_id"]: m
            for m in json.loads(row["umls_mapped_topics_json"] or "[]")
        }
        now_t = {m.topic_id: m for m in mapped}

        print(f"  -> {len(mapped)} mapped topics, {len(stored_t)} stored"
              f"   ({map_elapsed:.1f}s)")

        t_gone = sorted(set(stored_t) - set(now_t))
        t_new = sorted(set(now_t) - set(stored_t))
        t_moved = sorted(
            tid for tid in set(stored_t) & set(now_t)
            if abs(stored_t[tid]["similarity"] - now_t[tid].similarity) > 1e-4
        )
        if not stored_t:
            print("     nothing stored to compare against")
        elif not (t_gone or t_new or t_moved):
            topics_identical += 1
            print("     identical to the stored mapping")
        else:
            for tid in t_gone:
                print(f"     - {stored_t[tid]['display_name'][:50]}")
            for tid in t_new:
                print(f"     + {now_t[tid].display_name[:50]}")
            for tid in t_moved:
                print(f"     ~ {stored_t[tid]['similarity']:.4f} -> "
                      f"{now_t[tid].similarity:.4f}  "
                      f"{now_t[tid].display_name[:40]}")

        for m in mapped:
            print(f"        {m.similarity:.4f}  {m.display_name[:44]:46s}"
                  f"<- {m.source_name[:26]}")

        per_paper.append(mapped)
        topics = json.loads(row["topics_json"] or "{}")
        seed_papers.append(Paper.from_dict({
            "openalex_id": row["openalex_id"],
            "title": row["title"],
            "primary_topic": topics.get("primary_topic"),
            "topics": topics.get("topics") or [],
        }))

    print("=" * 78)
    print(f"{identical}/{len(rows)} seeds reproduced their stored concepts exactly")
    print(f"{topics_identical}/{len(rows)} seeds reproduced their stored topic mapping")

    # What step 3 offers, recomputed rather than read back. The OpenAlex
    # base set is rebuilt from the seeds' own topic assignments, then the
    # freshly-mapped UMLS topics are merged into it exactly as
    # ``aggregate_draft_topics`` does. Nothing here comes out of
    # ``profiles.topic_filters_json``, so the list below can be compared
    # against the frontend as an independent second opinion — if the two
    # agree, the stored set really is what this pipeline produces.
    #
    # One difference from the stored version by construction: on/off state
    # is a user choice held on the profile row, so everything here reads as
    # on. Ordering within the appended block follows the merge.
    base = Profile.aggregate_topic_filters(seed_papers)
    merged = merge_umls_topics(
        base,
        per_paper,
        max_additions=settings.RADAR_UMLS_MAX_TOPIC_ADDITIONS,
        cache_dir=str(settings.RADAR_UMLS_CACHE_DIR),
    )

    entries = merged.get("topics") or []
    openalex_entries = [t for t in entries if t.get("source") != "umls"]
    umls_entries = [t for t in entries if t.get("source") == "umls"]

    print(f"\nrecomputed topic set: {len(openalex_entries)} from OpenAlex "
          f"+ {len(umls_entries)} from UMLS")
    for t in openalex_entries:
        print(f"  openalex  {t.get('display_name', '')[:56]:58s}"
              f"n_papers={t.get('count', 0)}")
    if not umls_entries:
        print("  (UMLS added nothing — every mapped topic was already present)")
    for t in umls_entries:
        print(f"  umls      {t.get('display_name', '')[:56]:58s}"
              f"n_papers={t.get('count', 0)}")

    conn.close()
    return 0


# ---------------------------------------------------------------------------
# Ad-hoc check: one paper, no database
# ---------------------------------------------------------------------------
#
#     python -m rag_lib.umls.extractor paper --pdf /path/to/paper.pdf
#     python -m rag_lib.umls.extractor paper --text /path/to/paper.txt
#     python -m rag_lib.umls.extractor paper --title "..." --abstract "..."
#
# Answers "what should this paper yield, by the algorithm alone". Nothing
# is read from radar.db — the paper comes from a file or the command line,
# the parameters come from settings or flags, and the concept list is
# computed here and now. That is what makes it usable as an independent
# second opinion on whatever the deployment stored: if the two disagree,
# one of them is wrong, and this side can be re-read line by line.
#
# Two passes run over the same document, on purpose. The first calls the
# real ``extract_umls_concepts`` — its output is the answer, and nothing
# below is allowed to substitute for it. The second walks ``doc.ents``
# again to narrate why each mention lived or died, and then checks its own
# narration against the real result. A mismatch means the explanation has
# drifted from the implementation, which is worth knowing before trusting
# either one; it is reported rather than swallowed.
#
# The narration exists because the funnel is otherwise invisible. A concept
# that never appears in the output looks identical whether the NER missed
# it, the linker put it below threshold, the semantic-type filter dropped
# it, or ``max_concepts`` truncated it away — and those four call for four
# different fixes.


def _main_paper(argv: list[str]) -> int:
    import json
    import time
    from collections import Counter
    from pathlib import Path

    def _flag(name: str, default: str | None = None) -> str | None:
        if name in argv:
            i = argv.index(name)
            if i + 1 < len(argv):
                return argv[i + 1]
        return default

    pdf, txt, db = _flag("--pdf"), _flag("--text"), _flag("--db")
    title = _flag("--title") or ""
    abstract = _flag("--abstract") or ""
    body = ""
    topics_json = ""

    if db:
        # Reading the *inputs* from the papers row, never the outputs. The
        # four columns below are what PDF ingestion and OpenAlex enrichment
        # produced; they are the same thing the upload path would hand the
        # extractor. The two ``umls_*`` columns and anything on ``profiles``
        # are deliberately not in this SELECT — those are what this check
        # exists to recompute, and reading them back would make it a
        # tautology. The query is spelled out here so that claim can be
        # checked by eye.
        import sqlite3

        from ..api.settings import get_settings as _gs

        conn = sqlite3.connect(_gs().RADAR_DB_PATH)
        conn.row_factory = sqlite3.Row
        col = "openalex_id" if db.startswith("http") else "title"
        op = "=" if db.startswith("http") else "LIKE"
        rows = conn.execute(
            f"SELECT openalex_id, title, abstract, body_text, topics_json "  # noqa: S608
            f"FROM papers WHERE {col} {op} ? ORDER BY first_seen_at",
            (db,),
        ).fetchall()
        conn.close()
        if not rows:
            print(f"no paper matching {db!r} "
                  f"(titles take a LIKE pattern, e.g. 'Biofluid%')")
            return 1
        if len(rows) > 1:
            print(f"{len(rows)} papers match {db!r}:")
            for r in rows:
                print(f"  {(r['title'] or '')[:76]}")
            return 1
        row = rows[0]
        title = title or (row["title"] or "")
        abstract = abstract or (row["abstract"] or "")
        body = row["body_text"] or ""
        topics_json = row["topics_json"] or ""
        print(f"source       : papers row {row['openalex_id']}")
        print(f"               columns read: title, abstract, body_text, "
              f"topics_json  (no umls_* columns, no profiles)")
    elif pdf:
        # rag_lib.vault is stdlib + pdfplumber; it does not touch the DB.
        # Note what it does NOT give us: an abstract. On the upload path
        # that field is filled from OpenAlex enrichment, not from the PDF,
        # so a paper handed over as a bare file falls through to the body
        # branch below unless --abstract supplies one. That is a real
        # difference from production and is printed as such.
        from ..vault import ingest_pdf

        rec = ingest_pdf(pdf)
        title = title or rec.title
        body = rec.body_text
        print(f"source       : {pdf}")
        print(f"               {rec.n_pages} pages, {len(body):,} chars, doi={rec.doi}")
    elif txt:
        raw = Path(txt).read_text(encoding="utf-8")
        # "Title\n\nrest" when the first block is short enough to be one;
        # anything else is treated as a single lump of body text.
        head, sep, rest = raw.partition("\n\n")
        if sep and len(head) < 400:
            title = title or head.strip()
            abstract = abstract or rest.strip()
        else:
            body = raw
        print(f"source       : {txt}  ({len(raw):,} chars)")
    elif not (title or abstract):
        print("usage: python -m rag_lib.umls.extractor paper\n"
              "         --db 'Title%' | --db <openalex_id>   "
              "(reads only title/abstract/body_text/topics_json)\n"
              "         --pdf FILE | --text FILE | --title T --abstract A\n"
              "       optional: --min-confidence F  --max-concepts N  "
              "--max-additions N\n"
              "                 --min-topic-similarity F  --spacy-model M  "
              "--cache-dir D")
        return 2
    else:
        print("source       : command line")

    # Settings supply the deployment's own parameters so the answer is the
    # one this install would produce, not a set of library defaults. This
    # reads .env, not the database. Flags win when given, which is what
    # makes the thing usable for sweeping a threshold.
    try:
        from ..api.settings import get_settings

        s = get_settings()
        model = _flag("--spacy-model") or s.RADAR_UMLS_SPACY_MODEL
        cache = _flag("--cache-dir") or str(s.RADAR_UMLS_CACHE_DIR)
        min_conf = float(_flag("--min-confidence") or s.RADAR_UMLS_MIN_CONFIDENCE)
        max_n = int(_flag("--max-concepts") or s.RADAR_UMLS_MAX_CONCEPTS)
        src = "settings"
    except Exception as exc:  # noqa: BLE001 — a missing .env must not stop the check
        model = _flag("--spacy-model") or "en_core_sci_lg"
        cache = _flag("--cache-dir")
        min_conf = float(_flag("--min-confidence") or 0.7)
        max_n = int(_flag("--max-concepts") or 30)
        src = f"defaults ({type(exc).__name__})"

    # Same assembly the upload path uses. Imported rather than copied: a
    # local reimplementation would drift, and the whole value of this
    # check is that it runs what production runs. The fallback keeps the
    # check usable in an environment where the API package will not import.
    try:
        from ..api.services.vault import _umls_input_text

        text, rule = _umls_input_text(title, abstract, body), "vault._umls_input_text"
    except Exception as exc:  # noqa: BLE001
        head = (title or "").strip()
        tail = (abstract or "").strip() or (body or "")[:20_000]
        text = f"{head}\n\n{tail}" if head and tail else (head or tail)
        rule = f"local copy — could not import the real one ({type(exc).__name__})"

    print(f"title        : {title[:66] or '(none)'}")
    print(f"abstract     : {len(abstract):,} chars"
          f"{'' if abstract else '  <- absent; falling back to body text'}")
    print(f"text rule    : {rule}")
    print(f"params from  : {src}")
    print(f"  spacy model    {model}")
    print(f"  min confidence {min_conf}")
    print(f"  max concepts   {max_n}")
    print(f"  cache dir      {cache}")
    print(f"input        : {len(text):,} chars\n")
    if not text.strip():
        print("nothing to extract")
        return 1

    # Pass 1 — the real thing. This is the answer.
    t0 = time.monotonic()
    result = extract_umls_concepts(
        text,
        min_confidence=min_conf,
        spacy_model=model,
        cache_dir=cache,
        max_concepts=max_n,
    )
    elapsed = time.monotonic() - t0

    # Pass 2 — narrate the funnel over the same document.
    nlp = _get_nlp(model, cache, min_conf)
    linker = nlp.get_pipe("scispacy_linker")
    doc = nlp(text)

    title_low = (title or "").lower()
    freq: Counter[str] = Counter()
    surface: dict[str, set] = {}
    in_title: dict[str, bool] = {}
    kept: dict[str, tuple] = {}
    dropped: list[tuple] = []

    for ent in doc.ents:
        if not ent._.kb_ents:
            dropped.append((ent.text, "no KB candidate cleared the linker", "", 0.0))
            continue
        cui, score = ent._.kb_ents[0]
        if score < min_conf:
            dropped.append((ent.text, f"score {score:.3f} < {min_conf}", cui, score))
            continue
        kb = linker.kb.cui_to_entity.get(cui)
        if kb is None:
            dropped.append((ent.text, "CUI absent from the KB", cui, score))
            continue
        rel = next((t for t in kb.types if is_relevant(t)), None)
        if rel is None:
            types = "/".join(f"{t}={get_type_name(t) or '?'}" for t in kb.types)
            dropped.append((ent.text, f"no relevant semantic type [{types}]", cui, score))
            continue
        freq[cui] += 1
        surface.setdefault(cui, set()).add(ent.text)
        in_title[cui] = in_title.get(cui, False) or (ent.text.lower() in title_low)
        if cui not in kept or kept[cui][1] < score:
            kept[cui] = (kb.canonical_name, score, rel)

    print("=" * 96)
    print(f"NER found {len(doc.ents)} entities  ->  {len(kept)} distinct concepts "
          f"survive every filter  ->  {len(result)} returned (cap {max_n})")
    print(f"extraction took {elapsed:.1f}s")

    # The full survivor list in the order the algorithm ranks it, with the
    # cut drawn in. Everything below the line is computed, filtered and
    # then discarded — that is the part no downstream stage can recover.
    print("\n" + "-" * 96)
    print("survivors, ranked as the algorithm ranks them (by linker score)")
    print("-" * 96)
    ranked = sorted(kept.items(), key=lambda kv: kv[1][1], reverse=True)
    for i, (cui, (name, score, tui)) in enumerate(ranked, 1):
        if i == max_n + 1:
            print(f"  {'':4s} {'-' * 88}")
            print(f"  {'':4s} --- max_concepts = {max_n} cuts here; "
                  f"{len(ranked) - max_n} concepts below are discarded ---")
            print(f"  {'':4s} {'-' * 88}")
        mark = "T" if in_title.get(cui) else " "
        print(f"  {i:4d} {score:.4f} x{freq[cui]:<3d} {mark} {cui}  {name[:40]:42s}"
              f"[{get_type_name(tui) or tui}]")
        print(f"       {'':22s}mentions={sorted(surface.get(cui, []))[:4]}")

    # Same survivors by how often the paper actually says them. Not what
    # the algorithm does today — it is here because the two orders differ
    # enough to be the whole story on some papers, and seeing them side by
    # side is the cheapest way to tell whether the ranking key is the
    # problem or the filters are.
    print("\n" + "-" * 96)
    print("the same survivors, ranked by mention count (diagnostic — not the algorithm)")
    print("-" * 96)
    by_score = [c for c, _ in ranked]
    for i, (cui, n) in enumerate(freq.most_common(min(15, len(freq))), 1):
        name, score, tui = kept[cui]
        pos = by_score.index(cui) + 1
        fate = "kept" if pos <= max_n else f"CUT (score-rank {pos})"
        mark = "T" if in_title.get(cui) else " "
        print(f"  {i:4d} x{n:<3d} {score:.4f} {mark} {name[:44]:46s}{fate}")

    if dropped:
        print("\n" + "-" * 96)
        print(f"mentions the filters removed ({len(dropped)}) — first 25")
        print("-" * 96)
        for surf, why, cui, score in dropped[:25]:
            print(f"  {surf[:26]:28s} {score:.3f} {cui:10s} {why[:52]}")

    # Does the narration agree with the function it is narrating?
    explained = {c for c, _ in ranked[:max_n]}
    actual = {c.cui for c in result}
    print("\n" + "=" * 96)
    if explained == actual:
        print(f"OK — the walkthrough reproduces extract_umls_concepts() exactly "
              f"({len(actual)} CUIs)")
    else:
        print("MISMATCH — the walkthrough above disagrees with "
              "extract_umls_concepts(); trust the function, not the narration")
        for c in sorted(explained - actual):
            print(f"   only in walkthrough: {c}  {kept[c][0]}")
        for c in sorted(actual - explained):
            print(f"   only in the function: {c}")

    print("\nstage 1 result — the concepts this paper yields:")
    for c in result:
        print(f"  {c.confidence:.4f}  {c.cui}  {c.name[:46]:48s}[{c.semantic_type}]")

    # -----------------------------------------------------------------
    # Stage 2 — concepts to OpenAlex topics
    # -----------------------------------------------------------------
    #
    # Concept names are embedded and matched against the precomputed topic
    # index. Nothing is capped here, so the count below is every topic that
    # cleared min_similarity for any concept — which is worth seeing raw,
    # because the number tends to be far larger than the handful the wizard
    # ends up offering.
    from .topic_mapper import map_concepts_to_topics, merge_umls_topics

    try:
        emb = _flag("--embedding-model") or s.RADAR_UMLS_EMBEDDING_MODEL
        min_sim = float(_flag("--min-topic-similarity")
                        or s.RADAR_UMLS_MIN_TOPIC_SIMILARITY)
        max_add = int(_flag("--max-additions") or s.RADAR_UMLS_MAX_TOPIC_ADDITIONS)
    except NameError:  # settings never loaded
        emb = _flag("--embedding-model") or "mxbai-embed-large"
        min_sim = float(_flag("--min-topic-similarity") or 0.40)
        max_add = int(_flag("--max-additions") or 10)

    print("\n" + "=" * 96)
    print(f"stage 2 — mapping {len(result)} concepts to OpenAlex topics "
          f"(embedder {emb}, min similarity {min_sim})")
    print("=" * 96)
    t1 = time.monotonic()
    try:
        mapped = map_concepts_to_topics(
            result, min_similarity=min_sim, embedding_model=emb, cache_dir=cache,
        )
    except Exception as exc:  # noqa: BLE001 — ollama down, index missing, ...
        print(f"  could not map: {type(exc).__name__}: {str(exc)[:200]}")
        print("  (the embedder runs through ollama — is it up?)")
        return 1
    print(f"  {len(mapped)} distinct topics matched  ({time.monotonic() - t1:.1f}s)")
    if mapped:
        sims = [m.similarity for m in mapped]
        print(f"  similarity range {min(sims):.3f} - {max(sims):.3f}"
              f"   (min_similarity={min_sim} removed "
              f"{'nothing' if min(sims) > min_sim else 'some'})")
    for m in mapped:
        print(f"  {m.similarity:.4f}  {m.display_name[:46]:48s}<- {m.source_name[:34]}")

    # -----------------------------------------------------------------
    # Stage 3 — what the wizard would offer
    # -----------------------------------------------------------------
    #
    # The base is this paper's own OpenAlex topic assignment, aggregated
    # exactly as Profile.aggregate_topic_filters does for a seed corpus of
    # one. That matters: merge_umls_topics drops candidates that duplicate
    # something already in the base, so without it the accepted list would
    # be wrong in the optimistic direction.
    from ..paper import Paper
    from ..profile import Profile

    tj = json.loads(topics_json or "{}")
    seed = Paper.from_dict({
        "openalex_id": "local", "title": title,
        "primary_topic": tj.get("primary_topic"),
        "topics": tj.get("topics") or [],
    })
    base = Profile.aggregate_topic_filters([seed])
    existing = [(t["id"], t.get("display_name") or "")
                for t in base.get("topics", []) if t.get("id")]

    print("\n" + "=" * 96)
    print(f"stage 3 — merging into the topic list (max_additions={max_add})")
    print("=" * 96)
    if not topics_json:
        print("  NOTE: no topics_json for this paper, so the OpenAlex base is")
        print("        empty and nothing can be rejected as a duplicate. On a")
        print("        real profile the base is non-empty and this list shrinks.")
    print(f"  base from OpenAlex ({len(existing)}):")
    for _, name in existing:
        print(f"    {name[:66]}")

    merged = merge_umls_topics(
        base, [mapped], max_additions=max_add, cache_dir=cache,
    )
    offered = [t for t in (merged.get("topics") or []) if t.get("source") == "umls"]

    # Narrate the same decisions, then check the narration against the real
    # merge — same contract as the stage-1 walkthrough above.
    from .cache import get_topic_index
    from .topic_mapper import _embedding_similarity, _is_near_duplicate

    try:
        index = get_topic_index(cache)
    except Exception:  # noqa: BLE001
        index = None

    print(f"\n  every candidate, ranked as merge_umls_topics ranks them:")
    accepted: list[str] = []
    acc_names: set = set()
    seen_ids = {i for i, _ in existing}
    seen_names = {n.lower() for _, n in existing if n}
    cands = [m for m in mapped
             if m.topic_id not in seen_ids and m.display_name.lower() not in seen_names]
    for m in sorted(cands, key=lambda x: x.similarity, reverse=True):
        dup = next(((en, _embedding_similarity(index, m.topic_id, eid))
                    for eid, en in existing
                    if _is_near_duplicate(index, m.topic_id, m.display_name, eid, en)),
                   None)
        if dup:
            why = f"DROP  near-duplicate of '{dup[0][:30]}'" + (
                f" cos={dup[1]:.3f}" if dup[1] is not None else "")
        elif m.display_name.lower() in acc_names:
            why = "DROP  name already accepted"
        elif len(accepted) >= max_add:
            why = "cut   (max_additions reached)"
        else:
            accepted.append(m.topic_id)
            acc_names.add(m.display_name.lower())
            why = "OFFER"
        print(f"    {m.similarity:.4f}  {m.display_name[:44]:46s}{why}")

    if {t["id"] for t in offered} != set(accepted):
        print("\n  MISMATCH — this walkthrough disagrees with merge_umls_topics();"
              "\n  trust the function. Its answer is the list below.")

    print("\n" + "=" * 96)
    print(f"FINAL — what step 3 would offer for this paper: "
          f"{len(existing)} OpenAlex topics + {len(offered)} from UMLS")
    print("=" * 96)
    for _, name in existing:
        print(f"  openalex  {name[:60]}")
    for t in offered:
        print(f"  umls      {t.get('display_name', '')[:60]}")
    if not offered:
        print("  (UMLS added nothing)")
    return 0


if __name__ == "__main__":
    import sys as _sys

    _argv = _sys.argv[1:]
    if _argv and _argv[0] == "paper":
        raise SystemExit(_main_paper(_argv[1:]))
    raise SystemExit(_main())
