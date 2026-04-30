"""Profile — the target a selector scores candidates against.

A Profile owns:
  - a seed corpus of Papers (with embeddings + OpenAlex topic assignments),
  - aggregated topic_filters at each level of the OpenAlex hierarchy,
  - the embedding model key to use at select time,
  - selector and gatherer configs (opaque dicts — the respective classes
    own their shape),
  - a scoring threshold and bookkeeping dates.

The Profile JSON is the portable artifact. Once a Profile has been built
(PDFs or CSV manifest -> OpenAlex lookup -> embeddings), the JSON stands
on its own: the Gatherer can run against it, the Selector can fit and
score against it, and none of those steps need access to the original
PDF bytes. The PDF paths are kept on each Paper (``Paper.local_path``)
so downstream RAG chat can reopen them when it wants deeper context —
but the JSON does not carry binary data.

Build entry points are classmethods on this class (``from_csv``,
``from_pdfs``) plus the deserialization pair (``from_dict``,
``from_json``). The separate ``rag_lib.profile_builder`` module has been
folded in here so profile construction lives alongside profile state.
"""

from __future__ import annotations

import csv
import datetime
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from .embedders import placeholder_embed
from .paper import Paper

 
from .openalex_client import OpenAlexClient


Embedder = Callable[[str], list[float]]


@dataclass
class Profile:
    name: str
    papers: list[Paper] = field(default_factory=list)
    topic_filters: dict = field(default_factory=dict)
    embedding_model: str = "placeholder-v1"
    selector_config: dict = field(default_factory=dict)
    gatherer_config: dict = field(default_factory=dict)
    threshold: float | None = None
    created: str | None = None
    last_radar_date: str | None = None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "papers": [p.to_dict() for p in self.papers],
            "topic_filters": dict(self.topic_filters),
            "embedding_model": self.embedding_model,
            "selector_config": dict(self.selector_config),
            "gatherer_config": dict(self.gatherer_config),
            "threshold": self.threshold,
            "created": self.created,
            "last_radar_date": self.last_radar_date,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Profile":
        return cls(
            name=d["name"],
            papers=[Paper.from_dict(p) for p in (d.get("papers") or [])],
            topic_filters=dict(d.get("topic_filters") or {}),
            embedding_model=d.get("embedding_model") or "placeholder-v1",
            selector_config=dict(d.get("selector_config") or {}),
            gatherer_config=dict(d.get("gatherer_config") or {}),
            threshold=d.get("threshold"),
            created=d.get("created"),
            last_radar_date=d.get("last_radar_date"),
        )

    def to_json(self, path: Path | str, *, indent: int = 2) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=indent))

    @classmethod
    def from_json(cls, path: Path | str) -> "Profile":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ------------------------------------------------------------------
    # Seed embedding access
    # ------------------------------------------------------------------

    def seed_embeddings(self, model: str | None = None) -> list[list[float]]:
        """Return the seed papers' embeddings for the given model. Skips
        papers that don't carry that embedding key. Raises if no paper in
        the profile has one.
        """
        key = model or self.embedding_model
        vecs = [p.embedding_for(key) for p in self.papers]
        vecs = [v for v in vecs if v is not None]
        if not vecs:
            raise ValueError(
                f"Profile '{self.name}' has no papers with embedding '{key}'."
            )
        return vecs

    # ------------------------------------------------------------------
    # Build from inputs
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        csv_path: Path | str,
        name: str,
        openalex_client: "OpenAlexClient",
        *,
        embedder: Embedder | None = None,
        embedding_model: str = "placeholder-v1",
        embed_text_fn: Callable[[Paper], str] | None = None,
    ) -> "Profile":
        """Build a Profile from a CSV manifest.

        CSV columns (all optional but at least one resolution path
        required for a given row):
            doi          DOI of the paper. If present, tried first.
            title        Paper title. Used for OpenAlex search fallback.
            year         Publication year. Narrows the title-search filter.
            path         Path to the local PDF. Stored on Paper.local_path.
            abstract     Abstract text. Used only if OpenAlex lookup fails.

        Papers that fail OpenAlex resolution are still included as Paper
        records with ``source='user_csv'`` and whatever fields the CSV
        provided — so a profile with a few paywalled / not-indexed seeds
        still builds.

        ``embedder`` defaults to the hash-based placeholder; pass
        ``specter2_embed`` (Phase 1B) for the real one.
        ``embed_text_fn`` controls what text is fed to the embedder;
        defaults to ``TITLE / ABSTRACT`` concatenation.
        """
        csv_path = Path(csv_path)
        if embedder is None:
            embedder = placeholder_embed
        if embed_text_fn is None:
            embed_text_fn = _default_embed_text

        papers: list[Paper] = []
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                paper = _paper_from_csv_row(row, openalex_client)
                text = embed_text_fn(paper)
                paper.embeddings[embedding_model] = embedder(text)
                papers.append(paper)

        return cls(
            name=name,
            papers=papers,
            topic_filters=cls.aggregate_topic_filters(papers),
            embedding_model=embedding_model,
            created=_utcnow_iso(),
        )

    @classmethod
    def from_pdfs(
        cls,
        pdf_dir: Path | str,
        name: str,
        openalex_client: "OpenAlexClient",
        *,
        embedder: Embedder | None = None,
        embedding_model: str = "allenai/specter2_base",
        embed_text_fn: Callable[[Paper], str] | None = None,
        glob: str = "*.pdf",
    ) -> "Profile":
        """Ingest a directory of PDFs into a Profile.

        For each PDF: extract title + body_text + best-effort DOI via
        ``rag_lib.vault.ingest_pdf``; resolve OpenAlex (DOI first, title
        fallback); build a Paper with ``local_path``, ``body_text``,
        and OpenAlex-enriched fields; embed via the chosen embedder.
        Aggregates topic_filters across the seed at the end, same as
        ``from_csv``.

        Requires pdfplumber (``phase1b`` extras). The SPECTER2 path is
        selected by passing ``embedder=specter2_embed,
        embedding_model='specter2'``.
        """
        from .vault import ingest_pdf  # lazy import: pdfplumber is phase1b
        pdf_dir = Path(pdf_dir)
        if embedder is None:
            embedder = placeholder_embed
        if embed_text_fn is None:
            embed_text_fn = _default_embed_text

        papers: list[Paper] = []
        for pdf_path in sorted(pdf_dir.glob(glob)):
            try:
                rec = ingest_pdf(pdf_path)
            except ImportError:
                # phase1b not installed — surface the helpful message.
                raise
            except Exception as e:
                # Corrupt PDF (or HTML masquerading as one) shouldn't kill
                # the whole corpus build. Log and move on.
                print(f"warn: skipping {pdf_path.name}: {type(e).__name__}: {e}")
                continue
            paper = _paper_from_pdf_record(rec, openalex_client)
            text = embed_text_fn(paper)
            paper.embeddings[embedding_model] = embedder(text)
            papers.append(paper)

        return cls(
            name=name,
            papers=papers,
            topic_filters=cls.aggregate_topic_filters(papers),
            embedding_model=embedding_model,
            created=_utcnow_iso(),
        )

    # ------------------------------------------------------------------
    # Seed-corpus topic aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate_topic_filters(
        papers: list[Paper],
        *,
        top_k_each: int = 8,
    ) -> dict:
        """Count OpenAlex topic assignments across a seed corpus and
        return the top IDs at each level of the hierarchy.

        A paper contributes its ``primary_topic`` once and every entry in
        ``topics[]`` once. When a paper lacks topic data (e.g., a CSV
        fallback that never resolved on OpenAlex), it contributes nothing
        and is skipped silently.
        """
        topics: Counter = Counter()
        subfields: Counter = Counter()
        fields: Counter = Counter()
        domains: Counter = Counter()
        topic_names: dict[str, str] = {}
        subfield_names: dict[str, str] = {}
        field_names: dict[str, str] = {}
        domain_names: dict[str, str] = {}

        def _tally(t):
            if not t:
                return
            if t.id:
                topics[t.id] += 1
                topic_names.setdefault(t.id, t.display_name)
            if t.subfield and t.subfield.id:
                subfields[t.subfield.id] += 1
                subfield_names.setdefault(t.subfield.id, t.subfield.display_name)
            if t.field and t.field.id:
                fields[t.field.id] += 1
                field_names.setdefault(t.field.id, t.field.display_name)
            if t.domain and t.domain.id:
                domains[t.domain.id] += 1
                domain_names.setdefault(t.domain.id, t.domain.display_name)

        for p in papers:
            _tally(p.primary_topic)
            for t in p.topics:
                _tally(t)

        def _top(counter, names):
            return [
                {"id": tid, "display_name": names.get(tid, ""), "count": n}
                for tid, n in counter.most_common(top_k_each)
            ]

        return {
            "topics": _top(topics, topic_names),
            "subfields": _top(subfields, subfield_names),
            "fields": _top(fields, field_names),
            "domains": _top(domains, domain_names),
        }


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _paper_from_csv_row(row: dict, client: "OpenAlexClient") -> Paper:
    doi = _clean(row.get("doi"))
    path = _clean(row.get("path"))
    title = _clean(row.get("title"))
    abstract = _clean(row.get("abstract"))
    year = row.get("year")
    try:
        year_int = int(year) if year not in (None, "") else None
    except (TypeError, ValueError):
        year_int = None

    work = None
    if doi:
        work = client.lookup_by_doi(doi)
    if work is None and title:
        work = client.lookup_by_title(title, year_int)

    if work is not None:
        paper = client.paper_from_work(
            work,
            source="openalex",
            local_path=path,
            added=_utcnow_iso(),
        )
        # User-provided DOI / abstract win when OpenAlex is silent.
        if doi and not paper.doi:
            paper.doi = doi
        if abstract and not paper.abstract:
            paper.abstract = abstract
        return paper

    # Fallback: minimal Paper from CSV columns alone. OpenAlex couldn't
    # resolve it (paywalled, too obscure, not yet indexed). Still usable:
    # carries a local_path and whatever abstract the user pasted in, so
    # embedding + selector fit still work.
    return Paper(
        doi=doi,
        openalex_id=None,
        title=title or "",
        abstract=abstract or "",
        year=year_int,
        local_path=path,
        source="user_csv",
        added=_utcnow_iso(),
    )


def _default_embed_text(paper: Paper) -> str:
    """Default text for the embedder. Uses ``build_embedding_input``,
    which adds MeSH / keywords / substances / body as available and
    truncates by priority at the 512-token ceiling."""
    from .embed import build_embedding_input
    return build_embedding_input(paper)


def _paper_from_pdf_record(rec, client: "OpenAlexClient") -> Paper:
    """Given a PdfIngestRecord, resolve OpenAlex (DOI first, title
    fallback) and build a Paper. Falls back to a minimal Paper when
    OpenAlex is silent."""
    work = None
    if rec.doi:
        work = client.lookup_by_doi(rec.doi)
    if work is None and rec.title:
        work = client.lookup_by_title(rec.title)

    if work is not None:
        paper = client.paper_from_work(
            work,
            source="openalex",
            local_path=rec.path,
            added=_utcnow_iso(),
        )
        paper.body_text = rec.body_text
        if rec.doi and not paper.doi:
            paper.doi = rec.doi
        if not paper.title and rec.title:
            paper.title = rec.title
        return paper

    return Paper(
        doi=rec.doi,
        openalex_id=None,
        title=rec.title,
        abstract="",
        body_text=rec.body_text,
        local_path=rec.path,
        source="user_pdf",
        added=_utcnow_iso(),
    )


def _clean(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _utcnow_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
