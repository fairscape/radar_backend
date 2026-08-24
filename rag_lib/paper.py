"""Paper — first-class record for a single publication.

Serializable to JSON without the PDF bytes: `local_path` points at the PDF
on disk when we have it (so we can re-open for deeper inspection or chat
grounding), but the JSON artifact travels without binary data.

The topic hierarchy mirrors OpenAlex's four-level tree:
    Domain (4 buckets: Health Sciences, Life Sciences, Physical Sciences,
            Social Sciences)
        Field (~26: Medicine, Computer Science, Biochemistry, ...)
            Subfield (~250: Pediatrics, Software, AI, ...)
                Topic (~4,500: "Heart Rate Variability and Autonomic
                               Control", ...)
The Gatherer and profile-build stages use this structure to filter
candidates at whatever specificity a profile needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class TopicNode:
    """An OpenAlex Domain, Field, or Subfield node.

    The id is the OpenAlex URI (e.g., ``https://openalex.org/subfields/2712``).
    display_name is the human label used in UIs and summaries.
    """

    id: str | None
    display_name: str

    def to_dict(self) -> dict:
        return {"id": self.id, "display_name": self.display_name}

    @classmethod
    def from_dict(cls, d: dict | None) -> "TopicNode | None":
        if d is None:
            return None
        return cls(id=d.get("id"), display_name=d.get("display_name") or "")


@dataclass
class Topic:
    """An OpenAlex Topic with its position in the hierarchy.

    score is OpenAlex's classifier confidence for this topic on this paper
    (None when the Topic object came from a profile filter rather than a
    paper classification).
    """

    id: str | None
    display_name: str
    subfield: TopicNode | None = None
    field: TopicNode | None = None
    domain: TopicNode | None = None
    score: float | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "subfield": self.subfield.to_dict() if self.subfield else None,
            "field": self.field.to_dict() if self.field else None,
            "domain": self.domain.to_dict() if self.domain else None,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Topic":
        return cls(
            id=d.get("id"),
            display_name=d.get("display_name") or "",
            subfield=TopicNode.from_dict(d.get("subfield")),
            field=TopicNode.from_dict(d.get("field")),
            domain=TopicNode.from_dict(d.get("domain")),
            score=d.get("score"),
        )


@dataclass
class Paper:
    """A single publication record. JSON-serializable; holds no PDF bytes.

    embeddings is keyed by embedding-model name so the same Paper can carry
    vectors from multiple models side-by-side (e.g., a placeholder hash
    embedding and a real SPECTER2 one). The Selector decides which key to
    read at score time.

    local_path is the on-disk path to the PDF *when we have one*. Profiles
    built from a CSV manifest often have it; profiles built from
    OpenAlex-only lookups (abstract-only) do not. Keeping the path lets us
    re-open the PDF later for RAG chat grounding or body-text extraction
    without storing binary data in the Profile JSON.
    """

    doi: str | None
    openalex_id: str | None
    title: str
    abstract: str = ""
    year: int | None = None
    venue: str | None = None
    mesh: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    substances: list[str] = field(default_factory=list)
    body_text: str = ""
    embeddings: dict[str, list[float]] = field(default_factory=dict)
    primary_topic: Topic | None = None
    topics: list[Topic] = field(default_factory=list)
    local_path: str | None = None
    source: str = "unknown"
    added: str | None = None  # ISO-8601 UTC timestamp of ingest
    pdf_url: str | None = None
    oa_status: str | None = None

    def to_dict(self) -> dict:
        return {
            "doi": self.doi,
            "openalex_id": self.openalex_id,
            "title": self.title,
            "abstract": self.abstract,
            "year": self.year,
            "venue": self.venue,
            "mesh": list(self.mesh),
            "keywords": list(self.keywords),
            "substances": list(self.substances),
            "body_text": self.body_text,
            "embeddings": {k: list(v) for k, v in self.embeddings.items()},
            "primary_topic": self.primary_topic.to_dict() if self.primary_topic else None,
            "topics": [t.to_dict() for t in self.topics],
            "local_path": self.local_path,
            "source": self.source,
            "added": self.added,
            "pdf_url": self.pdf_url,
            "oa_status": self.oa_status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Paper":
        return cls(
            doi=d.get("doi"),
            openalex_id=d.get("openalex_id"),
            title=d.get("title") or "",
            abstract=d.get("abstract") or "",
            year=d.get("year"),
            venue=d.get("venue"),
            mesh=list(d.get("mesh") or []),
            keywords=list(d.get("keywords") or []),
            substances=list(d.get("substances") or []),
            body_text=d.get("body_text") or "",
            embeddings={k: list(v) for k, v in (d.get("embeddings") or {}).items()},
            primary_topic=Topic.from_dict(d["primary_topic"]) if d.get("primary_topic") else None,
            topics=[Topic.from_dict(t) for t in (d.get("topics") or [])],
            local_path=d.get("local_path"),
            source=d.get("source") or "unknown",
            added=d.get("added"),
            pdf_url=d.get("pdf_url"),
            oa_status=d.get("oa_status"),
        )

    def embedding_for(self, model: str) -> list[float] | None:
        """Return the embedding for a given model, or None if not stored."""
        return self.embeddings.get(model)


# A normalized title shorter than this is not evidence of anything. Two
# unrelated papers really can both be called "Editorial" or "Correction",
# and short generic titles are exactly where a title-based match stops
# meaning "same paper".
MIN_TITLE_KEY_LEN = 30


def title_key(title: str | None) -> str:
    """Dedup key for a title: lowercase alphanumerics, or "" if untrustworthy.

    OpenAlex regularly carries the preprint, the version of record and the
    conference copy of one paper as three works with three ids, so id
    equality does not answer "same paper" and titles have to.

    Punctuation is dropped because that is where the copies differ —
    hyphenation, a trailing period, a bracketed "[Preprint]". Comparing
    ``strip().lower()`` alone, as the persistence layer used to, let those
    pairs through while collapsing every paper titled "Editorial" into one.
    """
    key = "".join(c for c in (title or "").lower() if c.isalnum())
    return key if len(key) >= MIN_TITLE_KEY_LEN else ""
