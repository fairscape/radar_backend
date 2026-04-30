# radar-backend usage

End-to-end recipe: take a directory of PDFs, build a Profile JSON, gather +
embed related papers from OpenAlex, persist a ranked candidate list.

## 0. One-time setup

```bash
cd radar/radar-backend
pip install -e '.[dev,phase1b]'
pytest -q   # 114 passed, 1 skipped
```

`[phase1b]` pulls SPECTER2 (`sentence-transformers`, `adapters`, `torch`),
`pdfplumber`, `chromadb`, `scipy`, `scikit-learn`. First call to
`specter2_embed` downloads ~500 MB of model weights from HuggingFace into
the local cache, then reuses them.

## 1. Build a profile from a PDF directory

```bash
python -m cli.ingest \
  --pdf-dir ../pdfs/neonatal \
  --name neonatal_vitals \
  --email you@example.com \
  --embedder specter2 \
  --out ../profiles/neonatal_vitals.json
```

Flags:
- `--pdf-dir` — directory of `*.pdf` (use `--csv` for a CSV manifest of DOIs/titles instead).
- `--name` — profile identifier; goes in the JSON.
- `--email` — OpenAlex polite-pool mailto. **Required**; OpenAlex calls fail without it.
- `--embedder` — `placeholder-v1` (hash-based, fast, semantically meaningless) or `specter2` (real, 768-dim).
- `--out` — where to write the Profile JSON.

What it does:

1. Globs `*.pdf` in the directory (sorted).
2. For each PDF: pdfplumber extracts title + body + best-effort DOI from text/metadata. **Corrupt or HTML-masquerading-as-PDF files are skipped with a warning** rather than crashing the whole build.
3. Resolves each paper on OpenAlex: DOI lookup first (`/works/doi:...`), title-search fallback (`/works?search=...`). Both are single-entity endpoints, free-tier unlimited.
4. Embeds via `build_embedding_input(paper)` → SPECTER2. Input format is labeled sections: `TITLE: ...\n\nABSTRACT: ...\n\nMESH: ...\n\nKEYWORDS: ...\n\nSUBSTANCES: ...\n\nBODY: ...`, truncated to 512 tokens (drop priority: body → substances → keywords → mesh; title + abstract always preserved).
5. Aggregates `topic_filters` — counts each paper's `primary_topic` and entries in its `topics[]` (up to 5 from OpenAlex), keeping the top-8 IDs at each of four hierarchy levels: `topics`, `subfields`, `fields`, `domains`.
6. Fits a `CentroidSelector` (mean of seed embeddings) and prints coherence diagnostics.
7. Writes the Profile JSON.

Expected stdout:

```
== CentroidSelector fit: neonatal_vitals ==
  n_seed: 11
  coherence_median: 0.925
  coherence_iqr: 0.029
  coherence_bimodal: False
  embedding_model: specter2
  fit wall_seconds: 0.004

== Top OpenAlex topic_filters ==
  topics:
    https://openalex.org/T11184  Neonatal and fetal brain pathology  (n=8)
    https://openalex.org/T11196  Non-Invasive Vital Sign Monitoring  (n=7)
    ...
```

Coherence-median rule of thumb (calibrated for SPECTER2 title+abstract):
- `> 0.75` — tight, single-topic seed
- `0.60 – 0.75` — acceptable, loose but coherent
- `< 0.60` — bimodal; consider splitting the corpus

## 2. Gather + embed + persist OpenAlex candidates

```bash
python -m cli.gather \
  --profile ../profiles/neonatal_vitals.json \
  --email you@example.com \
  --days 365 \
  --per-page 100 \
  --rate-sleep 1.0 \
  --limit 500 \
  --min-results 50 \
  --out ../gathered/neonatal_vitals_candidates.json
```

Flags:
- `--profile` — built Profile JSON from step 1.
- `--email` — OpenAlex mailto (same as ingest).
- `--days` — look-back window. Filter is `from_publication_date:<today - days>` (free-tier; `from_created_date` requires a paid OpenAlex plan).
- `--per-page` — OpenAlex page size, max 200. Smaller is gentler on the burst budget. Default 100.
- `--rate-sleep` — seconds of idle between successful API calls. Default 1.0.
- `--limit` — cap on candidates per tier.
- `--min-results` — if a tier returns fewer than this, fall through to the next tier.
- `--threshold` — drop candidates scoring below this cosine. Default keeps all.
- `--selector` — `max_seed` (default) or `centroid`. `max_seed` scores each candidate as `max_i cos(candidate, seed_i)` instead of cosine to the seed mean, so a candidate that's a near-twin of one distinctive seed isn't diluted by the others. `centroid` is the legacy mean-based rule.
- `--out` — where to write the ranked, embedded candidates.

**No automatic retries.** A 429 (or any other HTTP error) raises immediately and the run stops. The previous `_get` had a 5-attempt exponential backoff that was unhelpful when an IP is genuinely throttled — it just amplified the throttle. Re-run by hand once the rate-limit window clears.

### The tier algorithm

Three tiers, executed in order, stopping at the first that returns
`>= --min-results` candidates:

| Tier | Topics | Subfields | Semantics |
|---|---|---|---|
| 1. `must-have-AND` | All topics with **distinct-paper prevalence ≥ `--must-have-prevalence`** (default 0.5), each one **AND-required** | none | Strictest. Candidate must contain *every* core topic. Implemented by repeating `topics.id:` in the OpenAlex filter string (comma-as-AND, even for the same key). |
| 2. `top-topics-OR-subfields-AND` | Top-`--top-topics-n` topics (default 5) — **OR** | Top-`--top-subfields-n` subfields (default 2) — **AND** each | Candidate matches any one core topic *and* covers all core subfields. |
| 3. `subfields-OR` | none | Top-`--top-subfields-n` subfields — **OR** | Broadest fallback. |

**Prevalence is computed exactly** by walking `profile.papers` and counting distinct papers per topic-id. Not the same as `profile.topic_filters["topics"][i]["count"]`, which double-counts a topic that appears as both `primary_topic` *and* in `topics[]`.

Example tier filters built for `neonatal_vitals` (11 papers, two topics ≥ 50%):

```
[must-have-AND]   from_publication_date:...,type:article,language:en,topics.id:T11184,topics.id:T11196
[top-topics-OR-subfields-AND]   ...,topics.id:T11184|T11196|T10218|T10549|T13248,topics.subfield.id:2735,topics.subfield.id:2204
[subfields-OR]   ...,topics.subfield.id:2735|2204
```

For each fetched candidate the script:
1. Builds the embedding input with `build_embedding_input` (TITLE + ABSTRACT — OpenAlex doesn't serve MeSH/keywords/substances/body).
2. Embeds with the same model the seed profile used.
3. Scores by `--selector`: `max_seed` (max cosine to any seed; default) or `centroid` (cosine to seed mean).
4. Persists the full ranked list (vectors included) to `--out`. Each entry carries both raw `score` (cosine in `[0,1]`) and `percentile` (rank-based, in `[0,1]`, top=1.0). Use `percentile` when comparing across profiles — raw cosine bands are profile-specific.

### Output JSON shape

```jsonc
{
  "source_profile": "neonatal_vitals",
  "embedding_model": "specter2",
  "selector": "max_seed",
  "since": "2025-04-26",
  "tier_used": "must-have-AND",
  "filter": "from_publication_date:2025-04-26,type:article,language:en,topics.id:T11184,topics.id:T11196",
  "fetched": 69,
  "kept": 69,
  "ranked": [
    {
      "score": 0.979,
      "percentile": 1.0,
      "paper": {
        "doi": "10.1234/...",
        "openalex_id": "https://openalex.org/W...",
        "title": "...",
        "abstract": "...",                      // reconstructed from inverted index
        "year": 2025,
        "venue": "...",
        "embeddings": { "specter2": [768 floats] },
        "primary_topic": { "id": "...", "display_name": "...", "subfield": {...}, "field": {...}, "domain": {...} },
        "topics": [...],
        "source": "openalex_gatherer",
        ...
      }
    },
    ...
  ]
}
```

## 3. What OpenAlex gives back

Per `OpenAlexClient.slim_work` (`rag_lib/openalex_client.py:175-208`):

| Field | Returned? | Notes |
|---|---|---|
| Title | ✅ | |
| Abstract | ✅ | As `abstract_inverted_index`; `reconstruct_abstract` reassembles it |
| Topics + 4-level hierarchy | ✅ | `primary_topic` + up to 5 from `topics[]` |
| Year, venue, type, citation count | ✅ | |
| `open_access` (incl. `oa_url`) | ✅ | The gatherer does not currently follow `oa_url` to fetch full PDFs |
| Full text / body | ❌ | OpenAlex doesn't serve full text |
| MeSH / keywords / substances | ❌ | OpenAlex doesn't have these (PubMed does) |

So an embedded candidate sees `TITLE + ABSTRACT` only — same input SPECTER2 was trained on.

## File layout after a full run

```
radar/
├── pdfs/
│   ├── neonatal/                  # seed PDFs
│   └── fair-data/
├── profiles/
│   ├── neonatal_vitals.json       # built by cli.ingest
│   └── fair_data.json
└── gathered/
    ├── neonatal_vitals_candidates.json   # built by cli.gather
    └── fair_data_candidates.json
```
