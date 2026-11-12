# UMLS topics: from a seed paper's text to OpenAlex topics

An interest searches OpenAlex by topics. Most come from the seeds' own
OpenAlex records. UMLS adds more, and it matters most for papers that
OpenAlex has no record for, such as an uploaded PDF. The steps:

1. Read the biomedical concepts in the paper's text with SciSpacy and the UMLS linker.
2. Keep, for each concept, the sentences it was read in.
3. Embed each concept in its sentences and find the nearest OpenAlex topics.
4. Choose which of those topics the interest gets. They are added switched off.

This document describes the rules in force since 2026-11-12: CTX, CTX-3b and
T-5b/T-5c in the functional spec. The record with the experiments behind
them is `tests_func/records/35_topic_matching_without_title.md` in the
deployment repo.

---

## 1. The text that is read

`rag_lib/umls/text.py::umls_input_text`

- **Title first**, then the abstract.
  - The linker expands an abbreviation only when its long form appears in
    the same text.
  - An abstract that says "T2DM" nine times gives no diabetes concept, but
    the title usually spells the subject out.
- **No abstract**: the body text is used instead, from where the reading
  starts (past the journal name, authors and affiliations of a PDF), capped
  at `UMLS_TEXT_LIMIT` = 20,000 characters.

## 2. Concept extraction

`rag_lib/umls/extractor.py::extract_umls_concepts`

**Pipeline**: `en_core_sci_lg` → `abbreviation_detector` → `scispacy_linker`.
The linker settings are:

| Setting | Value |
|---|---|
| `linker_name` | `umls` |
| `resolve_abbreviations` | `True` |
| `threshold` | `RADAR_UMLS_MIN_CONFIDENCE` |
| `k` | 30 |
| `max_entities_per_mention` | 1 |
| `no_definition_threshold` | 0.95 |

The abbreviation detector must come before the linker. Without it,
`resolve_abbreviations` silently does nothing, and acronyms are linked on
their surface form at about 1.0 (for example "eGFR" became Epidermal Growth
Factor Receptor).

**For each entity**:
1. Take the top UMLS match.
2. Keep it if its score is at least `RADAR_UMLS_MIN_CONFIDENCE` (0.7) and
   its semantic type is a relevant one (`rag_lib/umls/semantic_types.py`).
   Types that only say "this is research", such as Research Activity, are
   excluded.

**For each concept** (deduplicated by CUI):
- Keep its highest score as its `confidence`.
- Keep the `RADAR_UMLS_MAX_CONCEPTS` (30) surest concepts.

### Mentions (CTX-1)

Each concept keeps up to `MAX_MENTIONS` = 3 *mentions*. A mention is the
word the text used and the sentence it is in.

- **Order**: the surest mention comes first, and each sentence is used once.
- **Sentence**: the spaCy sentence. When it is shorter than 120
  characters, the next sentence is appended.
- **Length**: over `SENTENCE_MAX_CHARS` = 1,000 characters, only the 1,000
  centred on the word are kept. A PDF paragraph without a full stop is one
  "sentence", and its first 1,000 characters often did not contain the word.
- **Not kept as a mention**:
  - a word of three letters or digits or fewer, unless the text defines it
    as an abbreviation (for example "Middle East respiratory syndrome
    (MERS)"). A formula's "D" linked as aspartic acid at 1.0;
  - a mention in a formula-like sentence, meaning fewer than half of its
    tokens are words.

The concept itself is still extracted, with the same confidence; only these
mentions are left out. A concept with no usable mention is not mapped
(§3), because its name alone is where misread words come from.

Mentions do not change which concepts are found or their confidences.

Stored on the paper as `papers.umls_concepts_json`:

```json
{"cui": "C0002395", "name": "Alzheimer's Disease", "tui": "T047",
 "semantic_type": "Disease or Syndrome", "confidence": 0.9979,
 "mentions": [{"text": "Alzheimer’s Disease Study",
               "sentence": "Consistent with this model, recent advances in fluid and neuroimaging …"}]}
```

## 3. Mapping a concept to OpenAlex topics (CTX-2, CTX-3b)

`rag_lib/umls/topic_mapper.py::map_concepts_to_topics`

**Which concepts are mapped**: only those with confidence at least
`RADAR_UMLS_TOPIC_MIN_CONFIDENCE` (0.9) and at least one mention. Every
mention costs an embedding call, so the others are not embedded at all.

**How each is mapped**:
1. Embed every mention as `"<concept name>: <sentence>"` with
   `RADAR_UMLS_EMBEDDING_MODEL` (`mxbai-embed-large`, through Ollama).
2. Normalise each vector. Skip a zero or NaN vector.
3. Average the vectors.
4. Search the **rich topic index** (§7) for the `NEIGHBOURS` = 3 nearest
   topics, with no bar on the similarity.
5. Store them as rank 1, 2, 3.

Stored on the paper as `papers.umls_mapped_topics_json`:

```json
{"topic_id": "https://openalex.org/T10086", "display_name": "Alzheimer's disease research and treatments",
 "similarity": 0.7444, "source_cui": "C0002395", "source_name": "Alzheimer's Disease",
 "confidence": 0.9979, "mention": "Alzheimer’s Disease Study", "mapping": "context-2", "rank": 2}
```

(Both examples are real rows from a seed on the test site; the sentence is shortened.)

**Why sentences and not the name**: the name alone matched names.
"Middle East Respiratory Syndrome" landed on Inflammatory Bowel Disease. On
biomedical papers, the paper's primary topic was among the five chosen:

| Input | By name | By sentences |
|---|---|---|
| PDF bodies | 0% | 83% |
| Abstracts | 19% | 78% |

**Why three neighbours**: when a concept's nearest topic is one the paper
already has, the concept added nothing. Every concept of the paper
"Immunity in Alzheimer's disease" landed on its own two Alzheimer topics.
On 300 abstracts:

| Neighbours | Topics added per paper | Far-off topics |
|---|---|---|
| 1 | 2.91 | 18% |
| 2 | 4.21 | 14% |
| **3** | **4.58** | **11%** |
| 5 | 4.80 | 10% |

At 5, single papers start getting far neighbours, such as "Immune response"
→ Atherosclerosis. A far-off topic is one in another field and ranked below
50th by the whole text's embedding.

## 4. Choosing the interest's UMLS topics (U.2, CTX-3, CTX-3b)

`rag_lib/umls/topic_mapper.py::merge_umls_topics`, called from
`rag_lib/api/services/wizard.py::_merge_umls_topics`

**Per seed** — take all its (concept, topic) pairs, nearest first. Ties go
by confidence, then rank, then name. Skip a pair when:
- the interest already has the topic (see below);
- another concept of the same seed already gave the topic;
- the concept already gave a topic. A concept gives one topic: its nearest
  one that the interest lacks.

Take `RADAR_UMLS_TOPICS_PER_PAPER` (5) per seed.

**Across seeds**:
- Each topic is listed once, with its nearest concept.
- More seeds first, then nearer.
- At most `RADAR_UMLS_MAX_TOPIC_ADDITIONS` (20) for the interest.

**Similarity bar**: none, since `RADAR_UMLS_TOPIC_MIN_SIMILARITY` = 0.0. A
sentence measures 0.6–0.8 against the right topic's description and the
wrong ones' alike, so a bar cannot separate them. The ranking and the five
per seed keep wrong topics out.

**Switched off**: every UMLS topic is added off. The user switches on the
ones to search.

**"The interest already has it"** means any of:
- the same id;
- the same name;
- a cosine of 0.75 or more to one of the interest's topics in the
  *name-only* index.

UMLS topics are not compared with each other this way: sibling topics in one
field measure 0.72–0.82.

The interest's topics, for this purpose, are every non-UMLS topic of the
list. `wizard._already_the_interests` builds the same list once without the
UMLS merge to get them:
- the seeds' OpenAlex topics;
- topics from the researcher's profile, including not-wanted ones the user
  added and those kept while the profile cannot be read in full;
- adopted topics;
- Prosopia interests;
- Claude-generated topics;
- on a saved interest, the OpenAlex topics of removed seeds that the list
  keeps.

They used to be listed by hand, and the list missed cases. Then a UMLS
concept could take one of these topics: its source and label changed, and
its near-duplicates came in marked "new".

Each UMLS entry in `topic_filters_json` records:
- `concept`, `confidence` and `mention`: what it came from;
- `similarity`;
- `count`: how many seeds chose it;
- `seeds`: which seeds chose it.

## 5. What the user sees

The topic list shows a UMLS topic as **from "word" (UMLS: concept name)**.
When the word and the name are the same, only the name is shown (CTX-4).
"Disease" and "Source" are generic concepts that still reach the right
topic through their sentences; the word the paper used says where the topic
came from.

## 6. When the seeds change (spec T-2 … T-5c)

`wizard._keep_unmatched_seed_topics`, `wizard._seed_topic_codes`

A saved interest's list is rebuilt from its seeds. **No saved topic is
dropped by a rebuild.**

| Case | What happens |
|---|---|
| No current seed brings the topic (its seed was removed, or its seed now maps elsewhere) | Kept, on or off as saved, marked *unmatched* ("brought by seed papers you removed") |
| A current seed still brings it, but the merge did not choose it (cap, or a nearer concept) | Kept as saved, on or off, not marked (T-5c). The UMLS topics may then pass 20 |
| A topic the new seeds bring | Shown *new*, off |
| A new interest's draft | Follows its seeds: nothing is kept |

A UMLS topic counts as brought by a seed only when both hold:
- the seed is in the topic's `seeds`;
- the seed's current `umls_mapped_topics_json` still lists it, at any rank.
  A seed not extracted yet counts as bringing it.

Being one of a seed's unchosen neighbours does not count.

## 7. The rich topic index

These files in `RADAR_UMLS_CACHE_DIR` (`data/umls_cache/`) are data, not
code, and are not in git:

```
openalex_topics_rich.json              4,516 topics with description and keywords
openalex_topic_rich_embeddings.npz     row-normalised float32, 1024-d
openalex_topic_rich_index.meta.json    {"embedding_model": "mxbai-embed-large", "dim": 1024, ...}
```

**What each row is**: a topic embedded as its name, OpenAlex's description
and its keywords, higher-scored first (`cache.rich_topic_text`). The older
name-only files (`openalex_topics.json`, `openalex_topic_embeddings.npz`)
are still used for the "already has it" test and by other features.

**Build**:

```sh
python scripts/build_topic_index.py --rich --output-dir data/umls_cache
# or, from topics fetched before, without calling OpenAlex:
python scripts/build_topic_index.py --rich --topics-json openalex_topics_rich.json --output-dir data/umls_cache
```

- **Cost**: about 23 OpenAlex calls with the configured key, plus one
  embedding per topic. That takes minutes on a CUDA Ollama.
- **Swap**: the three files are swapped in together.
- **Model check**: the meta file records the model. An index built with
  another model is refused (`TopicIndexMismatch`). An index without a meta
  file is refused on its dimension.

**Without the index** the backend starts, but no paper's concepts can be
mapped:
- old UMLS data is kept as it is;
- a failed extraction waits six hours before it is retried;
- the problem is reported in the startup log (`api.umls_topic_index_unusable`),
  in `/api/health` (`"umls_topic_index": "unusable"`) and in `radar.sh status`.

## 8. When extraction runs, and older data

- **When**: at upload or import (`vault._try_extract_umls`), and for seeds
  still missing data when the topic list is built
  (`wizard._extract_missing_umls`).
- **Failures**: a failed extraction waits `_UMLS_RETRY_AFTER_S` (6 h) before
  the next try, and the list says "failed" rather than "computing".
- **Older data** is extracted again once, the next time an interest's topics
  are built (`wizard.forget_stale_umls_mappings`,
  `topic_mapper.stale_extraction`). Older data means either:
  - concepts stored without `mentions`, or
  - a mapping whose `mapping` is not `context-2`: name-only, or one topic
    per concept.

  The first build after the 2026-11-12 release is slower for that reason.
- **Nothing found**: `"[]"` is not older data.

## 9. Settings

| Setting | Default | What it does |
|---|---|---|
| `RADAR_UMLS_ENABLED` | `true` | UMLS on or off |
| `RADAR_UMLS_SPACY_MODEL` | `en_core_sci_lg` | SciSpacy model |
| `RADAR_UMLS_MIN_CONFIDENCE` | 0.7 | linker score to keep a concept |
| `RADAR_UMLS_MAX_CONCEPTS` | 30 | concepts kept per paper |
| `RADAR_UMLS_TOPIC_MIN_CONFIDENCE` | 0.9 | concept confidence to be mapped and chosen; a change affects newly extracted papers only |
| `RADAR_UMLS_TOPIC_MIN_SIMILARITY` | 0.0 | similarity bar when choosing (none) |
| `RADAR_UMLS_TOPICS_PER_PAPER` | 5 | topics per seed |
| `RADAR_UMLS_MAX_TOPIC_ADDITIONS` | 20 | UMLS topics per interest |
| `RADAR_UMLS_EMBEDDING_MODEL` | `mxbai-embed-large` | must match the index |
| `RADAR_UMLS_CACHE_DIR` | `data/umls_cache` | UMLS data and topic indexes |

`NEIGHBOURS` (3), `MAX_MENTIONS` (3), the 120 and 1,000 character limits,
and the 0.75 near-duplicate cosine are constants in the code.

## 10. Checking it by hand

Both commands run the real pipeline with the deployment's settings, and say
`MISMATCH` if their walkthrough disagrees with `merge_umls_topics`:

```sh
# re-extract an interest's seeds and diff against what is stored
python -m rag_lib.umls.extractor <profile-slug>

# one paper: concepts, mentions, its 3 nearest topics, and what step 3 would offer
python -m rag_lib.umls.extractor paper --pdf paper.pdf
python -m rag_lib.umls.extractor paper --title "…" --abstract "…"
```

## 11. Code map

| What | Where |
|---|---|
| Input text | `rag_lib/umls/text.py` |
| Extraction, mentions, self-check | `rag_lib/umls/extractor.py` |
| Semantic type filter | `rag_lib/umls/semantic_types.py` |
| Topic indexes (name-only and rich), model check | `rag_lib/umls/cache.py` |
| Mapping, choosing, staleness | `rag_lib/umls/topic_mapper.py` |
| When it runs, back-off | `rag_lib/api/services/vault.py` |
| Interest's topic list, "already has", seed removal | `rag_lib/api/services/wizard.py` |
| Index build | `scripts/build_topic_index.py` |
| Health | `rag_lib/api/routers/health.py` |
| Tests | `tests/test_umls_extractor.py`, `test_umls_topic_mapper.py`, `test_umls_topics_off.py`, `test_umls_ctx_review.py`, `test_seed_removal_topics.py`, `test_build_topic_index_rich.py` |
