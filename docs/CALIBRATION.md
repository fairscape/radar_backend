# Score calibration (SPECTER2 + proximity adapter)

Measured 2026-09-21 on two real seed folders (4 PDFs each) and 200
OpenAlex candidates per profile from the previous 30 days, using the
same code path as the wizard (`ingest_pdf` → OpenAlex enrichment →
`build_embedding_input` → SPECTER2). Reproduce with the measurement
script described at the bottom.

## What the scale looks like

| Quantity | Value |
|---|---|
| Pairwise cosine, random candidates sharing the profile's topics | median 0.847–0.850, p90 0.888, p95 0.90 |
| Pairwise cosine, candidates from different fields (FAIR × vitals) | median 0.819, p95 0.87 |
| Coherence median of random 4-paper sets from one field (300 draws) | mean 0.847, p95 0.88, max 0.891 |
| Coherence median of random mixed 4+4 sets | mean 0.831, max 0.867 |
| Coherence median, real FAIR seeds | 0.915 (IQR 0.026) |
| Coherence median, real vitals seeds | 0.926 (IQR 0.022) |
| Coherence median, FAIR + vitals seeds together | 0.878 (IQR 0.073) |
| Seeds' leave-one-out cosine to their own centroid | FAIR 0.927–0.952, vitals 0.928–0.962 |
| Other field's seeds against a centroid | 0.84–0.90 |
| 30-day candidates vs centroid (n=200) | p50 0.89, p90 0.92, p95 0.93, max 0.95–0.96 |
| Candidates passing at 0.85 / 0.90 / 0.92 / 0.94 | 190 / 73 / 25 / 3 of 200 |

Embedding seeds from the PDF body when OpenAlex has no abstract (two of
the four vitals papers) moved that set's coherence from 0.918 to 0.926
and left candidate scores unchanged to three decimals. That change is
not the cause of anything users noticed.

## What was wrong

- Coherence bands of 0.75 (tight) and 0.60 (loose) sit far below the
  noise floor: a random pick of four papers from the same field scores
  0.85, so every set read as "tight", including a deliberate mix of two
  topics at 0.88.
- The wizard's suggested threshold was "the 20th-best score, capped at
  0.85". On this scale 0.85 admits 95% of candidates, so the choice was
  effectively everything or nothing, on a slider drawn over 0.5–1.0
  where all the mass sits in the top tenth.
- Card colours used 0.95 / 0.925 on a raw score. No raw cosine reaches
  0.925 reliably, so nearly every card was grey. A later fix switched
  the displayed number to a rank percentile, which made the colours
  work but showed a number that no longer relates to the threshold.
- The profile page's threshold histogram read the `score` column,
  which the reranker overwrites with a batch-relative blend in [0, 1];
  the threshold is a raw cosine, so the two could not be compared.

## What `rag_lib/calibration.py` does instead

- **Coherence label** from the median: focused ≥ 0.90, broad ≥ 0.86,
  otherwise mixed; an IQR ≥ 0.07 downgrades focused to broad unless
  agreement is 80 or more (a real 5-paper set at median 0.934 / IQR
  0.058 was wrongly called broad at the earlier 0.05 gate). An
  **agreement** figure maps 0.82→0 and 0.95→100 for display.
- **Seed similarity band**: each seed's leave-one-out cosine to the
  centroid of the others. Stored on the profile (`seed_sim_*`).
- **Suggested threshold**: band minimum − 0.01, clamped between the
  75th and 98th percentile of the trial scan's scores. For the real
  sets this gives ≈0.917–0.918, i.e. roughly 25 papers a month.
- **Card level**: `high` when a paper is at least as similar as the
  least typical seed, `medium` when above the threshold, `low` below
  it. Rank percentile is kept only as a fallback for rows without a raw
  score, with cut-points at top 10% / top third.
- **Slider axis**: the observed scores plus the seed band, padded.

## Reproducing

The measurement script lives outside the repo (it needs network and the
`phase1b` extras). Its steps: ingest each PDF, resolve on OpenAlex,
embed three variants (current input, title+abstract only, title+body),
compute pairwise coherence per variant, gather 30 days of candidates
with `OpenAlexGatherer`, score with `CentroidSelector`, and sample
random subsets of candidates for the null distributions.
