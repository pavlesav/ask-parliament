# Calibrating the pattern judge

The golden-set retrieval eval rests on a cheap, deterministic **pattern judge** (`evallib/judge.py`):
a retrieved speech is "relevant" if its text matches the topic's regex patterns. That judge decides
every precision@k / MRR / nDCG number — so how much should we trust it? This directory measures it
against gold labels, the way you'd validate any weak-supervision signal.

## Method

1. **`dump_pairs.py`** (no API) samples a balanced set of (query, speech) pairs across a spread of
   domains and corpus languages. For each query it takes the highest-reranked speeches the pattern
   marks **relevant** *and* the highest-reranked ones it marks **irrelevant** — deliberately
   adversarial, to surface both error directions (especially false negatives). Writes `pairs.jsonl`
   with the full speech text, the pattern verdict, and the reranker score.
2. **Gold labels** (`labels.jsonl`): each pair is read in full and labelled relevant/irrelevant.
   Here the grader was a strong model (Opus 4.8) reading the original-language speech — a stand-in
   for a human annotator, not a replacement for one.
3. **`score.py`** reports accuracy, precision, recall, F1, Cohen's κ, the confusion matrix, and every
   disagreement, treating the gold label as truth and the pattern judge as the predictor.

```bash
QDRANT_URL=... python eval/calibration/dump_pairs.py      # sample pairs (no API)
# label pairs.jsonl -> labels.jsonl
python eval/calibration/score.py                          # agreement on the dumped verdicts
python eval/calibration/score.py --recompute              # re-score after a pattern/judge change
```

## What it found (46 pairs)

| | precision | recall | F1 | κ |
|---|---|---|---|---|
| **before** | 1.000 | 0.533 | 0.696 | 0.05 |
| **after fix** | 1.000 | 0.800 | 0.889 | — |

- **The pattern judge never over-credits** — across 24 pattern-relevant pairs in many languages, the
  gold label agreed every time (precision 1.000, zero false positives).
- **But it systematically under-credits relevant non-English speeches.** Every one of the 21
  disagreements was a *false negative*, and they clustered by language/spelling: Portuguese
  (*Ucrânia*, *Grécia*, *corrupção*), Spanish (*corrupción*), Dutch (*nucleaire*, *belasting*), French
  (*réforme*, judicial-independence phrasing), Ukrainian/Bulgarian (Cyrillic spelling variants),
  Polish (*rybołówstwo*), Icelandic (*fiskveiði*), plus a couple of English speeches that discuss a
  topic without the keyword (*NHS* funding). **The eval was English-biased.**

This matters: it means the retrieval eval's absolute precision/nDCG are *under-estimates*, and any
metric sensitive to the language mix of the results (e.g. cross-lingual breadth) is undercounted.
Relative method comparison is partly protected because all methods share the same judge.

## The fix it drove

Two changes, then re-scored on the same gold labels:

1. **Diacritic-insensitive matching** (`judge.fold`, NFKD): accented spellings now match unaccented
   stems (*Grécia*→grecia, *réforme*→reforme, *nucléaire*→nucleaire). Greek/Cyrillic are scripts, not
   accents, so they're untouched.
2. **Broadened stems** for the flagged topics (e.g. `corrupt`→`corrup` to catch ES/PT; added
   Norwegian/Danish migration stems, Cyrillic Ukrainian pension forms, `NHS`, Polish/Icelandic
   fisheries terms, Dutch/French judicial-independence phrasing).

Result: recall 0.53 → 0.80, **15 of 21 false negatives fixed, no new false positives.**

## Honest caveats

- **Adversarial sample.** Pairs were the highest-reranked speeches, half forced to the pattern's
  "irrelevant" bin — exactly where false negatives hide. So 0.53/0.80 is a *stress-test* recall, a
  lower bound, not the corpus-wide recall (which is higher, since most speeches aren't this borderline).
- **Precision is under-tested.** High-reranked speeches are nearly all on-topic, so the sample held no
  hard false-positive cases; precision 1.000 means "no over-crediting seen here", not "provably none".
- **Gold by a strong model, not a human.** Using Opus to read full speeches is a reasonable proxy and
  far better than trusting the pattern judge to grade itself, but it carries the model's own blind
  spots. A human-labelled subset would be the next rung.
