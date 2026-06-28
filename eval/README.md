# Evaluation

A RAG system has two failure surfaces, and this suite measures both:

1. **Retrieval** — does it fetch the right speeches? If not, no prompt saves the answer.
2. **Generation** — given those speeches, is the answer grounded, cited, relevant, and does it
   *decline* when the corpus can't answer?

Everything runs across all 29 ParlaMint parliaments, compares the plain and agentic pipelines,
and writes timestamped, provenance-stamped results to `eval/results/` so runs are comparable over
time. The metric and judging code is unit-tested offline (`eval/tests/`, no Qdrant/API needed).

```
eval/
  evallib/            shared library (unit-tested)
    metrics.py          ranking metrics + bootstrap CIs
    judge.py            pattern-based relevance judging (cross-lingual, deterministic)
    llm_judge.py        LLM-as-judge for generation (faithfulness / citations / relevance / refusal)
    searchers.py        ablation variants: plain · rerank · transform · agentic
    seed_topics.py      curated golden topics (human-authored starting point)
    goldenset.py        golden-set loader
    provenance.py       git sha + config + timestamp stamped on every run
    persistence.py      write results/<ts>_<kind>.json + history.jsonl
  build_golden_set.py   verify seed topics against the corpus -> golden_set.jsonl
  golden_set.jsonl      the verified golden set (committed)
  refusal_set.jsonl     out-of-corpus questions that SHOULD be refused
  run_eval.py           known-item retrieval eval (synthetic, single-source)
  retrieval_eval.py     golden-set retrieval eval (pattern-judged, ablations, CIs)
  generation_eval.py    end-to-end answer quality + refusal (LLM-as-judge)
  filter_check.py       asserts metadata filters actually constrain retrieval
  regrade.py            re-score a saved retrieval run with new patterns — offline, no Qdrant
  tune_rerank_pool.py   sweep the reranker candidate-pool depth (no API) to set RERANK_CANDIDATES
  report.py             render the newest runs into results/REPORT.md
  run_all.py            one-command orchestrator: build -> retrieval -> generation -> report
  calibration/          validate the pattern judge against gold labels (see calibration/README.md)
  tests/                offline unit tests for metrics + judge + parsing
```

## Two ways to judge retrieval

### 1. Known-item (`run_eval.py`) — synthetic, single-source

Sample a speech → an LLM writes a question it answers → retrieve over the whole corpus → find the
**rank of that exact speech**. Zero hand-labeling, covers all 29 countries, great for catching
regressions. But the question is derived from the target, so it's *optimistic*, and only the one
source counts as relevant — so it **under-credits** real recall.

```bash
python eval/run_eval.py --per-country 2          # both methods, k=10
```

### 2. Golden set (`retrieval_eval.py`) — pattern-judged, multi-relevant

The fix for under-crediting. A **curated set of real European parliamentary topics** (Russia's
2022 invasion of Ukraine, COVID measures, the Greek debt crisis, the 2015 refugee surge, energy
prices, NATO accession, …), each with a **relevance pattern**. Any retrieved speech matching the
pattern counts as relevant — so the eval credits *every* on-topic speech from *any* parliament.

**Cross-lingual judging without a translator.** The corpus is in each parliament's native language,
so English keywords would only match the English (GB) speeches. The patterns instead lean on tokens
that recur *across* languages — international proper nouns (Ukraine, COVID, NATO, Brexit, euro, with
Cyrillic/Greek spellings) and shared Latin/Greek stems (`migr`, `energ`, `infla`, `pandemi`,
`klima`, `pensi`). The result: an English question is judged correctly against speeches in ~27
languages. (Coverage per topic is printed when you build the set.)

**The golden set is verified against the corpus, not just asserted.** `build_golden_set.py` scans
all 29 parsed parquets and counts how many indexed speeches actually match each topic (per country),
keeping only well-attested items and baking the counts into `golden_set.jsonl`. So every golden
item is known to have real targets in the index.

```bash
python eval/build_golden_set.py                  # verify seeds -> golden_set.jsonl (+ coverage report)
python eval/retrieval_eval.py --k 10             # all methods, with CIs
python eval/retrieval_eval.py --methods plain,agentic --limit 5   # quick
```

**Metrics** (mean with 95% bootstrap CI over the golden items):

| metric | reads as |
|---|---|
| **MRR** | how high the first relevant speech lands |
| **success@k** (hit@k) | did *any* relevant speech reach the top k |
| **precision@k** | what fraction of the top k are on-topic — the metric reranking should move |
| **nDCG@k** | rewards putting relevant speeches higher (graded: on-point hits score 2) |
| **langs reached** | distinct parliaments among the relevant top-k hits (cross-lingual breadth) |

### Ablations — what each stage buys you

Both retrieval evals run four configurations of the *same* production code, so each component's
contribution is isolated:

| method | pipeline |
|---|---|
| `plain` | dense vector search only (baseline) |
| `rerank` | over-fetch → cross-encoder rerank → top k |
| `transform` | multi-query + HyDE → RRF fuse → top k (no rerank) |
| `agentic` | the full shipped pipeline (transform + RRF + rerank + self-correct) |

## Generation eval (`generation_eval.py`) — LLM-as-judge

Retrieval metrics say nothing about the answer the user reads. This runs real questions through
the production path (retrieve → generate) and has a **stronger model judge each answer** (Sonnet
judging the Haiku generator, so it isn't grading its own homework).

- **In-corpus quality** — the golden questions, scored 1–5 on **groundedness** (every claim
  supported by the speeches), **citation validity** (each `[n]` backs its statement), and
  **answer relevance**.
- **Out-of-corpus handling** — `refusal_set.jsonl` holds questions a parliamentary speech would
  not answer (a Sachertorte recipe, the square root of 2025, a Wi-Fi password…). Correct behaviour
  is to **either decline or answer only from a cited speech**; the one failure is a substantive
  answer with **no support in the provided speeches** (a hallucination). The judge sees the sources,
  so a properly-grounded answer is credited, not mislabelled (see the note below — this mattered).

```bash
python eval/generation_eval.py                   # golden Qs + refusal set, agentic retrieval
python eval/generation_eval.py --n 8 --retrieval plain
python eval/generation_eval.py --skip-refusal --gen-model claude-sonnet-4-6
```

## One command

```bash
python eval/run_all.py             # build -> retrieval -> generation -> report (needs Qdrant + API key)
python eval/run_all.py --quick     # tiny subset, just to smoke the wiring
```

## Supporting checks

```bash
python eval/filter_check.py        # assert country/year/party/domain filters constrain hits (needs index)
python -m pytest eval/tests -q     # offline unit tests for metrics + judge (no Qdrant/API)
python eval/report.py --print      # render eval/results/REPORT.md from the newest runs
python eval/regrade.py             # re-score the newest retrieval run with the current patterns (offline)
```

When you tighten a relevance pattern, `regrade.py` re-scores the *last* retrieval run against
the saved `hit_ids` in seconds — no 20-minute Qdrant re-run — so judge iteration is cheap. (That
loop is how the `eu_enlargement` and `housing` pattern gaps in this corpus were found and fixed:
the retriever was returning the right native-language speeches, but the judge's keyword list was
missing the German/Portuguese/Dutch forms and scoring them as misses.)

## Changes this eval drove

The suite isn't just a scoreboard — it's the optimisation signal. Concrete changes made by
running it and reading the results:

- **Relevance-judge bugs** (`eu_enlargement`, then housing/health/education/labour): the per-domain
  table showed topics at 0 success while the retriever was clearly returning the right speeches —
  the judge patterns were missing the German/Portuguese/Dutch/English surface forms. Fixed; housing
  precision@10 went 0.00 → 0.90.
- **`RERANK_CANDIDATES` 60 → 150** (`tune_rerank_pool.py`): a no-API sweep showed a deeper
  cross-encoder pool lifts rerank MRR, precision@10 and nDCG@10 together; past ~150 precision keeps
  creeping but MRR falls. `RERANK_MAX_LENGTH` was swept too and left at 512 (gains within noise).
- **A "hallucination" metric that was actually an eval bug** — the instructive one. The refusal set
  flagged the generator "hallucinating" on common-knowledge questions (Romeo & Juliet → Shakespeare,
  Titanic → 1912, photosynthesis → CO₂). Inspecting the answers showed the opposite: the model was
  **correctly grounding in real retrieved speeches and citing them** — across 3.4M speeches, MPs
  genuinely have mentioned Shakespeare, the Titanic sinking "en 1912", and photosynthesis. The fault
  was the eval: (1) those refusal items were *answerable* from the corpus (bad negatives — confirmed
  by checking the top reranked speech for each), and (2) `judge_refusal` never saw the sources, so it
  scored a properly-cited answer as a hallucination. Fixes: the refusal set was rebuilt from
  corpus-verified *truly* unanswerable questions (recipes, arithmetic, personal/real-time queries),
  and the judge now sees the speeches and credits *refused OR grounded-in-a-cited-speech*, counting
  only an unsupported answer as a hallucination. A tentative "stricter prompt" change was **reverted**
  — it made even Sonnet decline a legitimately-grounded answer (1912 *was* in the speech). Result on
  the corrected set: **0% hallucination**, i.e. the system was grounding correctly all along.
- **`RERANK_MAX_LENGTH`** swept and left at 512 (gains within noise). **Resilient generation eval**:
  a single failed item used to abort a whole run; it now retries with backoff, throttles between
  items, and records per-item failures instead of crashing.
- **Calibrating the judge exposed an English bias — then fixed it.** A strong model (Opus) graded a
  balanced sample of query–speech pairs (`calibration/`); the pattern judge scored **precision 1.000
  but recall 0.533**, and every miss was a relevant *non-English* speech the keyword list couldn't see
  (Portuguese *corrupção*, Spanish *corrupción*, Dutch *nucleaire*, Ukrainian Cyrillic, …). Fix:
  diacritic-insensitive matching (`judge.fold`) + broadened stems → recall **0.533 → 0.800**, no new
  false positives. The golden set was also grown to **100 corpus-verified topics** across ~25 domains.

## What these numbers do and don't tell you

- **Known-item is relative, not absolute.** The question is derived from the target, so scores run
  optimistic. Fair for comparing methods and catching regressions; not a true recall figure.
- **The golden set fixes under-crediting but adds its own bias.** Pattern judging credits any
  on-topic speech, but a pattern can miss a relevant speech that avoids the keywords (false misses)
  or, rarely, credit an off-sense match (false hits). The corpus-verification step and the
  multilingual stems keep both low, but it is a heuristic judge, not human relevance labels.
- **Scoped retrieval.** Golden items retrieve within their declared country/date scope (mirroring
  how a user asks), which makes the haystack smaller than an unscoped search — read the absolute
  numbers with that in mind; the *method comparison* is still apples-to-apples.
- **LLM-as-judge is a model, not ground truth.** Using a stronger judge than the generator and
  strict JSON rubrics reduces self-bias, but judges have their own blind spots; treat the scores as
  a strong signal, not a certificate.
- **Synthetic questions are stochastic.** Bootstrap CIs are reported precisely because small samples
  are noisy — widen the sample (`--per-country`, more golden items) for a steadier signal.

## Results

Each run writes a full JSON (`eval/results/<timestamp>_<kind>.json`) with provenance, config, every
per-item record, and the aggregate summary, plus a headline line to `eval/results/history.jsonl`.
`report.py` turns the newest runs into `eval/results/REPORT.md`. Results are gitignored — they're
machine-specific artefacts, not source.
