# Retrieval evaluation

This measures one thing: **when someone asks a question, does the retriever find
the speeches from the debate they're asking about?** Generation quality isn't
scored here — if retrieval surfaces the right speeches, the grounded-generation
step has what it needs; if it doesn't, no prompt can fix it.

## How the golden set is built

`golden_set.jsonl` has 15 questions, each tied to one real, identifiable debate
(the 2022 compulsory-vaccination law, the day Russia invaded Ukraine, the 2014
Hypo Alpe Adria bad-bank debate, the 2017 face-veil ban, …). The themes span
COVID/health, defence and foreign affairs, the 2015 migration crisis, banking
and budget scandals, energy/environment, and civil rights.

For each question we need a **ground-truth set of relevant speeches**. We define
it lexically and **independently of the embeddings**: the speeches that match a
distinctive phrase (e.g. `eurofighter`, `temel[ií]n`) within the debate's date
window. Each question stores its exact criterion (`pattern`, `date_from`,
`date_to`), so the relevant set is fully reproducible and auditable — nothing is
hand-picked behind the scenes.

This is deliberately **not circular**: relevance is decided by keyword + date,
then we test whether the *semantic* (BGE-m3) retriever surfaces those speeches
from a one-line, often paraphrased question. It's the standard way to bootstrap a
small IR test set when you have no human relevance judgements, and it needs no
LLM-as-judge.

## Metrics

For each question we retrieve the top *k* speeches and compute:

- **MRR** (mean reciprocal rank) — `1 / rank` of the *first* relevant speech.
  Robust to how many speeches are relevant; the headline number.
- **hit@10** — was at least one relevant speech in the top 10? The most
  intuitive "did it work" signal.
- **recall@k** — fraction of the relevant set found in the top *k*. Reported for
  completeness, but read it with the ceiling in mind: a relevant set is a whole
  debate (7–68 speeches here), so when |R| > k, even perfect retrieval can't
  reach recall = 1 (e.g. |R| = 68, k = 20 caps recall@20 at 0.29).

## Two conditions

- **unfiltered** — query the whole 27-year corpus with no metadata filter.
- **year-scoped** — also pass the debate's year (what a user does with the
  sidebar year slider). Isolates the value of metadata filtering.

## Results (k = 20)

```
                              |  --- unfiltered ---   |  --- year-scoped ---
                              |  MRR   recall@10 hit10 | MRR  recall@10 hit10
 MEAN (15 questions)          | 0.23     0.04    0.40  | 0.50    0.11    0.80
```

**What this says.** Scoping to the year roughly **doubles** retrieval
effectiveness (MRR 0.23 → 0.50, hit@10 0.40 → 0.80): for 12 of 15 questions, a
speech from the *exact* target debate lands in the top 10. Unfiltered, the
retriever still nails distinctive, time-bound events — the compulsory-vaccination
and Brexit debates come back at rank 1 — but recurring topics get diluted, because
with no date cue the embeddings (correctly) return the topic from across all 27
years, pushing any single debate down.

**Why the unfiltered numbers are a conservative floor.** Our judged set is *one*
debate, but the same topic is debated on many other dates, and those speeches are
genuinely relevant too — they just aren't in our judged set, so they count as
misses when they displace judged speeches. True retrieval quality is therefore
at least what's reported here.

**Where it's genuinely weak.** `minimum_wage` and `pension_reform` score ~0 even
year-scoped: the target debates are phrased very differently from the query, and
exact-term matching might catch them where dense vectors don't — which motivated
trying hybrid retrieval (below).

## Hybrid (BM25 + vector) vs vector

`--method hybrid` fuses BM25 keyword search with the dense retriever using
Reciprocal Rank Fusion (see [../src/ask_parliament/hybrid.py](../src/ask_parliament/hybrid.py)).
Measured against the same golden set:

| method | condition | MRR | recall@10 | recall@20 | hit@10 |
|---|---|---|---|---|---|
| vector | unfiltered | 0.23 | 0.04 | 0.08 | 0.40 |
| hybrid | unfiltered | 0.23 | 0.04 | 0.06 | 0.40 |
| vector | year-scoped | **0.50** | 0.11 | 0.21 | 0.80 |
| hybrid | year-scoped | 0.42 | 0.11 | **0.22** | 0.80 |

**Hybrid is roughly a wash here — and that's the finding, not a failure.** It's a
genuine trade, visible per-question: hybrid *helps* exact-term and recurring
queries (`climate_targets` 0.33 → 1.00, `pension_reform` and `minimum_wage` off
the floor, `ukraine_invasion` to rank 1) but *hurts* queries where the dense
retriever was already perfect (`temelin_nuclear` 1.00 → 0.11, `vaccination_mandate`
1.00 → 0.50). RRF rewards *consensus*: a speech both retrievers like beats one only
the dense side ranks first, so a lone dense winner gets demoted.

Two honest caveats temper the apparent regression. First, some of that demotion is
our conservative ground truth: hybrid surfaces *other* genuinely on-topic speeches
from adjacent dates that aren't in our single-debate judged set, pushing the judged
ones down. Second, BM25 over machine-translated, salutation-heavy speeches is noisy.
Net: hit@10 is identical (0.80) and recall@20 is marginally better, so hybrid mainly
broadens coverage of exact-term queries without clearly beating dense retrieval on
this corpus. A tuned fusion weight (favouring the dense side) is the obvious next
step — but tuning on 15 questions would just overfit, so it's left as future work.

## Run it

```bash
python eval/run_eval.py                    # vector, k = 20
python eval/run_eval.py --method hybrid     # hybrid (BM25 + vector)
python eval/run_eval.py --k 30
```

Loads BGE-m3 and the full index, so it takes a couple of minutes (longer on a
weak laptop). Deterministic — the same corpus and golden set give the same numbers.
