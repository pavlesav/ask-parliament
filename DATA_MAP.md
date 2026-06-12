# DATA_MAP — Inventory of the master-thesis corpus

Produced 2026-06-12 (Phase 0 of Ask Parliament). Every claim below was verified by
loading the actual files (each pickle opened in a subprocess and inspected: shape, dtypes,
null counts, value distributions, embedding norms). Where something is inferred rather than
loaded, it says so explicitly.

---

## 1. TL;DR for the RAG build

- **Canonical files** (one per country, everything joined): `{AT,GB,HR}_final.pkl`.
  All upstream files are intermediates — the RAG pipeline only ever needs the finals.
- **Speech-level BGE-m3 vectors (1024-d float32) exist for AT and HR in both languages,
  ready to reuse. GB has NO speech-level vectors** — they were dropped during preprocessing
  to save memory; only segment-level vectors survive for GB. Including GB at speech
  granularity means re-embedding.
- **Unit of retrieval**: speech rows. `ID` is the unique per-speech key. Segments
  (`Segment_ID_English` / `Segment_ID_Native`) group 9–18 speeches into agenda episodes and
  carry the thesis CAP labels.
- **Gotchas**: embeddings of overlong texts are chunk-averaged and **not re-normalized**
  (norms slightly < 1.0) → the vector store must use **cosine** distance. ~50% of AT/HR
  rows are Chairperson procedural turns. AT/HR English text is **machine translation**.

## 2. Where the data actually lives (≠ thesis README)

The thesis README describes an idealized layout that does not exist on disk. Real layout:

```
master-thesis/
├── code/
│   ├── data_preprocessing.ipynb      # load → speech & segment embeddings → segmentation
│   ├── topic_modelling.ipynb         # BERTopic/GMM → GPT-4o-mini CAP labels → merge LIWC
│   ├── visualization.ipynb           # consumes *_final.pkl, produces figures/
│   ├── demo_emotions.ipynb           # side experiment (emotion scores on HR sample)
│   └── data folder/                  # ← ALL data, gitignored (note: space in name)
│       ├── AT/  GB/  HR/             # per-country processed + raw
│       ├── ches AT.csv               # Chapel Hill Expert Survey, AT parties (35×78)
│       └── LIWC-22.Descriptive.Statistics-Test.Kitchen.xlsx   # LIWC norms (z-scores)
├── figures/   outputs(absent)   README.md   Master's thesis.pdf
```

README discrepancies found (details in §11): different notebook names, `data/raw|processed`
paths that don't exist, cluster counts 170/180/185 vs actual 215/170/150, output filenames
(`{XX}_segments.pkl` etc.) that were never produced.

## 3. File inventory (all in `master-thesis/code/data folder/`)

| File | Size | Role |
|---|---|---|
| `AT/AT_final.pkl` | 3.25 GB | **canonical** 231,759 × 160 |
| `AT/AT_with_topics.pkl` | 3.02 GB | intermediate (final minus consensus/LIWC/derived) |
| `AT/AT_speeches_processed.pkl` | 3.02 GB | intermediate (pre-topics) |
| `AT/AT_LIWC_results.csv` | 109 MB | LIWC-22 scores, joins on `ID` (120 cols) |
| `AT/AT_for_LIWC_native_language.csv` | 464 MB | `ID` + native text (LIWC input artifact) |
| `AT/AT_english_topic_metadata.csv`, `AT_german_topic_metadata.csv` | ~35 KB | 170 topics each: `Topic_ID, Keywords, Topic_Name, CAP_Category` |
| `AT/ParlaMint-AT/`, `AT/ParlaMint-AT-en.ana/` | — | raw ParlaMint v5 (native + English MT) |
| `GB/GB_final.pkl` | 1.68 GB | **canonical** 670,912 × 155 |
| `GB/GB_with_topics.pkl` | 1.02 GB | intermediate, 670,912 × 32 (verified) |
| `GB/GB_speeches_processed.pkl` | 1.01 GB | intermediate, 670,912 × 29 (verified) |
| `GB/GB_LIWC_results.csv` | 313 MB | as above |
| `GB/GB_topic_metadata.csv` | 42 KB | 215 topics |
| `GB/ParlaCAP-test-en.jsonl` | 0.7 MB | 876 human-labelled speeches (test set) |
| `GB/ParlaMint-GB/` | — | raw (native English only) |
| `HR/HR_final.pkl` | 6.21 GB | **canonical** 504,338 × 161 |
| `HR/HR_with_topics.pkl` | 5.72 GB | intermediate |
| `HR/HR_speeches_processed.pkl` | 5.70 GB | intermediate |
| `HR/HR_LIWC_results.csv` | 218 MB | as above |
| `HR/HR_for_LIWC_native_language.csv` | 577 MB | as above |
| `HR/HR_croatian_topic_metadata.csv`, `HR_english_topic_metadata.csv` | ~32 KB | 150 topics each |
| `HR/ParlaCAP-test-hr.jsonl` | 1.2 MB | 869 human-labelled speeches |
| `HR/HR_emotions_smoke.parquet` | 44 KB | 500-row emotion-scores smoke test (`j-hartmann/emotion-english-distilroberta-base`), from `demo_emotions.ipynb` |
| `HR/ParlaMint-HR/`, `HR/ParlaMint-HR-en.ana/` | — | raw |

Lineage: `data_preprocessing.ipynb` → `*_speeches_processed.pkl` → `topic_modelling.ipynb`
→ `*_with_topics.pkl` → (+ consensus + ParlaCAP labels + LIWC merge) → `*_final.pkl`.
The `checkpoints/` and `processed/` dirs the preprocessing notebook writes to no longer
exist; outputs were moved into the per-country folders.

## 4. Canonical `*_final.pkl` schema

Column blocks (AT 160 / HR 161 / GB 155 columns — all counts cross-check exactly):

**ParlaMint metadata (24 cols, all countries)** — from the raw `-meta.tsv` files:
`Text_ID, ID, Title, Date, Body, Term, Session, Meeting, Sitting, Agenda, Subcorpus, Lang,
Speaker_role, Speaker_MP, Speaker_minister, Speaker_party, Speaker_party_name, Party_status,
Party_orientation, Speaker_ID, Speaker_name, Speaker_gender, Speaker_birth, Topic`

**Text (2)**: `Text_English` (GB native; AT/HR machine translation), `Text_Native`
(German/Croatian; `None` for GB).

**Segments (2)**: `Segment_ID_English`, `Segment_ID_Native` (GB: native all-null).
Format `{Text_ID}_seg_{n}`.

**Embeddings (object cols of `np.ndarray(1024,) float32`)**:

| Column | AT | HR | GB |
|---|---|---|---|
| `Speech_Embeddings_English` | ✅ all rows | ✅ all rows | ❌ absent |
| `Speech_Embeddings_Native` | ✅ all rows | ✅ all rows | ❌ absent |
| `Segment_Embeddings_English` | ✅ | ✅ | ✅ |
| `Segment_Embeddings_Native` | ✅ | ✅ | ❌ absent |

Segment embeddings are the **same ndarray object shared across all rows of a segment**
(verified by identity check) — dedupe before indexing at segment granularity.
Norms: mostly 1.0; chunk-averaged ones (texts > 8,192 BGE-m3 tokens) are < 1.0
(e.g. 0.974 observed) → **use cosine similarity, never raw dot product**.

**Thesis topic labels** (per language; GB English only): `Topic_Keywords_{CC}_{lang}`,
`Topic_Name_{CC}_{lang}`, `CAP_Category_{CC}_{lang}` — episode-level labels mapped onto
every speech row of the segment.

**`topic_consensus`** (all): agreement rule between the two languages (strict for
chairpersons). GB's equals `CAP_Category_GB_english` by construction (verified).
Caveat: conservative — AT is 40% `Other` + 21% `Mix`; HR 36% + 20%.

**`True_label`** (GB and HR only, NOT AT): human CAP label from the ParlaCAP test sets
(GB 876, HR 869 rows non-null). 22 classes incl. `Public Lands`.

**LIWC-22 (118 cols, all countries)**: `WC, Analytic, Clout, Authentic, Tone, WPS, …, Emoji`
— raw LIWC percentages scored on English text. Full coverage (inner join lost 0 rows).

**Derived (3)**: `Country` ("Austria"/"Croatia"/"Great Britain"), `Year` (int),
`Speaker_age` (Year − Speaker_birth; all-NaN for GB).

Dtypes note: `Date` is `datetime64[ns]` **only in `*_final.pkl`** (string in intermediates).

## 5. Key facts per country (verified)

| | AT | HR | GB |
|---|---|---|---|
| Speeches | 231,759 | 504,338 | 670,912 |
| Date range | 1996-01-15 → 2022-10-12 | 2003-12-22 → 2022-07-15 | 2015-01-05 → 2022-07-21 |
| Body | Lower house (Nationalrat) only | Unicameral (Sabor) | Commons 472,856 + **Lords 198,056** |
| Sessions (`Text_ID`) | 1,221 | 1,708 | 2,209 |
| Segments EN / native | 25,568 / 23,752 | 32,749 / 39,889 | 37,605 / — |
| Avg speeches per EN segment | 9.1 | 15.4 | 17.8 |
| `Speaker_role` | Chair 125,042 / Regular 106,277 / Guest 440 | Chair 246,585 / Regular 257,753 | Chair 16,345 / Regular 654,567 |
| `Party_status` | Coalition 77,431 / Opposition 43,073 / "-" 111,255 | Coalition 277,802 / Opposition 150,897 / "-" 75,639 | mostly "-" (490,933); Opposition 162,760 / Coalition 17,219 — **not usable for GB** |
| Main parties | SPÖ, ÖVP, FPÖ, Grüne, NEOS, BZÖ… | 45 parties (HDZ, SDP…) | 50 parties (CON, LAB…) |
| `Speaker_birth` | 99.98% filled | 94.6% filled | **all null** (→ `Speaker_age` null) |
| Human `True_label` | — | 869 rows | 876 rows |
| Text_English median/p90 chars (all rows) | 320 / 4,945 | 246 / 3,509 | 464 / 2,987 |
| Regular-only median/p90 | 2,868 / 6,744 | 1,044 / 6,054 | 467 / 3,050 |

Other quirks worth knowing:
- `Subcorpus` ∈ {Reference, COVID, "COVID,War"} — free era filter (COVID ≈ 2020+, War ≈ 2022).
- `Lang` is always "English" (artifact of loading from the `-en.ana` corpora) — useless.
- AT `Speaker_party` has a case-duplicate: `Grüne` (16,448) vs `GRÜNE` (279) — normalize at ingest.
- HR has 7,201 rows (GB 2,996) with missing speaker metadata ("-"); HR `Topic` null on those.
- `ID` formats: GB/HR `ParlaMint-XX_<date>…​.u<N>`; AT `ParlaMint-AT_<date>…_<hash>`.
  `Text_ID` for AT/HR carries an `-en` infix (`ParlaMint-HR-en_2015-07-02-0`) because the
  English corpora were the load base; segment IDs inherit it.
- `Session`/`Agenda` are always "-"; `Term`/`Meeting`/`Sitting` formats differ per country.

## 6. Four CAP-label sources (don't mix them up)

| Column | Granularity | Source | Coverage |
|---|---|---|---|
| `CAP_Category_{CC}_{lang}` | segment (episode) | thesis: c-TF-IDF keywords → GPT-4o-mini | all rows |
| `topic_consensus` | segment | cross-language agreement of the above | all rows; Other/Mix-heavy for AT/HR |
| `Topic` (ParlaMint meta) | **per speech** | ParlaCAP automatic classifier (ships with ParlaMint v5) | ~99% rows; very Other-heavy (AT 52%, HR 53%) |
| `True_label` | per speech | human annotation (ParlaCAP test sets) | GB 876 / HR 869 rows only |

For sidebar filtering, `CAP_Category_{CC}_english` is the richest; `Topic` and `True_label`
are independent signals → good for building/validating the eval golden set.

## 7. Sidecar files

- **`{…}_topic_metadata.csv`** (5 files): one row per discovered topic
  (`Topic_ID, Keywords, Topic_Name, CAP_Category`). Note: `Topic_ID` is NOT a column in the
  final pickles — to join you'd have to match on `Topic_Keywords`/`Topic_Name` strings.
- **`ParlaCAP-test-{en,hr}.jsonl`**: `id, lang, text_id, text, speaker_role, labels` —
  already merged into finals as `True_label`.
- **`{…}_LIWC_results.csv`**: `ID, Segment, WC, …` (120 cols) — already merged
  (`Segment` col dropped).
- **`ches AT.csv`**: CHES expert survey for Austrian parties (35×78: `party, year, lrgen,
  eu_position, …`) — unused by the pipeline; potential party-metadata enrichment.
- **`LIWC-22…Test.Kitchen.xlsx`**: LIWC norm means/stds used by `visualization.ipynb`
  for z-scoring.

## 8. Raw ParlaMint v5 layout (for provenance only)

`ParlaMint-XX[-en.ana]/ParlaMint-XX[-en].txt/<year>/ParlaMint-XX_<date>-<body>[-meta].tsv|.txt`
— year folders of per-sitting text + metadata pairs. The pipeline never needs them again
(the finals contain everything); listed here for provenance.

## 9. Candidate demo subsets (exact counts)

"Regular" = `Speaker_role == "Regular"` (drops procedural chair turns);
length filter on `Text_English` characters.

| Subset | all rows | Regular | Reg ≥300 chars | Reg ≥500 chars |
|---|---|---|---|---|
| AT all years (1996–2022) | 231,759 | 106,277 | **99,089** | 93,688 |
| AT 2015–2022 | 64,522 | 29,131 | **27,909** | 26,779 |
| AT 2017–2022 | 46,368 | 21,038 | 20,070 | 19,188 |
| HR 2015–2022 | 266,896 | 136,778 | 119,210 | 110,209 |
| HR 2017–2022 | 226,379 | 116,229 | 100,955 | 92,902 |
| GB 2015–2022 (= all) | 670,912 | 654,567 | 477,367 | 302,227 |

AT Regular speeches are evenly spread (~2.7–5.3k/year, no gaps). GB at speech granularity
requires re-embedding (no speech vectors). HR is large even filtered.

## 10. Open questions for Phase 1 (decided at the phase gate)

1. Corpus subset — options + recommendation in the phase summary.
2. Index vectors: English-MT speech embeddings (uniform, queries in English) vs native.
   Native text can ride along as displayed metadata either way.
3. Keep or drop Chairperson turns (recommend drop).
4. Which CAP column drives the sidebar filter (recommend `CAP_Category_{CC}_english`).

## 11. Suggested hygiene improvements for master-thesis (NOT applied — repo untouched)

1. README "Repository Structure" and "Reproducing Results" sections describe notebooks
   (`01_segmentation.ipynb`…) and outputs (`{XX}_segments.pkl`, `{XX}_embeddings.pkl`…)
   that don't exist; actual names differ (§2–3).
2. README cluster counts (AT 180, GB 170, HR 185) don't match the executed notebook
   (AT 170, GB 215, HR 150 — silhouette-chosen on 2025-12-02). Worth reconciling with
   what the thesis PDF reports, especially with a journal submission attached.
3. README claims segmentation/embedding outputs live in `data/processed/` — real path is
   `code/data folder/{CC}/`; `requirements.txt` (155 bytes) is far thinner than the pip
   list in the README.
4. `*_speeches_processed.pkl` and `*_with_topics.pkl` (~19 GB combined) are fully
   superseded by `*_final.pkl` — archivable to free disk if the finals are backed up.
5. `Master's thesis.pdf` filename contains an apostrophe — breaks naive shell quoting;
   consider `masters_thesis.pdf`.
6. The citation block and license placeholder in README are still unfilled.
