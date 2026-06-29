# Corpus pipeline — full ParlaMint 5.0

Build the index for the **full ParlaMint 5.0** corpus (29 parliaments, ~8M
utterances, ~1.2B words). Four stages, each resumable and independent:

```
download_parlamint.py → parse_parlamint.py → embed_corpus.py → build_qdrant_index.py
   (network, CPU)         (CPU, laptop-ok)     (GPU batch job)    (CPU; needs Qdrant)
 raw/{CC}/…txt/        parsed/{CC}.parquet   embeddings/{CC}/    Qdrant "speeches"
                                              shard_*.npy         + facets.json
```

Data source: **ParlaMint 5.0** on CLARIN.SI (handle `11356/2004`), **CC BY 4.0**,
direct download, no login. We use the **native** plain text (BGE-m3 is multilingual
and cross-lingual, so English queries still hit native speeches) and the **English
TSV metadata** (uniform `Speaker_role` + per-speech CAP `Topic`). Embeddings are
computed with `BAAI/bge-m3`, so the index is one uniform cross-lingual cosine space.

## Setup

```bash
pip install -e .                         # package + deps (incl. pyarrow) from pyproject.toml
# The embedding step needs a CUDA torch build matching your driver, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Run order

```bash
# 1. Download + extract (resumable; skips finished countries; deletes .tgz after).
python scripts/scale/download_parlamint.py --list           # sanity-check URLs
python scripts/scale/download_parlamint.py                  # all 29 (~6 GB of .tgz)
#    or a subset:  --countries AT,HR,RS,SI

# 2. Parse to tidy parquet (CPU, fast; safe on a laptop).
python scripts/scale/parse_parlamint.py                     # all downloaded countries

# 3. Embed on the GPU (resumable, shard by shard).
python scripts/scale/embed_corpus.py                        # all parsed countries

# 4. Load the shards into a Qdrant index (resumable, country by country).
#    Embedded local Qdrant only suits small subsets — for the full ~3.4M-vector
#    corpus run a Qdrant server and point at it (HNSW, fast, bounded RAM):
docker run -d --name qdrant -p 6333:6333 -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant
$env:QDRANT_URL = "http://localhost:6333"                   # PowerShell; export on bash
python scripts/scale/build_qdrant_index.py                  # all embedded countries
#    or a subset / smoke test:  --countries LV
```

Then serve it (same `QDRANT_URL` env) — see the main [README](../../README.md) for the
FastAPI backend + Streamlit frontend, or `docker compose up`. The CLIs hit Qdrant directly:

```bash
python scripts/search.py "renewable energy" --country GR --year-from 2015
python scripts/ask.py "What did MPs say about energy prices?" --agentic
```

## Optional: English view of cited speeches (ParlaMint-en)

The UI can show each cited speech in English. The text is **ParlaMint's own machine
translation** (ParlaMint-en.ana 5.0, handle `11356/2006`, EasyNMT/OPUS-MT) — not generated
by us — joined onto the indexed speeches by their native utterance id. Vectors are **never
re-computed**; English is a display-only payload field, so this just patches the existing
index. Three resumable stages:

```bash
# A. Download the English edition + extract its derived plain-text subtree (big .tgz per
#    country; only the `ParlaMint-{CC}-en.txt/` tree is kept, then the archive is deleted).
python scripts/scale/download_parlamint_en.py               # all 29  (--countries LV for one)

# B. Join to the indexed speeches -> parsed/{CC}_en.parquet (id, text_en). No Qdrant needed.
python scripts/scale/add_english_text.py                    # reports per-country coverage (~100%)

# C. Patch text_en onto the existing Qdrant points (payload-only; no re-embed, no rebuild).
python scripts/scale/patch_english_payload.py               # needs QDRANT_URL up
```

A fresh `build_qdrant_index.py` run also merges `{CC}_en.parquet` automatically, so a
from-scratch rebuild carries the English text too. If the edition isn't ingested, the
toggle simply falls back to the original-language text.

## Resumability & parallelism

- **Resume:** every stage skips work already done — a country with an `.extracted`
  marker, an existing `{CC}.parquet`, or a complete shard `.npy`. Kill and rerun
  any stage freely; pass `--force` to redo.
- **Parallel across countries:** stages are per-country, so just run more than one
  process on disjoint `--countries`.
- **Parallel across GPUs (one country):** shards are independent. Split them with
  `--workers N --worker-id K` (process *K* takes shards where `i % N == K`); pin
  each to a card with `CUDA_VISIBLE_DEVICES`:
  ```bash
  CUDA_VISIBLE_DEVICES=0 python scripts/scale/embed_corpus.py --workers 2 --worker-id 0 &
  CUDA_VISIBLE_DEVICES=1 python scripts/scale/embed_corpus.py --workers 2 --worker-id 1 &
  ```

## Knobs (defaults in `src/ask_parliament/config.py`)

| Flag | Default | Notes |
|---|---|---|
| `--batch-size` | 64 | raise on a big GPU; lower if you hit OOM |
| `--shard-size` | 50000 | speeches per `.npy`; the resume/parallel unit |
| `--max-seq-length` | 1024 | truncates long speeches for throughput; raise (≤8192) for fidelity |
| `--device` | auto | `cuda` / `cuda:0` / `cpu` |
| `--dry-run` | off | fake encoder, no torch/model — validates the pipeline without a GPU |

## Output layout (all gitignored under `data/`)

```
data/parlamint/
  raw/{CC}/ParlaMint-{CC}.txt/<year>/...   # extracted text + meta TSV
  parsed/{CC}.parquet                      # id, text, country, date, year, speaker,
                                           #   party, party_status, cap_topic, ...
  embeddings/{CC}/shard_00000.npy          # float16, L2-normalized, dim 1024
  embeddings/{CC}/manifest.json            # rows, shard_size, model, dtype, ...
  facets.json                              # filter values for the app (written by step 4)
```

Vectors are float16 and unit-norm: ~2 GB per million speeches per language, so the
full corpus is on the order of tens of GB. The index step (`build_qdrant_index.py`)
loads these shards into **Qdrant**; at full scale use a **Qdrant server** (embedded
local mode is brute-force and only suits small subsets — it warns past 20k points).
The build script and retriever pick the server automatically when `QDRANT_URL` is set,
so the same code serves a laptop subset or the full corpus.

## Validate without a GPU (what was tested on the laptop)

```bash
python scripts/scale/download_parlamint.py --countries LV   # smallest, ~51 MB
python scripts/scale/parse_parlamint.py --countries LV
python scripts/scale/embed_corpus.py --countries LV --dry-run --shard-size 20000
python scripts/scale/embed_corpus.py --countries LV --dry-run   # reruns → all skipped
```

## License

ParlaMint 5.0 is CC BY 4.0 — cite the corpus if you publish or deploy. Verify terms
before any public, non-attributed redistribution.
