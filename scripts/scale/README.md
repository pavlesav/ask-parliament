# Scaling pipeline — full ParlaMint 5.0

Take Ask Parliament from the Austria-only demo to the **full ParlaMint 5.0**
corpus (29 parliaments, ~8M speeches, ~1.2B words). Three stages, each resumable
and independent:

```
download_parlamint.py   →   parse_parlamint.py   →   embed_corpus.py
   (network, CPU)              (CPU, laptop-ok)         (GPU batch job)
 raw/{CC}/…txt/            parsed/{CC}.parquet      embeddings/{CC}/shard_*.npy
```

Data source: **ParlaMint 5.0** on CLARIN.SI (handle `11356/2004`), **CC BY 4.0**,
direct download, no login. We use the **native** plain text (BGE-m3 is multilingual
and cross-lingual, so English queries still hit native speeches) and the **English
TSV metadata** (uniform `Speaker_role` + per-speech CAP `Topic`). Embeddings are
**recomputed from scratch** with `BAAI/bge-m3`, so this index is one uniform cosine
space, independent of the thesis vectors.

## Setup

```bash
pip install -r requirements.txt          # base app deps
pip install -r requirements-scale.txt    # + requests, pyarrow, CUDA torch (see file)
pip install -e .                         # registers ask_parliament for the scripts
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
```

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
```

Vectors are float16 and unit-norm: ~2 GB per million speeches per language, so the
full corpus is on the order of tens of GB. At this scale prefer **Qdrant or FAISS**
over local Chroma for the index step (next phase).

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
