# CLAUDE.md — Ask Parliament

## What this is

A production-shaped RAG system over the full **ParlaMint 5.0** corpus — 3,399,888 parliamentary
speeches across 29 European parliaments (1996–2024). Two goals, in order: (1) the author learns
production RAG by building every stage explicitly; (2) portfolio piece. Bar: "engineer who
understands every stage", not "framework showcase".

**Current state:** end to end and runnable — corpus pipeline (download → parse → embed) → Qdrant
index → retrieval (plain vector + agentic) → grounded cited generation → FastAPI backend + thin
Streamlit frontend → Docker compose → multi-country retrieval eval.

## Hard rules

- **No LangChain/LlamaIndex.** Direct API calls, thin self-written orchestration.
- Simple, explicit code over heavy abstraction; the author must understand every file.
- Embedding model is **BAAI/bge-m3** (1024-d) via sentence-transformers; reranker is its sibling
  **bge-reranker-v2-m3**. Models run on GPU via `config.resolve_device()`.
- Generation via the Anthropic API; models in config (`claude-haiku-4-5` default for iteration,
  `claude-sonnet-4-6` for quality). `ANTHROPIC_API_KEY` from `.env`, never hardcoded.
- Vector search uses **cosine** over L2-normalised vectors.
- Never commit: data, vector stores, `.env` (all gitignored).
- Work phase by phase; stop at each phase gate, summarize, wait for go-ahead.

## Architecture decisions

| Decision | Choice | Why |
|---|---|---|
| Vector store | Qdrant — embedded for subsets, **server** (`QDRANT_URL`) for the full ~3.4M | HNSW, on-disk vectors, bounded RAM, native payload filtering |
| Embeddings | BAAI/bge-m3 (1024-d, multilingual) | one cross-lingual space; English queries hit native-language speeches |
| Retrieval unit | speech-level | clean citations |
| Agentic retrieval | query transform (multi-query + HyDE) → RRF → cross-encoder rerank → CRAG self-correct | recall from transforms, precision from reranking, robustness from a gated correction loop; thin self-written orchestration |
| Reranker | bge-reranker-v2-m3 (cross-encoder, GPU) | joint query–speech scoring; multilingual, pairs with bge-m3 |
| Generation | Anthropic Claude (Haiku default, Sonnet switchable) | cheap iteration, quality on demand |
| Serving | FastAPI backend + thin Streamlit frontend + Qdrant, via `docker compose` | backend owns the models so the UI is a model-free container; tiers deploy/scale independently |
| Eval | synthetic known-item retrieval (plain vs agentic) | no hand-labeling; covers all 29 countries; fair relative measure |

## Corpus & pipeline

**Source:** ParlaMint 5.0 on CLARIN.SI (handle `11356/2004`), CC BY 4.0, direct download. We use the
**native** plain text (bge-m3 is cross-lingual) + the **English** TSV metadata (uniform
`Speaker_role` + per-speech CAP `Topic`). Selection: Regular speakers, text ≥ 300 chars.

**Pipeline** (`scripts/scale/`, each stage resumable and independent):
```
download_parlamint.py → parse_parlamint.py → embed_corpus.py → build_qdrant_index.py
   raw/{CC}/…txt/         parsed/{CC}.parquet   embeddings/{CC}/*.npy   Qdrant "speeches"
```
Index notes: cosine, on-disk vectors, payload indexes on `country/year/cap_topic/party`. The build
**defers HNSW indexing** during bulk load (`indexing_threshold=0`, re-enabled after) so upserts
don't slow down / time out as the collection grows. It writes `data/parlamint/facets.json` so the
app populates filters without sweeping millions of payloads. Point ids are `uuid5(speech_id)` →
idempotent reruns. Per-country resume via a payload-filtered count.

## Module map (`src/ask_parliament/`)

- `config.py` — paths, model names, knobs, `resolve_device()`.
- `models.py` — `RetrievedSpeech`, `Facets` dataclasses (import-light; no heavy deps).
- `parlamint.py` — ParlaMint 5.0 parser (native text + English metadata → tidy DataFrame).
- `retrieval.py` — `Retriever`: Qdrant vector search, metadata filters, facets from the sidecar.
- `query_transform.py` — multi-query + HyDE + corrective rewrite (one Haiku call each; best-effort).
- `rerank.py` — `Reranker`: bge-reranker-v2-m3 cross-encoder, scores → sigmoid → [0,1].
- `agentic.py` — `AgenticRetriever`: transform → RRF-fuse → rerank → self-correct if top score <
  `RERANK_SCORE_FLOOR` (≤ `AGENTIC_MAX_ROUNDS`). `last_trace` exposes what happened.
- `generation.py` — `generate_answer()`: numbered context + question → Claude, `[n]` citations,
  one retry on transient errors, clear messages on auth/400.
- `api.py` — FastAPI backend (`/health`, `/meta`, `/ask`); builds + warms the retrievers at startup.

## How to run

```bash
pip install -e .                                 # deps from pyproject.toml; ANTHROPIC_API_KEY in .env
# Build the corpus once (see scripts/scale/README.md): download → parse → embed (GPU).

# Index + serve
docker run -d --name qdrant -p 6333:6333 -v "$PWD/qdrant_storage:/qdrant/storage" qdrant/qdrant
export QDRANT_URL=http://localhost:6333          # PowerShell: $env:QDRANT_URL="http://localhost:6333"
python scripts/scale/build_qdrant_index.py       # load shards (resumable)

uvicorn ask_parliament.api:app --host 0.0.0.0 --port 8000   # backend (models; wants GPU)
streamlit run app.py                                        # frontend (thin client; API_URL→:8000)
# …or the whole stack:  docker compose up --build

# CLIs (direct to Qdrant) — add --agentic for transform+rerank+self-correct
python scripts/search.py "renewable energy" --country GR --year-from 2015
python scripts/ask.py "What did MPs say about energy prices?" --agentic

# Eval — synthetic known-item retrieval, plain vs agentic, across all countries
python eval/run_eval.py --per-country 2
```
