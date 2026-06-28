# Ask Parliament

A retrieval-augmented chat app over **European parliamentary speeches** — the full
[ParlaMint 5.0](https://www.clarin.si/repository/xmlui/handle/11356/2004) corpus: **3,399,888
speeches across 29 parliaments (1996–2024)**. Ask a question in natural language —
*"How did parliaments respond to rising energy prices?"* — and get an answer grounded **only**
in retrieved speeches, with citations (speaker, party, country, date), across languages.

Built stage by stage as a production-shaped RAG system, with **no RAG frameworks** (no
LangChain/LlamaIndex) — every component is written directly and is explainable.

**Highlights**
- **Cross-lingual.** Ask in English; retrieve speeches in their **original language** (BGE-m3 is
  multilingual); answers come back in English.
- **Agentic retrieval.** Query transformation (paraphrases + HyDE) → reciprocal-rank fusion →
  cross-encoder **reranking** → a **self-correcting** re-retrieval loop (Corrective-RAG).
- **Grounded generation.** Claude answers from the retrieved speeches only, with `[n]` citations,
  and says so when the context doesn't support an answer.
- **Service-split + containerised.** A FastAPI backend (models + retrieval + generation), a thin
  Streamlit frontend, and Qdrant — wired together with `docker compose`.

## How it works

```
ParlaMint .tgz ─► parse ─► BGE-m3 embed ─► Qdrant index
                                              │
   question ─►  Retriever (vector)  ──or──  AgenticRetriever (transform→rerank→self-correct)
                                              │
                                     grounded generation (Claude, cited)  ─►  answer
```

1. **Ingest** (`scripts/scale/`) — download each country's ParlaMint corpus, parse the native
   plain text + English metadata into one tidy parquet per country (Regular speakers, ≥300 chars),
   and embed with **BAAI/bge-m3** (1024-d, L2-normalised) into on-disk shards.
2. **Index** (`scripts/scale/build_qdrant_index.py`) — load the shards into a **Qdrant** `speeches`
   collection (cosine, HNSW, on-disk vectors, payload indexes for filtering). Writes a small
   `facets.json` so the UI never sweeps millions of payloads.
3. **Retrieve** (`src/ask_parliament/retrieval.py`, `agentic.py`) — embed the query and run top-k
   cosine search with optional filters (country, year, CAP domain, party). The **agentic** mode
   adds query transformation, [BGE-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
   reranking, and a correction loop that re-retrieves when the top results look weak.
4. **Generate** (`generation.py`) — assemble the numbered speeches + question and ask Claude to
   answer **only** from them, with `[n]` citations and an "insufficient evidence" guardrail.
5. **Serve** — a **FastAPI** backend (`api.py`: `/health`, `/meta`, `/ask`) owns the models; the
   **Streamlit** frontend (`app.py`) is a thin HTTP client; **Qdrant** holds the vectors.

## Run it

```bash
pip install -e .                 # installs the package and its dependencies (pyproject.toml)
# Put ANTHROPIC_API_KEY in .env (gitignored)
```

### Build the corpus (one-time)

```bash
# The GPU embedding step needs a CUDA torch build matching your driver, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu128

python scripts/scale/download_parlamint.py     # ~6 GB of .tgz from CLARIN.SI (or --countries LV)
python scripts/scale/parse_parlamint.py        # → data/parlamint/parsed/{CC}.parquet
python scripts/scale/embed_corpus.py           # GPU → data/parlamint/embeddings/{CC}/*.npy
```

See [scripts/scale/README.md](scripts/scale/README.md) for the resumable/parallel pipeline details.

### Index + serve

```bash
# Qdrant (a server is needed at full scale: HNSW, on-disk vectors, bounded RAM)
docker run -d --name qdrant -p 6333:6333 -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant
export QDRANT_URL=http://localhost:6333        # PowerShell: $env:QDRANT_URL="http://localhost:6333"
python scripts/scale/build_qdrant_index.py     # load the shards (resumable)

uvicorn ask_parliament.api:app --host 0.0.0.0 --port 8000   # API backend (wants a GPU)
streamlit run app.py                                        # UI at http://localhost:8501
```

### Or the whole stack in Docker

```bash
docker compose up --build      # qdrant + api + web → UI at http://localhost:8501
```

Prereqs: a populated `./qdrant_storage` (run the index build against the `qdrant` service first),
`./data/parlamint/facets.json`, and `ANTHROPIC_API_KEY` in `.env`. The `api` service requests an
NVIDIA GPU via the Container Toolkit so the reranker is fast — remove the `deploy` block in
`docker-compose.yml` to run CPU-only.

### CLIs (talk to Qdrant directly)

```bash
python scripts/search.py "renewable energy" --country GR --year-from 2015   # retrieval only
python scripts/ask.py "What did MPs say about energy prices?" --agentic      # retrieval + answer
```

## Evaluation

Retrieval is scored across all 29 parliaments with **synthetic known-item retrieval**: sample
speeches corpus-wide, have an LLM write a question each speech answers, then check whether (and at
what rank) the source speech comes back — comparing **plain** vector search against the **agentic**
pipeline. Full method and caveats in [eval/README.md](eval/README.md).

```bash
python eval/run_eval.py --per-country 2          # MRR, hit@1/5/10 for plain vs agentic
```

The pattern: **agentic improves ranking** — the cross-encoder reranker lifts the right speech
toward rank 1 (higher MRR/hit@1) even when plain search already had it in the top *k*. The
self-correction loop rarely fires, because dense retrieval over 3.4M multilingual speeches is
already strong — reported honestly rather than assumed.

## Architecture choices

| Decision | Choice | Why |
|---|---|---|
| Vector store | Qdrant server (embedded for subsets) | scales to 3.4M: HNSW, on-disk vectors, bounded RAM, native payload filtering |
| Embeddings | BAAI/bge-m3 (1024-d, multilingual) | one cross-lingual space; English queries retrieve native-language speeches |
| Distance | cosine over L2-normalised vectors | standard for sentence embeddings |
| Retrieval unit | speech-level | clean citations |
| Agentic retrieval | query transform (multi-query + HyDE) → RRF → cross-encoder rerank → CRAG self-correct | recall from transforms, precision from reranking, robustness from a gated correction loop |
| Reranker | BAAI/bge-reranker-v2-m3 (cross-encoder, GPU) | joint query–speech scoring; multilingual, pairs with bge-m3 |
| Generation | Anthropic Claude (Haiku default, Sonnet switchable) | cheap iteration, quality on demand |
| Serving | FastAPI backend + thin Streamlit frontend + Qdrant, via `docker compose` | backend owns the models so the UI is a model-free container; tiers scale independently |
| Evaluation | synthetic known-item retrieval (plain vs agentic) | no hand-labeling; covers all 29 countries; fair relative measure |

## Repository layout

```
ask-parliament/
├── app.py                      # Streamlit frontend — thin HTTP client over the API
├── docker-compose.yml          # qdrant + api + web
├── Dockerfile.api / .web       # backend (torch/CUDA + models) / frontend (light)
├── scripts/
│   ├── search.py / ask.py      # retrieval-only / retrieval+answer CLIs
│   └── scale/                  # corpus pipeline: download → parse → embed → build_qdrant_index
├── src/ask_parliament/
│   ├── config.py               # paths, model names, knobs
│   ├── models.py               # shared dataclasses (RetrievedSpeech, Facets) — import-light
│   ├── parlamint.py            # ParlaMint 5.0 parser (native text + English metadata)
│   ├── retrieval.py            # Qdrant vector search + metadata filters + facets
│   ├── query_transform.py      # multi-query + HyDE + corrective rewrite (Claude)
│   ├── rerank.py               # BGE-reranker-v2-m3 cross-encoder
│   ├── agentic.py              # AgenticRetriever: transform → fuse → rerank → self-correct
│   ├── generation.py           # grounded, cited answer generation (Claude)
│   └── api.py                  # FastAPI backend: /health, /meta, /ask
├── eval/                       # synthetic known-item retrieval eval (run_eval.py + README)
└── CLAUDE.md                   # architecture decisions & build notes
```

## Limitations

- **No BM25 hybrid / debate-context expansion.** Dense + reranking only. In-memory BM25 over 3.4M
  speeches doesn't fit (Qdrant sparse vectors are the way to add it), and the native ParlaMint
  distribution exposes no debate-segment id (only a session id) to expand into.
- **Native-language sources.** Retrieved speeches are shown in their original language; the model
  reads them and answers in English, quoting substance not exact wording.
- **Noisy policy labels.** CAP domains are episode-level and automatically assigned, so the domain
  filter is optional and off by default — semantic search finds on-topic speeches without it.
- **Eval is a relative measure.** Known-item retrieval is optimistic by construction (see
  [eval/README.md](eval/README.md)); it compares methods fairly but isn't an absolute recall figure.

## License

Code: MIT. Data: ParlaMint 5.0 is CC BY 4.0 — cite the corpus if you publish or deploy.
