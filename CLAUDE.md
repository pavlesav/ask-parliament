# CLAUDE.md — Ask Parliament

## What this is

A small production-shaped RAG demo over the ParlaMint parliamentary-speech corpus from the
master's thesis at `../master-thesis`. Two goals, in order: (1) the author learns production
RAG by building every stage explicitly; (2) portfolio piece. Bar: "junior engineer who
understands every stage", not "framework showcase".

## Hard rules

- **No LangChain/LlamaIndex.** Direct API calls, thin self-written orchestration.
- Simple, explicit code over heavy abstraction; the author must understand every file.
- Embedding model is **BAAI/bge-m3** via sentence-transformers and must match the thesis
  vectors (1024-d, float32). Reuse precomputed vectors; never re-embed what already exists.
- Generation via Anthropic API; model in config (`claude-haiku-4-5` default for iteration,
  `claude-sonnet-4-6` for quality). `ANTHROPIC_API_KEY` from `.env`, never hardcoded.
- Never commit: data, vector stores, `.env` (gitignored from first commit).
- **Never modify `../master-thesis`** (public research artifact, journal submission) —
  suggest changes in DATA_MAP.md §11 instead.
- Vector search must use **cosine** distance (some stored vectors are chunk-averaged and
  not unit-norm).
- Work phase by phase; stop at each phase gate, summarize, wait for go-ahead.

## Data

Source corpus inventory: see `DATA_MAP.md` (Phase 0 deliverable, verified 2026-06-12).
The short version:

- Canonical inputs: `../master-thesis/code/data folder/{CC}/{CC}_final.pkl`
  (CC ∈ AT, GB, HR; note the space in `data folder`).
- Speech-level BGE-m3 vectors exist for **AT and HR only** (English-MT + native);
  GB has segment-level only.
- Retrieval unit: speech rows, key `ID`; metadata: speaker, party, `Party_status`
  (coalition/opposition; unusable for GB), date/year, CAP labels (4 sources — see
  DATA_MAP §6), segment IDs.
- ~50% of AT/HR rows are Chairperson procedural turns — filter for the index.

## Architecture decisions

| Decision | Choice | Why |
|---|---|---|
| Vector store | ChromaDB, persistent local dir (gitignored) | precomputed-embedding ingest + metadata filtering, zero infra |
| Unit | speech-level | clean citations; segments too long (avg 9–18 speeches) |
| UI | Streamlit chat | author knows it well |
| Eval | golden set, recall@k + MRR | explainable, no LLM-judge dependency |

## Phase status

- [x] **Phase 0 — Inventory**: `DATA_MAP.md` written; embeddings/granularity verified.
- [ ] **Phase 1 — Corpus selection + ingestion** (`scripts/build_index.py`): awaiting
  corpus-subset decision at phase gate (options in DATA_MAP §9–10).
- [ ] Phase 2 — Retrieval module + CLI (`src/ask_parliament/retrieval.py`, `scripts/search.py`)
- [ ] Phase 3 — Grounded generation (`src/ask_parliament/generation.py`)
- [ ] Phase 4 — Streamlit app (`app.py`)
- [ ] Phase 5 — Evaluation (`eval/golden_set.jsonl`, `eval/run_eval.py`)
- [ ] Stretch: hybrid BM25+vector, Docker, FastAPI split

## How to run

Nothing runnable yet (Phase 0 only produced documentation).
