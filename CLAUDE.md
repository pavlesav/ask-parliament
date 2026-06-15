# CLAUDE.md — Ask Parliament

## What this is

A small production-shaped RAG demo over the ParlaMint parliamentary-speech corpus from the
master's thesis at `../master-thesis`. Two goals, in order: (1) the author learns production
RAG by building every stage explicitly; (2) portfolio piece. Bar: "junior engineer who
understands every stage", not "framework showcase".

**Current state:** feature-complete end to end — ingestion → retrieval → grounded generation
→ Streamlit chat → evaluation, plus hybrid BM25+vector retrieval and small-to-big segment
context expansion (Phases 0–5 + two stretches done; see Phase status below). Remaining
stretch: Dockerfile/compose and a FastAPI↔Streamlit split.

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
| Hybrid fusion | Reciprocal Rank Fusion (RRF) | combines by rank, no score normalization; explainable |
| Context assembly | small-to-big: retrieve speech, expand to its debate segment | conversational context for generation without losing per-speech citations; reuses the index, no re-embedding |

## Corpus (Phase 1 decision)

**Austria, full range 1996–2022**: `Speaker_role == "Regular"`, `Text_English` ≥ 300 chars
→ **99,089 speeches**, English-MT text + precomputed English BGE-m3 speech vectors.
Chairperson/Guest turns dropped; GB excluded (no speech vectors); HR deferred.
Collection `speeches_at` in `chroma_db/` (cosine), ~2.7 GB on disk, ~9 min build.
Metadata per record: `date, year, speaker, speaker_id, party, party_status, cap_domain,
topic_name, segment_id` (party code `GRÜNE` normalized to `Grüne`; native German text NOT
stored in the index — kept lean, can be joined back via `ID` later if wanted).

## Phase status

- [x] **Phase 0 — Inventory**: `DATA_MAP.md` written; embeddings/granularity verified.
- [x] **Phase 1 — Corpus selection + ingestion**: `scripts/build_index.py` built and
  verified (count match, self-retrieval distance ~1e-7, metadata filter checks).
- [x] **Phase 2 — Retrieval module + CLI**: `Retriever.search()` with metadata filters
  (country, year range, CAP domains, party); verified incl. cross-lingual German queries.
  (Observed here, addressed by the small-to-big stretch below: hits cluster in one debate
  segment — turned from a problem into the basis for segment context expansion.)
- [x] **Phase 3 — Grounded generation**: `generate_answer()` assembles numbered
  context + question, calls Claude (plain `messages.create`, no thinking/effort — works
  across Haiku/Sonnet), returns a `GenerationResult` (answer + sources + usage + latency).
  One explicit retry on transient errors (SDK `max_retries=0`); clear messages on
  auth/400. Verified: correct [n] citations, speaker attribution, and the "context
  doesn't contain the answer" guardrail (refuses to fabricate). CLI: `scripts/ask.py`.
- [x] **Phase 4 — Streamlit app** (`app.py`): chat UI with sidebar filters (country
  multiselect, year-range slider, CAP-domain multiselect, top-k, model selector),
  expandable Sources per answer, token/latency caption. Retriever cached with
  `@st.cache_resource` (BGE-m3 loads once); facets cached with `@st.cache_data`.
  Retrieval `country`→`countries` (list, `$in`) for multi-country readiness;
  `Retriever.facets()` populates filters from the index (paginated get). Verified
  end-to-end via Streamlit AppTest.
- [x] **Phase 5 — Evaluation**: `eval/golden_set.jsonl` (15 questions, each tied to a
  real debate; relevant set defined lexically + date-windowed, embedding-independent,
  stored as an auditable criterion). `eval/run_eval.py` reports recall@k, MRR, hit@10
  under two conditions (unfiltered vs year-scoped). Result: year-scoping ~doubles MRR
  (0.23→0.50) and hit@10 (0.40→0.80). `eval/README.md` + main README explain it.
  The two clean misses (minimum_wage, pension_reform) motivate the hybrid-retrieval stretch.
- [x] **Stretch — hybrid BM25 + vector** (`src/ask_parliament/hybrid.py`): RRF fusion,
  built on the fly and cached in-process; hybrid is the app default (toggle to compare
  with semantic), available via `--hybrid` on the CLIs and `--method hybrid` in eval.
  Eval finding: **roughly neutral** on this corpus (year-scoped MRR 0.42 vs vector 0.50,
  identical hit@10) — helps exact-term/recurring queries, demotes lone dense winners
  (RRF favors consensus). Reported honestly; fusion-weight tuning left as future work
  (avoid overfitting 15 questions).
- [x] **Stretch — segment context expansion (small-to-big)**
  (`Retriever.expand_to_segments`): retrieval still returns speech-level hits as the
  citation anchors; before generation each hit is expanded to its sibling speeches in the
  same debate segment (ordered by spoken sequence via the trailing TEI id, windowed to
  `max_per_segment`, siblings truncated to `sibling_char_cap`), so the model sees the
  surrounding exchange while citations stay pinned to the matched speech. `format_context`
  groups by debate with headers and flags context turns; numbering is continuous so `[n]`
  matches everywhere (prompt + Sources). Default on in the app (toggle) and `ask.py`
  (`--no-expand`). Orthogonal to the retrieval eval — it changes generation context, not
  ranking, so the recall@k/MRR numbers are unaffected.
- [ ] Stretch remaining: Docker, FastAPI split.

## How to run

```bash
pip install -r requirements.txt
pip install -e .                      # registers src/ask_parliament for script imports
python scripts/build_index.py        # idempotent; --rebuild to force; needs ~4 GB RAM
python scripts/search.py "refugee crisis" --year-from 2015 --year-to 2016 --k 5
```
`search.py` filters: `--country --year-from --year-to --domain (repeatable) --party --chars`.
First query in a process loads BGE-m3 (~5-7 s); warm queries embed in <1 s.

```bash
# Grounded Q&A end-to-end (needs ANTHROPIC_API_KEY in .env)
python scripts/ask.py "What did MPs say about the 2015 refugee crisis?" --year-from 2015 --year-to 2016
```
`ask.py` takes the same filters as `search.py`, plus `--model` (default `claude-haiku-4-5`)
and `--expand`/`--no-expand` (small-to-big debate context, on by default).

```bash
# Streamlit chat app (hybrid retrieval + "Expand to full debate" on by default; both toggle in the sidebar)
streamlit run app.py

# Retrieval evaluation (recall@k, MRR, hit@10)
python eval/run_eval.py                 # vector
python eval/run_eval.py --method hybrid  # BM25 + vector
```
Hybrid (`--hybrid` on the CLIs, default in the app) builds a BM25 index over all speeches
in-process (~1 min on a weak laptop) and fuses with vector results via RRF.
