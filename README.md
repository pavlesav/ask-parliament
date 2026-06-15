# Ask Parliament

A retrieval-augmented chat app over ~1.4M European parliamentary speeches
(ParlaMint v5.0: Austria, Croatia, UK). Ask a question in natural language —
*"What did Austrian MPs say about immigration during the 2015 crisis?"* — and get an
answer grounded **only** in retrieved speeches, with citations (speaker, party, date,
country, policy domain) and metadata filters.

Built on the corpus, BGE-m3 embeddings, and CAP policy-domain labels from my
[master's thesis pipeline](https://github.com/pavlesav/master-thesis): the thesis
produced the data; this repo turns it into a production-shaped RAG system built
stage by stage — no RAG frameworks, every component explicit.

## Status

🚧 In progress — **working end to end**: a Streamlit chat app answers natural-language
questions over 99,089 Austrian parliamentary speeches (1996–2022), grounded in retrieved
speeches with citations and sidebar filters (country, year range, policy domain, top-k).
Built on a persistent ChromaDB index of precomputed BGE-m3 vectors. Remaining: evaluation
(Phase 5). See [DATA_MAP.md](DATA_MAP.md) for the corpus inventory and schemas.

## Run it

```bash
pip install -r requirements.txt
pip install -e .
# Put ANTHROPIC_API_KEY in .env (gitignored)
python scripts/build_index.py     # builds the Chroma index from the thesis pickle
streamlit run app.py              # chat UI
```

## Planned architecture

- **Vector store**: ChromaDB (persistent, precomputed BGE-m3 vectors + metadata filtering)
- **Embeddings**: BAAI/bge-m3 (1024-d, multilingual) — same model as the thesis, vectors reused
- **Generation**: Anthropic API (Claude), grounded-only prompting with numbered citations
- **UI**: Streamlit chat with country / year / policy-domain filters
- **Evaluation**: golden-set retrieval metrics (recall@k, MRR)

## License

MIT
