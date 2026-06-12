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

🚧 Early — **Phase 0 (data inventory) complete.** See [DATA_MAP.md](DATA_MAP.md) for the
full corpus inventory and schemas.

## Planned architecture

- **Vector store**: ChromaDB (persistent, precomputed BGE-m3 vectors + metadata filtering)
- **Embeddings**: BAAI/bge-m3 (1024-d, multilingual) — same model as the thesis, vectors reused
- **Generation**: Anthropic API (Claude), grounded-only prompting with numbered citations
- **UI**: Streamlit chat with country / year / policy-domain filters
- **Evaluation**: golden-set retrieval metrics (recall@k, MRR)

## License

MIT
