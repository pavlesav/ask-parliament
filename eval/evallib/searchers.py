"""Searcher variants for ablation — isolate what each retrieval stage buys you.

The agentic pipeline stacks three ideas on top of plain vector search (query
transformation, cross-encoder reranking, and a self-correction loop). To report
*which* of them actually moves the numbers, the eval runs the same questions through
each configuration in isolation:

    plain      vector search only (the baseline)
    rerank     vector over-fetch -> cross-encoder rerank -> top k   (rerank, no transform)
    transform  multi-query + HyDE -> RRF fuse -> top k              (transform, no rerank)
    agentic    the full pipeline (transform + RRF + rerank + self-correct)

Every variant exposes the same `search(query, k, **filters)` signature as the base
`Retriever`, so the eval treats them uniformly. They reuse the real production code
(`Retriever`, `Reranker`, `transform_query`, `AgenticRetriever`) — no re-implementation,
so the eval measures the system as shipped.
"""
from __future__ import annotations

from ask_parliament.agentic import AgenticRetriever, _rrf
from ask_parliament.config import PER_QUERY_LIMIT, RERANK_CANDIDATES
from ask_parliament.query_transform import transform_query
from ask_parliament.rerank import Reranker
from ask_parliament.retrieval import Retriever


class PlainSearcher:
    """Baseline: a single dense vector search."""

    name = "plain"

    def __init__(self, base: Retriever) -> None:
        self.base = base

    def search(self, query: str, k: int = 8, **filters):
        return self.base.search(query, k=k, **filters)


class RerankSearcher:
    """Over-fetch with the bi-encoder, then re-order with the cross-encoder.

    Isolates the reranker's contribution — same single query as `plain`, but the top
    k are chosen by joint query–speech scoring instead of cosine alone.
    """

    name = "rerank"

    def __init__(self, base: Retriever, reranker: Reranker | None = None,
                 pool: int = RERANK_CANDIDATES) -> None:
        self.base = base
        self.reranker = reranker or Reranker()
        self.pool = pool

    def search(self, query: str, k: int = 8, **filters):
        candidates = self.base.search(query, k=self.pool, **filters)
        return self.reranker.rerank(query, candidates)[:k]


class TransformSearcher:
    """Multi-query + HyDE, RRF-fused — but no reranking.

    Isolates query transformation's contribution to *recall*: the fused order is
    purely from where each variant placed each speech, with no cross-encoder pass.
    """

    name = "transform"

    def __init__(self, base: Retriever, per_query: int = PER_QUERY_LIMIT) -> None:
        self.base = base
        self.per_query = per_query

    def search(self, query: str, k: int = 8, **filters):
        tq = transform_query(query)
        variants = [query, *tq["sub_queries"]]
        if tq["hypothetical"]:
            variants.append(tq["hypothetical"])
        lists, by_id = [], {}
        for q in variants:
            hits = self.base.search(q, k=self.per_query, **filters)
            lists.append([h.id for h in hits])
            for h in hits:
                if h.id not in by_id or h.similarity > by_id[h.id].similarity:
                    by_id[h.id] = h
        fused = _rrf(lists)[:k]
        return [by_id[i] for i in fused]


class AgenticSearcher:
    """The full shipped pipeline (transform + RRF + rerank + self-correct)."""

    name = "agentic"

    def __init__(self, base: Retriever, reranker: Reranker | None = None) -> None:
        self.inner = AgenticRetriever(base, reranker=reranker)

    def search(self, query: str, k: int = 8, **filters):
        return self.inner.search(query, k=k, **filters)


def build_searchers(names: list[str], base: Retriever, reranker: Reranker | None = None) -> dict:
    """Instantiate the requested variants, sharing one reranker (load the model once)."""
    shared = reranker
    if any(n in ("rerank", "agentic") for n in names) and shared is None:
        shared = Reranker()
    factory = {
        "plain": lambda: PlainSearcher(base),
        "rerank": lambda: RerankSearcher(base, shared),
        "transform": lambda: TransformSearcher(base),
        "agentic": lambda: AgenticSearcher(base, shared),
    }
    out = {}
    for n in names:
        if n not in factory:
            raise ValueError(f"unknown searcher '{n}' (have {sorted(factory)})")
        out[n] = factory[n]()
    return out
