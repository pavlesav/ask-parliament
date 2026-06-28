"""Agentic, self-correcting retrieval (Corrective-RAG style).

Wraps the plain `Retriever` with three upgrades, behind the same `search()`
signature so the app and CLIs swap it in with one line:

  1. query transformation — expand the question into paraphrases + a HyDE speech
     (`query_transform`), retrieve each, and RRF-fuse into one candidate pool;
  2. reranking — score the pool with a cross-encoder (`rerank`) for true relevance;
  3. self-correction — if the best reranked score is below a floor, the system
     decides the first pass was too thin, asks for broader queries, retrieves again,
     and re-ranks the enlarged pool. One round by default.

The correction *decision* uses the reranker score we already computed, so the common
(good-retrieval) case costs no extra LLM call. `last_trace` records what happened —
the variants tried, whether correction fired, the top score — so every step is visible.
"""
from __future__ import annotations

import time

from ask_parliament.config import (
    AGENTIC_MAX_ROUNDS,
    PER_QUERY_LIMIT,
    RERANK_CANDIDATES,
    RERANK_SCORE_FLOOR,
)
from ask_parliament.models import RetrievedSpeech
from ask_parliament.query_transform import corrective_rewrite, transform_query
from ask_parliament.rerank import Reranker
from ask_parliament.retrieval import Retriever

RRF_K = 60  # reciprocal-rank-fusion damping (the conventional default)


def _rrf(ranked_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    """Fuse ranked id lists into one, best first (1-based ranks)."""
    scores: dict[str, float] = {}
    for ids in ranked_lists:
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


class AgenticRetriever:
    """Query-transform + rerank + corrective loop over a base Retriever."""

    def __init__(self, base: Retriever, reranker: Reranker | None = None) -> None:
        self.base = base
        self.reranker = reranker or Reranker()
        self.last_timings: dict[str, float] = {}
        self.last_trace: dict = {}

    def _retrieve_pool(self, queries, filters) -> tuple[list[RetrievedSpeech], dict]:
        """Retrieve each query variant, keep one hit per id, RRF-fuse, truncate."""
        lists: list[list[str]] = []
        by_id: dict[str, RetrievedSpeech] = {}
        for q in queries:
            hits = self.base.search(q, k=PER_QUERY_LIMIT, **filters)
            lists.append([h.id for h in hits])
            for h in hits:
                # keep the variant where this speech scored best (highest cosine)
                if h.id not in by_id or h.similarity > by_id[h.id].similarity:
                    by_id[h.id] = h
        fused = _rrf(lists)[:RERANK_CANDIDATES]
        return [by_id[i] for i in fused], by_id

    def search(
        self,
        query: str,
        k: int = 8,
        countries: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        cap_domains: list[str] | None = None,
        party: str | None = None,
    ) -> list[RetrievedSpeech]:
        filters = dict(countries=countries, year_from=year_from, year_to=year_to,
                       cap_domains=cap_domains, party=party)

        # 1. transform: original question + paraphrases + HyDE hypothetical speech
        t0 = time.time()
        tq = transform_query(query)
        variants = [query, *tq["sub_queries"]]
        if tq["hypothetical"]:
            variants.append(tq["hypothetical"])
        t1 = time.time()

        # 2. retrieve + fuse, then 3. rerank
        pool, by_id = self._retrieve_pool(variants, filters)
        t2 = time.time()
        ranked = self.reranker.rerank(query, pool)
        t3 = time.time()

        # 4. self-correct if the best evidence is weak
        rounds = 0
        corrected = False
        correct_s = 0.0
        while ranked and ranked[0].rerank_score is not None \
                and ranked[0].rerank_score < RERANK_SCORE_FLOOR and rounds < AGENTIC_MAX_ROUNDS:
            rounds += 1
            corrected = True
            cs = time.time()
            extra = corrective_rewrite(query, [h.text for h in ranked[:3]])
            if not extra:
                break
            more, more_by_id = self._retrieve_pool(extra, filters)
            by_id.update(more_by_id)
            merged = list({h.id: h for h in (pool + more)}.values())
            pool = merged
            ranked = self.reranker.rerank(query, pool)
            correct_s += time.time() - cs

        self.last_timings = {
            "transform_s": t1 - t0, "retrieve_s": t2 - t1,
            "rerank_s": t3 - t2, "correct_s": correct_s,
        }
        self.last_trace = {
            "n_variants": len(variants),
            "sub_queries": tq["sub_queries"],
            "used_hyde": bool(tq["hypothetical"]),
            "pool_size": len(pool),
            "top_rerank": round(ranked[0].rerank_score, 3) if ranked and ranked[0].rerank_score is not None else None,
            "corrected": corrected,
            "rounds": rounds,
        }
        return ranked[:k]
