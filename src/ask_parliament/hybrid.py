"""Hybrid retrieval: BM25 (keyword) fused with vector (semantic) search.

Dense embeddings capture meaning but miss exact terms; BM25 nails exact terms
but misses paraphrase. We run both and fuse them with Reciprocal Rank Fusion
(RRF), which combines by *rank* rather than score — so we never have to
reconcile BM25's unbounded scores with cosine similarity's [-1, 1] scale.

    rrf_score(doc) = Σ  1 / (RRF_K + rank_in_list)   over each list the doc is in

A doc ranked highly by either retriever scores well; a doc ranked highly by
both scores best. RRF_K=60 is the conventional default (from the original RRF
paper) and damps the influence of very high ranks.

The BM25 index is built in-process from the same speeches in the Chroma
collection, so the two retrievers always cover an identical corpus.
"""
from __future__ import annotations

import re
import time

from rank_bm25 import BM25Okapi

from ask_parliament.retrieval import RetrievedSpeech, Retriever

RRF_K = 60
DEFAULT_CANDIDATES = 50  # per-retriever depth fed into the fusion

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens, dropping single chars. Same for docs and queries."""
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 1]


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = RRF_K) -> list[str]:
    """Fuse ranked ID lists into one, best first. Ranks are 1-based."""
    scores: dict[str, float] = {}
    for ids in ranked_lists:
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


class HybridRetriever:
    """Wraps a vector Retriever and adds a BM25 index over the same speeches."""

    def __init__(self, retriever: Retriever, corpus: tuple | None = None) -> None:
        self.vector = retriever
        # Keep text + metadata in memory for BM25 and for building results.
        # Pass `corpus=(ids, docs, metas)` to reuse an already-loaded sweep
        # (the eval does this); otherwise pull it once, paginated — a single
        # get() over the whole collection trips SQLite's variable limit.
        if corpus is not None:
            ids, docs, metas = corpus
        else:
            ids, docs, metas = [], [], []
            col = retriever.collection
            for offset in range(0, col.count(), 10_000):
                g = col.get(include=["documents", "metadatas"], limit=10_000, offset=offset)
                ids += g["ids"]
                docs += g["documents"]
                metas += g["metadatas"]
        self.ids = ids
        self.docs = docs
        self.metas = metas
        self.pos = {doc_id: i for i, doc_id in enumerate(ids)}
        self.bm25 = BM25Okapi([tokenize(d) for d in docs])
        self.last_timings: dict[str, float] = {}

    def _passes(self, meta, countries, year_from, year_to, cap_domains, party) -> bool:
        """Same metadata filter as the vector side, applied to BM25 candidates."""
        if countries and meta["country"] not in countries:
            return False
        if year_from is not None and meta["year"] < year_from:
            return False
        if year_to is not None and meta["year"] > year_to:
            return False
        if cap_domains and meta["cap_domain"] not in cap_domains:
            return False
        if party and meta["party"] != party:
            return False
        return True

    def _to_speech(self, doc_id: str, similarity: float) -> RetrievedSpeech:
        m = self.metas[self.pos[doc_id]]
        return RetrievedSpeech(
            id=doc_id,
            text=self.docs[self.pos[doc_id]],
            similarity=similarity,  # cosine if the dense side found it, else 0.0 (keyword-only)
            country=m["country"], date=m["date"], year=m["year"],
            speaker=m["speaker"], speaker_id=m["speaker_id"], party=m["party"],
            party_status=m["party_status"], cap_domain=m["cap_domain"],
            topic_name=m["topic_name"], segment_id=m["segment_id"],
        )

    def search(
        self,
        query: str,
        k: int = 8,
        countries: list[str] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        cap_domains: list[str] | None = None,
        party: str | None = None,
        candidates: int = DEFAULT_CANDIDATES,
    ) -> list[RetrievedSpeech]:
        n = max(candidates, k)

        # Dense side: Chroma applies the metadata filter natively.
        t0 = time.time()
        vector_hits = self.vector.search(
            query, k=n, countries=countries, year_from=year_from,
            year_to=year_to, cap_domains=cap_domains, party=party,
        )
        t1 = time.time()
        vector_ids = [h.id for h in vector_hits]
        sim_by_id = {h.id: h.similarity for h in vector_hits}

        # Lexical side: score every doc, then take the top n that pass the filter.
        bm25_scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_ids: list[str] = []
        for i in order:
            if self._passes(self.metas[i], countries, year_from, year_to, cap_domains, party):
                bm25_ids.append(self.ids[i])
                if len(bm25_ids) >= n:
                    break
        t2 = time.time()

        fused = reciprocal_rank_fusion([vector_ids, bm25_ids])[:k]
        self.last_timings = {"vector_s": t1 - t0, "bm25_s": t2 - t1}
        return [self._to_speech(doc_id, sim_by_id.get(doc_id, 0.0)) for doc_id in fused]
