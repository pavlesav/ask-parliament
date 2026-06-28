"""Cross-encoder reranking.

Dense retrieval scores a query and a speech independently (bi-encoder): fast over
millions of vectors, but it never looks at the pair together. A cross-encoder reads
the query and one speech *jointly* and outputs a single relevance score — far more
accurate, but too slow to run over the whole corpus. So we use it the standard way:
over-fetch cheap candidates with the bi-encoder, then rerank just those.

BAAI/bge-reranker-v2-m3 is the reranker that pairs with bge-m3 — multilingual, so it
scores the native-language speeches directly. Scores are squashed to [0,1] with a
sigmoid so they read as a relevance probability and give a stable threshold for the
agentic loop's self-correction decision.
"""
from __future__ import annotations

import math

from sentence_transformers import CrossEncoder

from ask_parliament.config import RERANK_MAX_LENGTH, RERANK_MODEL, resolve_device
from ask_parliament.models import RetrievedSpeech


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Reranker:
    """Lazy cross-encoder. Loading the model (~2 GB) is deferred to first use."""

    def __init__(self) -> None:
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(RERANK_MODEL, max_length=RERANK_MAX_LENGTH, device=resolve_device())
        return self._model

    def rerank(self, query: str, hits: list[RetrievedSpeech]) -> list[RetrievedSpeech]:
        """Score each hit against the query and return them sorted best-first.

        Sets `rerank_score` (sigmoid of the cross-encoder logit) on every hit in
        place; the original cosine `similarity` is left untouched so both signals
        stay visible.
        """
        if not hits:
            return hits
        pairs = [(query, h.text) for h in hits]
        logits = self.model.predict(pairs, convert_to_numpy=True)
        for h, logit in zip(hits, logits):
            h.rerank_score = _sigmoid(float(logit))
        return sorted(hits, key=lambda h: h.rerank_score, reverse=True)
