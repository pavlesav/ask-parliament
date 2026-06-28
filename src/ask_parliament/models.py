"""Shared retrieval data types.

Deliberately import-light (just dataclasses) so any layer — the API backend, the
reranker, the agentic orchestrator — can use them without pulling in a vector-store
backend or the embedding stack.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Facets:
    """Distinct filter values present in the index, for populating the UI."""

    countries: list[str]
    parties: list[str]
    cap_domains: list[str]
    year_min: int
    year_max: int


@dataclass
class RetrievedSpeech:
    """One retrieval hit with everything needed to display and cite it."""

    id: str
    text: str
    similarity: float  # cosine similarity in [-1, 1]; higher is better
    country: str
    date: str
    year: int
    speaker: str
    speaker_id: str
    party: str
    party_status: str
    cap_domain: str
    segment_id: str  # parliamentary session id (provenance)
    rerank_score: float | None = None  # cross-encoder relevance in [0,1] when reranked, else None
