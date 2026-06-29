"""Semantic retrieval over the multi-country Qdrant index.

Embeds the user query with BGE-m3 — the exact model that produced the stored
vectors, so queries and documents share one space — and runs a top-k cosine search
in Qdrant, optionally constrained by metadata (country, year, CAP domain, party).

The store is chosen by config: QDRANT_URL set -> that server (HNSW, scales to the
full corpus); else an embedded on-disk store at QDRANT_PATH (fine for small subsets).
"""
from __future__ import annotations

import json
import time

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from ask_parliament.config import (
    EMBEDDING_MODEL,
    FACETS_PATH,
    QDRANT_COLLECTION,
    QDRANT_PATH,
    QDRANT_URL,
    resolve_device,
)
from ask_parliament.models import Facets, RetrievedSpeech


def _connect() -> QdrantClient:
    # A modest timeout cushions the first queries while the server may still be
    # building the HNSW index in the background right after a bulk load.
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, timeout=60)
    return QdrantClient(path=str(QDRANT_PATH))


def build_filter(
    countries: list[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    cap_domains: list[str] | None = None,
    party: str | None = None,
) -> models.Filter | None:
    """Translate filter arguments into a Qdrant must-filter (cap_domains -> cap_topic)."""
    must: list[models.FieldCondition] = []
    if countries:
        must.append(models.FieldCondition(key="country", match=models.MatchAny(any=countries)))
    if year_from is not None or year_to is not None:
        must.append(models.FieldCondition(key="year", range=models.Range(gte=year_from, lte=year_to)))
    if cap_domains:
        must.append(models.FieldCondition(key="cap_topic", match=models.MatchAny(any=cap_domains)))
    if party:
        must.append(models.FieldCondition(key="party", match=models.MatchValue(value=party)))
    return models.Filter(must=must) if must else None


class Retriever:
    """Owns the Qdrant collection and the query-embedding model."""

    def __init__(self) -> None:
        self.client = _connect()
        self._model: SentenceTransformer | None = None
        self.last_timings: dict[str, float] = {}

    @property
    def model(self) -> SentenceTransformer:
        # Lazy: loading BGE-m3 takes ~5-30 s; skip it until the first query.
        if self._model is None:
            self._model = SentenceTransformer(EMBEDDING_MODEL, device=resolve_device())
        return self._model

    def facets(self) -> Facets:
        """Read the filter values from the sidecar the index build wrote (no payload sweep)."""
        data = json.loads(FACETS_PATH.read_text())
        return Facets(
            countries=data["countries"],
            parties=data["parties"],
            cap_domains=data["cap_domains"],
            year_min=data["year_min"],
            year_max=data["year_max"],
        )

    def embed_query(self, query: str) -> list[float]:
        # Plain encode(), no instruction prefix: BGE-m3 takes none, and the corpus
        # was embedded the same way. Cosine ranking is invariant to query norm.
        return self.model.encode(query).tolist()

    @staticmethod
    def _to_speech(point) -> RetrievedSpeech:
        m = point.payload
        return RetrievedSpeech(
            id=m["speech_id"],
            text=m["text"],
            similarity=point.score,  # Qdrant returns cosine similarity directly
            country=m["country"],
            date=m["date"],
            year=m["year"],
            speaker=m["speaker"],
            speaker_id=m["speaker_id"],
            party=m["party"],
            party_status=m["party_status"],
            cap_domain=m["cap_topic"],
            segment_id=m.get("text_id", "-"),  # parliamentary session id
            text_en=m.get("text_en"),  # present once the ParlaMint-en payload is patched in
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
    ) -> list[RetrievedSpeech]:
        t0 = time.time()
        query_embedding = self.embed_query(query)
        t1 = time.time()
        res = self.client.query_points(
            QDRANT_COLLECTION,
            query=query_embedding,
            limit=k,
            query_filter=build_filter(countries, year_from, year_to, cap_domains, party),
            with_payload=True,
        ).points
        self.last_timings = {"embed_s": t1 - t0, "search_s": time.time() - t1}
        return [self._to_speech(p) for p in res]
