"""Semantic retrieval over the speech index.

Embeds the user query with BGE-m3 (the exact model that produced the stored
vectors — queries and documents must live in the same space) and runs a top-k
cosine search in Chroma, optionally constrained by metadata filters.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import chromadb
from sentence_transformers import SentenceTransformer

from ask_parliament.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL


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
    topic_name: str
    segment_id: str


def build_where(
    countries: list[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    cap_domains: list[str] | None = None,
    party: str | None = None,
) -> dict | None:
    """Translate filter arguments into a Chroma `where` clause.

    Chroma wants a bare clause for one condition and an explicit $and for
    several, so we collect then wrap.
    """
    clauses: list[dict] = []
    if countries:
        clauses.append({"country": {"$in": countries}})
    if year_from is not None:
        clauses.append({"year": {"$gte": year_from}})
    if year_to is not None:
        clauses.append({"year": {"$lte": year_to}})
    if cap_domains:
        clauses.append({"cap_domain": {"$in": cap_domains}})
    if party:
        clauses.append({"party": {"$eq": party}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class Retriever:
    """Owns the Chroma collection and the query-embedding model."""

    def __init__(self) -> None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = client.get_collection(COLLECTION_NAME)
        self._model: SentenceTransformer | None = None
        self.last_timings: dict[str, float] = {}

    @property
    def model(self) -> SentenceTransformer:
        # Lazy: loading BGE-m3 takes ~10-30 s on CPU; skip it until first query.
        if self._model is None:
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def facets(self) -> Facets:
        """Scan the collection's metadata once to list available filter values.

        Chroma has no DISTINCT, so we sweep all metadatas (cheap fields only, no
        embeddings or documents) and reduce. Paginated because a single get() over
        the whole collection trips SQLite's variable limit. Cached at app startup.
        """
        countries, parties, cap_domains = set(), set(), set()
        year_min, year_max = None, None
        batch = 10_000
        for offset in range(0, self.collection.count(), batch):
            metas = self.collection.get(
                include=["metadatas"], limit=batch, offset=offset
            )["metadatas"]
            for m in metas:
                countries.add(m["country"])
                cap_domains.add(m["cap_domain"])
                if m["party"] != "-":
                    parties.add(m["party"])
                y = m["year"]
                year_min = y if year_min is None else min(year_min, y)
                year_max = y if year_max is None else max(year_max, y)
        return Facets(
            countries=sorted(countries),
            parties=sorted(parties),
            cap_domains=sorted(cap_domains),
            year_min=year_min,
            year_max=year_max,
        )

    def embed_query(self, query: str) -> list[float]:
        # Plain encode(), no instruction prefix: BGE-m3 takes none, and the
        # thesis embedded documents the same way. The model L2-normalizes.
        return self.model.encode(query).tolist()

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
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=build_where(countries, year_from, year_to, cap_domains, party),
            include=["documents", "metadatas", "distances"],
        )
        self.last_timings = {"embed_s": t1 - t0, "search_s": time.time() - t1}

        hits = []
        for id_, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append(
                RetrievedSpeech(
                    id=id_,
                    text=doc,
                    similarity=1.0 - dist,  # Chroma cosine distance = 1 - cos_sim
                    country=meta["country"],
                    date=meta["date"],
                    year=meta["year"],
                    speaker=meta["speaker"],
                    speaker_id=meta["speaker_id"],
                    party=meta["party"],
                    party_status=meta["party_status"],
                    cap_domain=meta["cap_domain"],
                    topic_name=meta["topic_name"],
                    segment_id=meta["segment_id"],
                )
            )
        return hits
