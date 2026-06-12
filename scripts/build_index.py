"""Build the persistent Chroma index from the thesis pickle (idempotent).

Loads AT_final.pkl (~3.2 GB on disk, ~4 GB RAM, ~15 s), selects the demo corpus
(Regular speakers, >=300 chars English text, 1996-2022), and upserts speeches with
their precomputed BGE-m3 vectors into a persistent Chroma collection.

Usage:
    python scripts/build_index.py            # builds, or no-ops if already complete
    python scripts/build_index.py --rebuild  # drop the collection and rebuild
"""
from __future__ import annotations

import argparse
import gc
import logging
import time

import chromadb
import numpy as np
import pandas as pd

from ask_parliament.config import (
    AT_FINAL_PKL,
    CHROMA_DIR,
    COLLECTION_NAME,
    COUNTRY,
    EMBEDDING_DIM,
    EXPECTED_SPEECH_COUNT,
    MIN_TEXT_CHARS,
    SPEAKER_ROLE,
)

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("build_index")

UPSERT_BATCH = 2_000

# Pickle column -> Chroma metadata key. Everything the app filters or cites on.
METADATA_COLUMNS = {
    "Date": "date",
    "Year": "year",
    "Speaker_name": "speaker",
    "Speaker_ID": "speaker_id",
    "Speaker_party": "party",
    "Party_status": "party_status",
    "CAP_Category_AT_english": "cap_domain",
    "Topic_Name_AT_english": "topic_name",
    "Segment_ID_English": "segment_id",
}


def load_corpus() -> pd.DataFrame:
    """Load the thesis pickle and reduce it to the demo corpus."""
    t0 = time.time()
    log.info("Loading %s ...", AT_FINAL_PKL)
    df = pd.read_pickle(AT_FINAL_PKL)
    log.info("Loaded %s rows x %s cols in %.0f s", f"{len(df):,}", df.shape[1], time.time() - t0)

    needed = ["ID", "Text_English", "Speech_Embeddings_English", "Speaker_role", *METADATA_COLUMNS]
    corpus = df[needed].copy()
    del df
    gc.collect()

    n0 = len(corpus)
    corpus = corpus[corpus["Speaker_role"] == SPEAKER_ROLE]
    n1 = len(corpus)
    corpus = corpus[corpus["Text_English"].str.len().fillna(0) >= MIN_TEXT_CHARS]
    n2 = len(corpus)
    corpus = corpus[corpus["Speech_Embeddings_English"].notna()]
    n3 = len(corpus)
    log.info(
        "Filters: %s rows -> %s (role=%s) -> %s (len>=%s) -> %s (has embedding)",
        f"{n0:,}", f"{n1:,}", SPEAKER_ROLE, f"{n2:,}", MIN_TEXT_CHARS, f"{n3:,}",
    )
    if n3 != EXPECTED_SPEECH_COUNT:
        log.warning("Corpus has %s rows, expected %s — check filters/data version",
                    f"{n3:,}", f"{EXPECTED_SPEECH_COUNT:,}")

    # Normalisations (DATA_MAP.md §5): case-duplicate party code, Chroma-safe values.
    corpus["Speaker_party"] = corpus["Speaker_party"].replace({"GRÜNE": "Grüne"})
    corpus["Date"] = corpus["Date"].dt.strftime("%Y-%m-%d")
    corpus["Year"] = corpus["Year"].astype(int)
    for col in METADATA_COLUMNS:
        if col not in ("Date", "Year"):
            corpus[col] = corpus[col].fillna("-").astype(str)

    return corpus.reset_index(drop=True)


def get_collection(client: chromadb.api.ClientAPI, rebuild: bool) -> chromadb.Collection:
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            log.info("Dropped existing collection %s", COLLECTION_NAME)
        except Exception:
            pass  # didn't exist
    # Cosine is mandatory: chunk-averaged thesis vectors are not unit-norm.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    space = (collection.metadata or {}).get("hnsw:space")
    if space != "cosine":
        raise RuntimeError(f"Collection space is {space!r}, expected 'cosine'")
    return collection


def build(corpus: pd.DataFrame, collection: chromadb.Collection) -> None:
    embeddings = np.stack(corpus["Speech_Embeddings_English"].to_numpy()).astype(np.float32)
    if embeddings.shape != (len(corpus), EMBEDDING_DIM):
        raise RuntimeError(f"Unexpected embedding matrix shape {embeddings.shape}")

    ids = corpus["ID"].tolist()
    documents = corpus["Text_English"].tolist()
    metadatas = (
        corpus[list(METADATA_COLUMNS)]
        .rename(columns=METADATA_COLUMNS)
        .to_dict(orient="records")
    )
    for m in metadatas:
        m["country"] = COUNTRY

    t0 = time.time()
    for start in range(0, len(ids), UPSERT_BATCH):
        end = min(start + UPSERT_BATCH, len(ids))
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        if (start // UPSERT_BATCH) % 10 == 0:
            log.info("Upserted %s/%s (%.0f s elapsed)", f"{end:,}", f"{len(ids):,}", time.time() - t0)
    log.info("Upsert complete: %s records in %.0f s", f"{len(ids):,}", time.time() - t0)


def verify(corpus: pd.DataFrame, collection: chromadb.Collection) -> None:
    """Smoke checks: count, self-retrieval, metadata filtering."""
    count = collection.count()
    log.info("Collection count: %s (expected %s)", f"{count:,}", f"{len(corpus):,}")
    if count != len(corpus):
        raise RuntimeError("Count mismatch after build")

    probe = corpus.iloc[0]
    res = collection.query(
        query_embeddings=[probe["Speech_Embeddings_English"].astype(np.float32).tolist()],
        n_results=1,
    )
    top_id, distance = res["ids"][0][0], res["distances"][0][0]
    if top_id != probe["ID"] or distance > 1e-3:
        raise RuntimeError(f"Self-retrieval failed: got {top_id} at distance {distance}")
    log.info("Self-retrieval OK (distance %.2e)", distance)

    filtered = collection.query(
        query_embeddings=[probe["Speech_Embeddings_English"].astype(np.float32).tolist()],
        n_results=3,
        where={"$and": [{"year": {"$gte": 2015}}, {"cap_domain": {"$eq": "Immigration"}}]},
    )
    years = [m["year"] for m in filtered["metadatas"][0]]
    domains = {m["cap_domain"] for m in filtered["metadatas"][0]}
    if not years or min(years) < 2015 or domains != {"Immigration"}:
        raise RuntimeError(f"Metadata filter check failed: years={years}, domains={domains}")
    log.info("Metadata filter OK (year>=2015 & cap_domain=Immigration -> %d hits)", len(years))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="drop and rebuild the collection")
    args = parser.parse_args()

    t0 = time.time()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = get_collection(client, rebuild=args.rebuild)

    if collection.count() == EXPECTED_SPEECH_COUNT and not args.rebuild:
        log.info("Collection %s already complete (%s records) — nothing to do. "
                 "Use --rebuild to force.", COLLECTION_NAME, f"{EXPECTED_SPEECH_COUNT:,}")
        return

    corpus = load_corpus()
    build(corpus, collection)
    verify(corpus, collection)
    log.info("Done in %.0f s total. Store: %s", time.time() - t0, CHROMA_DIR)


if __name__ == "__main__":
    main()
