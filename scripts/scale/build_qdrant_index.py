"""Load the embedded ParlaMint shards into a searchable Qdrant index (resumable).

Reads, per country, data/parlamint/parsed/{CC}.parquet (speech text + metadata)
and data/parlamint/embeddings/{CC}/shard_*.npy (the matching BGE-m3 vectors,
row-aligned 1:1 with the parquet) and upserts speech-level points into one Qdrant
collection spanning all parliaments. Memory stays bounded: countries are streamed
one at a time and shards are loaded individually.

Connection follows config: QDRANT_URL set -> that server; else an embedded
on-disk store at QDRANT_PATH (no server needed).

Examples:
  python scripts/scale/build_qdrant_index.py                 # all parsed countries
  python scripts/scale/build_qdrant_index.py --countries LV  # one country (smoke test)
  python scripts/scale/build_qdrant_index.py --recreate      # drop + rebuild from scratch
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models

from ask_parliament.config import (
    EMB_DIR,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    FACETS_PATH,
    PARSED_DIR,
    QDRANT_COLLECTION,
    QDRANT_PATH,
    QDRANT_URL,
    point_id,
)

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("build_qdrant_index")

UPSERT_BATCH = 1_000  # points per upsert call
CLIENT_TIMEOUT = 120  # generous: a wait=true upsert can briefly stall behind a segment flush
INDEXING_THRESHOLD = 20_000  # Qdrant default; re-applied after the bulk load
# Speech id -> Qdrant point id mapping lives in config (config.point_id), so this
# build and the payload patchers stay in lock-step on the id scheme.

# Parquet column -> Qdrant payload key. Everything we filter, cite, or display on.
PAYLOAD_COLUMNS = {
    "id": "speech_id",
    "text": "text",
    "country": "country",
    "date": "date",
    "year": "year",
    "speaker": "speaker",
    "speaker_id": "speaker_id",
    "party": "party",
    "party_name": "party_name",
    "party_status": "party_status",
    "party_orientation": "party_orientation",
    "gender": "gender",
    "cap_topic": "cap_topic",
    "subcorpus": "subcorpus",
    "term": "term",
    "text_id": "text_id",
}
# Payload fields the retriever filters on -> indexed for efficient filtering on a server.
PAYLOAD_INDEXES = {
    "country": models.PayloadSchemaType.KEYWORD,
    "year": models.PayloadSchemaType.INTEGER,
    "cap_topic": models.PayloadSchemaType.KEYWORD,
    "party": models.PayloadSchemaType.KEYWORD,
}


def connect() -> QdrantClient:
    if QDRANT_URL:
        log.info("Connecting to Qdrant server at %s", QDRANT_URL)
        return QdrantClient(url=QDRANT_URL, timeout=CLIENT_TIMEOUT)
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    log.info("Using embedded Qdrant at %s", QDRANT_PATH)
    return QdrantClient(path=str(QDRANT_PATH))


def ensure_collection(client: QdrantClient, recreate: bool) -> None:
    if recreate and client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)
        log.info("Dropped existing collection %s", QDRANT_COLLECTION)
    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            # Cosine over unit-norm BGE-m3 vectors; on_disk keeps RAM bounded on a server.
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIM, distance=models.Distance.COSINE, on_disk=True
            ),
            on_disk_payload=True,
            # Defer HNSW indexing during the bulk load: with the threshold at 0 each
            # upsert is a constant-time storage write (no inline index building), so
            # upserts stay fast and don't slow down / time out as the collection grows.
            # enable_indexing() restores the normal threshold once the load is done.
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )
        for field, schema in PAYLOAD_INDEXES.items():
            client.create_payload_index(QDRANT_COLLECTION, field_name=field, field_schema=schema)
        log.info("Created collection %s (dim %d, cosine) + payload indexes; indexing deferred",
                 QDRANT_COLLECTION, EMBEDDING_DIM)


def enable_indexing(client: QdrantClient) -> None:
    """Restore the HNSW indexing threshold after the bulk load; the server then
    builds the vector index in the background (queries still work meanwhile)."""
    client.update_collection(
        QDRANT_COLLECTION,
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=INDEXING_THRESHOLD),
    )
    log.info("Re-enabled HNSW indexing (threshold=%d) — server builds the index in the background",
             INDEXING_THRESHOLD)


def load_manifest(cc: str) -> dict:
    return json.loads((EMB_DIR / cc / "manifest.json").read_text())


def country_count(client: QdrantClient, cc: str) -> int:
    return client.count(
        QDRANT_COLLECTION,
        count_filter=models.Filter(must=[models.FieldCondition(key="country", match=models.MatchValue(value=cc))]),
        exact=True,
    ).count


def iter_shards(cc: str, manifest: dict):
    """Yield (start, end, vectors) per shard, validating row counts against the manifest."""
    n_rows, shard_size, n_shards = manifest["n_rows"], manifest["shard_size"], manifest["n_shards"]
    for i in range(n_shards):
        start, end = i * shard_size, min((i + 1) * shard_size, n_rows)
        path = EMB_DIR / cc / f"shard_{i:05d}.npy"
        vecs = np.load(path)
        if vecs.shape != (end - start, EMBEDDING_DIM):
            raise RuntimeError(f"{cc} shard {i}: shape {vecs.shape}, expected {(end - start, EMBEDDING_DIM)}")
        yield start, end, vecs.astype(np.float32)


def upsert_country(client: QdrantClient, cc: str, force: bool) -> int:
    pq = PARSED_DIR / f"{cc}.parquet"
    manifest = load_manifest(cc)
    df = pd.read_parquet(pq, columns=list(PAYLOAD_COLUMNS))
    if len(df) != manifest["n_rows"]:
        raise RuntimeError(f"{cc}: parquet rows {len(df)} != manifest n_rows {manifest['n_rows']}")

    if not force:
        have = country_count(client, cc)
        if have == manifest["n_rows"]:
            log.info("%s: already indexed (%s points) — skipping", cc, f"{have:,}")
            return 0
        if have:
            log.info("%s: %s/%s points present — re-upserting (idempotent by id)", cc, f"{have:,}", f"{manifest['n_rows']:,}")

    payloads = df.rename(columns=PAYLOAD_COLUMNS).to_dict(orient="records")
    # Merge the English machine-translation sidecar (id -> text_en) if it's been built,
    # so a fresh build carries the optional English view too. Missing/empty -> no key
    # (the app falls back to the native text).
    en_path = PARSED_DIR / f"{cc}_en.parquet"
    if en_path.exists():
        en = dict(pd.read_parquet(en_path).itertuples(index=False, name=None))
        n_en = 0
        for p in payloads:
            t = en.get(p["speech_id"], "")
            if t:
                p["text_en"] = t
                n_en += 1
        log.info("%s: merged English text for %s/%s points", cc, f"{n_en:,}", f"{len(payloads):,}")
    ids = [point_id(p["speech_id"]) for p in payloads]

    t0, done = time.time(), 0
    for start, end, vecs in iter_shards(cc, manifest):
        for b in range(start, end, UPSERT_BATCH):
            e = min(b + UPSERT_BATCH, end)
            client.upsert(
                QDRANT_COLLECTION,
                points=models.Batch(
                    ids=ids[b:e],
                    vectors=vecs[b - start : e - start].tolist(),
                    payloads=payloads[b:e],
                ),
            )
            done += e - b
        log.info("%s: %s/%s upserted (%.0f s)", cc, f"{done:,}", f"{manifest['n_rows']:,}", time.time() - t0)
    return done


def write_facets(countries: list[str]) -> None:
    """Scan the lean payload columns of every parsed parquet to build the filter
    sidecar the app reads (so it never sweeps millions of points just for facets)."""
    countries_present: set[str] = set()
    cap_topics: set[str] = set()
    parties: set[str] = set()
    year_min = year_max = None
    for cc in countries:
        pq = PARSED_DIR / f"{cc}.parquet"
        if not pq.exists():
            continue
        df = pd.read_parquet(pq, columns=["country", "year", "cap_topic", "party"])
        countries_present.update(df["country"].unique().tolist())
        cap_topics.update(t for t in df["cap_topic"].unique().tolist() if t not in ("-", ""))
        parties.update(p for p in df["party"].unique().tolist() if p not in ("-", ""))
        ymin, ymax = int(df["year"].min()), int(df["year"].max())
        year_min = ymin if year_min is None else min(year_min, ymin)
        year_max = ymax if year_max is None else max(year_max, ymax)
    facets = {
        "countries": sorted(countries_present),
        "cap_domains": sorted(cap_topics),
        "parties": sorted(parties),
        "year_min": year_min,
        "year_max": year_max,
    }
    FACETS_PATH.write_text(json.dumps(facets, indent=2))
    log.info("Wrote facets -> %s (%d countries, %d cap domains, %d parties, %s-%s)",
             FACETS_PATH, len(facets["countries"]), len(facets["cap_domains"]),
             len(facets["parties"]), year_min, year_max)


def verify(client: QdrantClient, cc: str) -> None:
    """Self-retrieval probe: a stored vector should return its own point at score ~1.0."""
    manifest = load_manifest(cc)
    df = pd.read_parquet(PARSED_DIR / f"{cc}.parquet", columns=["id"])
    probe_id = point_id(df["id"].iloc[0])
    vec = np.load(EMB_DIR / cc / "shard_00000.npy")[0].astype(np.float32).tolist()
    hit = client.query_points(QDRANT_COLLECTION, query=vec, limit=1, with_payload=True).points[0]
    if hit.id != probe_id or hit.score < 0.99:
        raise RuntimeError(f"{cc} self-retrieval failed: got {hit.id} at score {hit.score:.4f}")
    log.info("%s self-retrieval OK (score %.4f, speaker=%s)", cc, hit.score, hit.payload.get("speaker"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--countries", help="comma-separated codes (default: all parsed parquet)")
    p.add_argument("--recreate", action="store_true", help="drop the collection and rebuild from scratch")
    p.add_argument("--force", action="store_true", help="re-upsert a country even if already complete")
    a = p.parse_args()

    if a.countries:
        countries = [c.strip() for c in a.countries.split(",") if c.strip()]
    elif PARSED_DIR.exists():
        countries = sorted(f.stem for f in PARSED_DIR.glob("*.parquet"))
    else:
        countries = []
    if not countries:
        log.warning("No parsed parquet under %s — run parse_parlamint.py first.", PARSED_DIR)
        return

    client = connect()
    ensure_collection(client, recreate=a.recreate)

    t0 = time.time()
    for cc in countries:
        upsert_country(client, cc, force=a.force)
    enable_indexing(client)
    verify(client, countries[0])
    write_facets(countries)
    total = client.count(QDRANT_COLLECTION, exact=True).count
    log.info("Done (%d countries, %s points total) in %.0f s", len(countries), f"{total:,}", time.time() - t0)


if __name__ == "__main__":
    main()
