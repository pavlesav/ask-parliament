"""Patch `text_en` onto the existing Qdrant points (no re-embed, no rebuild).

Reads the per-country English sidecars (`parsed/{CC}_en.parquet`, built by
add_english_text.py) and sets a `text_en` payload field on each matching point, found
by the same id mapping the index build uses (config.point_id). The vectors are never
touched — English is display-only — so this is a cheap payload-only update over the
already-indexed corpus.

Resumable: a country whose first speech already carries `text_en` is skipped (use
--force to re-patch). Needs the Qdrant server up (QDRANT_URL).

Examples:
  python scripts/scale/patch_english_payload.py                 # all sidecars present
  python scripts/scale/patch_english_payload.py --countries LV
  python scripts/scale/patch_english_payload.py --force
"""
from __future__ import annotations

import argparse
import logging
import time

import pandas as pd
from qdrant_client import QdrantClient, models

from ask_parliament.config import (
    PARSED_DIR,
    QDRANT_COLLECTION,
    QDRANT_PATH,
    QDRANT_URL,
    point_id,
)

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("patch_english_payload")

SET_BATCH = 1_000  # points per set_payload call


def connect() -> QdrantClient:
    if QDRANT_URL:
        log.info("Connecting to Qdrant server at %s", QDRANT_URL)
        return QdrantClient(url=QDRANT_URL, timeout=120)
    log.info("Using embedded Qdrant at %s", QDRANT_PATH)
    return QdrantClient(path=str(QDRANT_PATH))


def already_patched(client: QdrantClient, first_id: str) -> bool:
    """True if the country's first speech point already carries a non-empty text_en."""
    pts = client.retrieve(QDRANT_COLLECTION, ids=[point_id(first_id)], with_payload=True)
    return bool(pts and pts[0].payload.get("text_en"))


def patch_country(client: QdrantClient, cc: str, force: bool) -> None:
    sidecar = PARSED_DIR / f"{cc}_en.parquet"
    if not sidecar.exists():
        log.warning("%s: no sidecar (%s) — run add_english_text.py first", cc, sidecar.name)
        return
    df = pd.read_parquet(sidecar)
    df = df[df["text_en"].fillna("").str.len() > 0]  # only points that actually have a translation
    if df.empty:
        log.warning("%s: sidecar has no non-empty translations — skipping", cc)
        return

    if not force and already_patched(client, df["id"].iloc[0]):
        log.info("%s: already patched (%s translations) — skipping", cc, f"{len(df):,}")
        return

    # Each point gets a *different* text_en, so we batch many single-point SetPayload
    # operations into one request via batch_update_points — one round-trip per SET_BATCH
    # points instead of per point. Payload-only writes don't need indexing, so wait=False
    # is durable-enough and fast; the last batch waits so a follow-on verify sees it.
    t0, done = time.time(), 0
    rows = list(df.itertuples(index=False, name=None))  # (id, text_en)
    for b in range(0, len(rows), SET_BATCH):
        chunk = rows[b:b + SET_BATCH]
        ops = [
            models.SetPayloadOperation(
                set_payload=models.SetPayload(payload={"text_en": text_en}, points=[point_id(sid)])
            )
            for sid, text_en in chunk
        ]
        client.batch_update_points(QDRANT_COLLECTION, update_operations=ops,
                                   wait=(b + SET_BATCH >= len(rows)))
        done += len(chunk)
        if b // SET_BATCH % 10 == 0:
            log.info("%s: %s/%s patched (%.0f s)", cc, f"{done:,}", f"{len(rows):,}", time.time() - t0)
    log.info("%s: done — %s translations patched (%.0f s)", cc, f"{len(rows):,}", time.time() - t0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--countries", help="comma-separated codes (default: all sidecars present)")
    p.add_argument("--force", action="store_true", help="re-patch countries already carrying text_en")
    a = p.parse_args()

    if a.countries:
        countries = [c.strip() for c in a.countries.split(",") if c.strip()]
    elif PARSED_DIR.exists():
        countries = sorted(f.stem[:-3] for f in PARSED_DIR.glob("*_en.parquet"))
    else:
        countries = []
    if not countries:
        log.warning("No *_en.parquet sidecars under %s — run add_english_text.py first.", PARSED_DIR)
        return

    client = connect()
    for cc in countries:
        patch_country(client, cc, force=a.force)


if __name__ == "__main__":
    main()
