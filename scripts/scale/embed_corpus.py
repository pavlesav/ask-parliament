"""Embed parsed ParlaMint speeches with BGE-m3 (sharded, resumable, parallel).

Reads data/parlamint/parsed/{CC}.parquet and writes float16 vector shards to
data/parlamint/embeddings/{CC}/shard_{i:05d}.npy plus a manifest.json. Each shard
is an independent unit of work, so:

  * rerun to RESUME — complete shards are detected and skipped;
  * run several processes over disjoint shards/countries to PARALLELIZE
    across GPUs.

Vectors are L2-normalized (cosine space, matching the app) and truncated to
--max-seq-length tokens for throughput.

Examples:
  # one GPU, every parsed country (resumable)
  python scripts/scale/embed_corpus.py

  # just a few countries
  python scripts/scale/embed_corpus.py --countries HR,RS,SI

  # two GPUs, split the shards of every country across them
  CUDA_VISIBLE_DEVICES=0 python scripts/scale/embed_corpus.py --workers 2 --worker-id 0 &
  CUDA_VISIBLE_DEVICES=1 python scripts/scale/embed_corpus.py --workers 2 --worker-id 1 &

  # validate the whole pipeline on a laptop with no GPU/model:
  python scripts/scale/embed_corpus.py --countries LV --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ask_parliament.config import (
    EMB_DIR,
    EMBED_BATCH_SIZE,
    EMBED_DTYPE,
    EMBED_MAX_SEQ_LENGTH,
    EMBED_SHARD_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    PARSED_DIR,
)

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("embed_corpus")


def resolve_device(arg: str | None) -> str:
    if arg:
        return arg
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class Bge:
    """Lazy BGE-m3 encoder. torch/sentence-transformers import only on a real run."""

    def __init__(self, device: str, max_seq_length: int):
        from sentence_transformers import SentenceTransformer
        log.info("Loading %s on %s ...", EMBEDDING_MODEL, device)
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        self.model.max_seq_length = max_seq_length
        if device.startswith("cuda"):
            self.model = self.model.half()  # fp16 — faster, less VRAM

    def __call__(self, texts: list[str], batch_size: int) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )


def fake_encoder(texts: list[str], batch_size: int) -> np.ndarray:
    """Deterministic unit vectors for --dry-run (no torch/model required)."""
    out = np.empty((len(texts), EMBEDDING_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        rng = np.random.default_rng((len(t) * 1_000_003 + i) & 0xFFFFFFFF)
        v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
        out[i] = v / (np.linalg.norm(v) + 1e-12)
    return out


def _shard_ok(path: Path, expect_rows: int) -> bool:
    """True if the shard file exists and has the expected number of rows."""
    if not path.exists():
        return False
    try:
        return int(np.load(path, mmap_mode="r").shape[0]) == expect_rows
    except Exception:
        return False


def embed_country(cc, encoder, batch_size, shard_size, workers, worker_id, force, max_seq_length):
    pq = PARSED_DIR / f"{cc}.parquet"
    if not pq.exists():
        log.warning("%s: no parsed parquet (%s) — skipping", cc, pq)
        return
    texts = pd.read_parquet(pq, columns=["text"])["text"].tolist()
    n = len(texts)
    n_shards = max(1, math.ceil(n / shard_size))
    out_dir = EMB_DIR / cc
    out_dir.mkdir(parents=True, exist_ok=True)

    mine = [i for i in range(n_shards) if i % workers == worker_id]
    log.info("%s: %s speeches, %d shards (%d assigned to worker %d/%d)",
             cc, f"{n:,}", n_shards, len(mine), worker_id, workers)

    for i in mine:
        start, end = i * shard_size, min((i + 1) * shard_size, n)
        path = out_dir / f"shard_{i:05d}.npy"
        if not force and _shard_ok(path, end - start):
            continue
        t0 = time.time()
        vecs = encoder(texts[start:end], batch_size).astype(EMBED_DTYPE)
        if vecs.shape != (end - start, EMBEDDING_DIM):
            raise RuntimeError(f"{cc} shard {i}: unexpected shape {vecs.shape}")
        tmp = out_dir / f"shard_{i:05d}.npy.tmp"
        with open(tmp, "wb") as fh:           # explicit handle so np.save keeps our name
            np.save(fh, vecs)
        os.replace(tmp, path)
        log.info("%s: shard %d/%d (%s rows) in %.1f s",
                 cc, i, n_shards - 1, f"{end - start:,}", time.time() - t0)

    # Write the manifest once every shard exists (any worker may do this).
    if all(_shard_ok(out_dir / f"shard_{i:05d}.npy", min((i + 1) * shard_size, n) - i * shard_size)
           for i in range(n_shards)):
        manifest = {
            "country": cc, "model": EMBEDDING_MODEL, "dim": EMBEDDING_DIM,
            "dtype": EMBED_DTYPE, "normalized": True, "n_rows": n,
            "shard_size": shard_size, "n_shards": n_shards,
            "max_seq_length": max_seq_length, "source": "ParlaMint 5.0 (native text)",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        log.info("%s: complete — manifest written (%s vectors)", cc, f"{n:,}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--countries", help="comma-separated codes (default: all parsed parquet)")
    p.add_argument("--batch-size", type=int, default=EMBED_BATCH_SIZE)
    p.add_argument("--shard-size", type=int, default=EMBED_SHARD_SIZE)
    p.add_argument("--max-seq-length", type=int, default=EMBED_MAX_SEQ_LENGTH)
    p.add_argument("--device", help="cuda | cuda:0 | cpu (default: auto-detect)")
    p.add_argument("--workers", type=int, default=1, help="number of parallel processes sharing the work")
    p.add_argument("--worker-id", type=int, default=0, help="this process's id in [0, workers)")
    p.add_argument("--dry-run", action="store_true", help="fake encoder, no torch/model (pipeline test)")
    p.add_argument("--force", action="store_true", help="recompute shards even if present")
    a = p.parse_args()
    if not (0 <= a.worker_id < a.workers):
        p.error("--worker-id must be in [0, --workers)")

    if a.countries:
        countries = [c.strip() for c in a.countries.split(",") if c.strip()]
    elif PARSED_DIR.exists():
        countries = sorted(f.stem for f in PARSED_DIR.glob("*.parquet"))
    else:
        countries = []
    if not countries:
        log.warning("No parsed parquet under %s — run parse_parlamint.py first.", PARSED_DIR)
        return

    if a.dry_run:
        log.info("DRY RUN: using a fake encoder (no model loaded)")
        encoder = fake_encoder
    else:
        device = resolve_device(a.device)
        if device == "cpu":
            log.warning("Running on CPU — BGE-m3 over the full corpus will be very slow. "
                        "Use a CUDA GPU for the real run (or pass --dry-run to test the pipeline).")
        encoder = Bge(device, a.max_seq_length)

    t0 = time.time()
    for cc in countries:
        embed_country(cc, encoder, a.batch_size, a.shard_size, a.workers, a.worker_id,
                      a.force, a.max_seq_length)
    log.info("Done (%d countries) in %.1f s", len(countries), time.time() - t0)


if __name__ == "__main__":
    main()
