"""Build the per-country English-text sidecar parquets (resumable, no Qdrant needed).

For each country, joins the ParlaMint-en machine translations (read from the extracted
`raw_en/{CC}/` subtree) onto the speeches we actually indexed (`parsed/{CC}.parquet`, by
utterance id) and writes `parsed/{CC}_en.parquet` with two columns: `id`, `text_en`.

This is the durable artefact: `patch_english_payload.py` pushes it into the live index,
and `build_qdrant_index.py` merges it on a fresh rebuild. Coverage (how many indexed
speeches got a translation) is reported per country — it should be ~100%, since the
English edition mirrors the native one.

Examples:
  python scripts/scale/add_english_text.py                  # all countries with raw_en + parquet
  python scripts/scale/add_english_text.py --countries LV
  python scripts/scale/add_english_text.py --force          # rebuild existing sidecars
"""
from __future__ import annotations

import argparse
import logging
import time

import pandas as pd

from ask_parliament.config import PARSED_DIR, RAW_EN_DIR
from ask_parliament.parlamint import read_english_texts

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("add_english_text")


def build_country(cc: str, force: bool) -> None:
    pq = PARSED_DIR / f"{cc}.parquet"
    out = PARSED_DIR / f"{cc}_en.parquet"
    if not pq.exists():
        log.warning("%s: no parsed parquet — skipping", cc)
        return
    if out.exists() and not force:
        log.info("%s: sidecar exists (%s) — skipping", cc, out.name)
        return

    t0 = time.time()
    ids = pd.read_parquet(pq, columns=["id"])
    en = read_english_texts(cc)
    if not en:
        log.warning("%s: no English texts under %s — run download_parlamint_en.py first",
                    cc, RAW_EN_DIR / cc)
        return

    ids["text_en"] = ids["id"].map(en)
    matched = int(ids["text_en"].notna().sum())
    total = len(ids)
    ids["text_en"] = ids["text_en"].fillna("")  # keep all rows; missing -> "" (UI falls back)
    ids.to_parquet(out, index=False)
    log.info("%s: %s/%s indexed speeches translated (%.1f%%) -> %s (%.1f s)",
             cc, f"{matched:,}", f"{total:,}", 100 * matched / max(total, 1), out.name, time.time() - t0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--countries", help="comma-separated codes (default: all with a parsed parquet)")
    p.add_argument("--force", action="store_true", help="rebuild sidecars that already exist")
    a = p.parse_args()

    if a.countries:
        countries = [c.strip() for c in a.countries.split(",") if c.strip()]
    elif PARSED_DIR.exists():
        countries = sorted(f.stem for f in PARSED_DIR.glob("*.parquet") if not f.stem.endswith("_en"))
    else:
        countries = []
    if not countries:
        log.warning("No parsed parquet under %s — run parse_parlamint.py first.", PARSED_DIR)
        return

    for cc in countries:
        build_country(cc, force=a.force)


if __name__ == "__main__":
    main()
