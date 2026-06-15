"""Parse downloaded ParlaMint corpora into tidy speech parquet files (resumable).

Reads data/parlamint/raw/{CC}/ (produced by download_parlamint.py) and writes
data/parlamint/parsed/{CC}.parquet — one row per Regular speech (>= min chars),
native text plus uniform English metadata and the per-speech CAP topic. CPU-only
and fast, so it is safe to run on a laptop. Countries already parsed are skipped.

Examples:
  python scripts/scale/parse_parlamint.py                 # all downloaded countries
  python scripts/scale/parse_parlamint.py --countries LV
"""
from __future__ import annotations

import argparse
import logging
import time

from ask_parliament.config import PARSED_DIR, RAW_DIR
from ask_parliament.parlamint import parse_country

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("parse_parlamint")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--countries", help="comma-separated codes (default: all under raw/)")
    p.add_argument("--force", action="store_true", help="re-parse even if the parquet exists")
    a = p.parse_args()

    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    if a.countries:
        countries = [c.strip() for c in a.countries.split(",") if c.strip()]
    elif RAW_DIR.exists():
        countries = sorted(d.name for d in RAW_DIR.iterdir() if d.is_dir() and not d.name.startswith("_"))
    else:
        countries = []
    if not countries:
        log.warning("No countries under %s — run download_parlamint.py first.", RAW_DIR)
        return

    for cc in countries:
        out = PARSED_DIR / f"{cc}.parquet"
        if out.exists() and not a.force:
            log.info("%s: already parsed (%s) — skipping", cc, out.name)
            continue
        t0 = time.time()
        df = parse_country(cc)
        if df.empty:
            log.warning("%s: no rows parsed — nothing written", cc)
            continue
        df.to_parquet(out, index=False)
        log.info("%s: wrote %s rows -> %s (%.1f s)", cc, f"{len(df):,}", out.name, time.time() - t0)


if __name__ == "__main__":
    main()
