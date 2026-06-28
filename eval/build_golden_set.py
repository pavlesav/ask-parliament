"""Build (and verify) the golden set from the parsed corpus.

A golden item is only worth keeping if speeches that match it actually exist in the
index. So this script doesn't just write the curated topics out — it *measures* each
one against every parsed parquet:

    for each parsed corpus (streamed one country at a time, to stay light on RAM):
        for each seed topic that applies to this country:
            count speeches whose text matches the patterns (within any date window),
            accumulate total matches, matches-per-country, and a few example ids.

Then it keeps the well-attested items (>= --min-matches, and >= --min-countries for the
pan-European ones) and writes them to eval/golden_set.jsonl with the verification fields
baked in. It prints a coverage table so you can see which patterns are genuinely
multilingual and prune/extend the seed list from evidence.

Usage:
    python eval/build_golden_set.py                 # verify all seeds, write golden_set.jsonl
    python eval/build_golden_set.py --min-matches 20 --min-countries 4
    python eval/build_golden_set.py --report-only   # print coverage, don't write the file
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import evallib` works

from ask_parliament.config import PARLAMINT_COUNTRIES, PARSED_DIR, REPO_ROOT  # noqa: E402
from evallib.seed_topics import SEED_TOPICS  # noqa: E402

GOLDEN_PATH = REPO_ROOT / "eval" / "golden_set.jsonl"
EXAMPLES_PER_ITEM = 3


def _item_mask(df: pd.DataFrame, text_lower: pd.Series, item: dict) -> pd.Series:
    """Combined mask for one item over one country's df: ALL pattern groups match
    (each pattern is an OR-regex) and the date window holds."""
    mask = pd.Series(True, index=df.index)
    for pat in item["patterns"]:
        mask &= text_lower.str.contains(pat.lower(), regex=True, na=False)
    if item.get("date_from"):
        mask &= df["date"] >= item["date_from"]
    if item.get("date_to"):
        mask &= df["date"] <= item["date_to"]
    return mask


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min-matches", type=int, default=10, help="drop items with fewer corpus matches (default 10)")
    p.add_argument("--min-countries", type=int, default=2,
                   help="for pan-European items (countries=None), require hits in >= this many parliaments (default 2)")
    p.add_argument("--report-only", action="store_true", help="print the coverage table; don't write golden_set.jsonl")
    a = p.parse_args()

    # Per-item accumulators.
    acc = {it["id"]: {"by_country": {}, "examples": []} for it in SEED_TOPICS}

    print("Scanning parsed corpora (streamed per country)…")
    for cc in PARLAMINT_COUNTRIES:
        pq = PARSED_DIR / f"{cc}.parquet"
        if not pq.exists():
            print(f"  (skip {cc}: no parquet)")
            continue
        df = pd.read_parquet(pq, columns=["id", "text", "country", "date"])
        text_lower = df["text"].str.lower()
        applied = 0
        for item in SEED_TOPICS:
            if item.get("countries") and cc not in item["countries"]:
                continue
            applied += 1
            mask = _item_mask(df, text_lower, item)
            n = int(mask.sum())
            if n:
                acc[item["id"]]["by_country"][cc] = n
                ex = acc[item["id"]]["examples"]
                if len(ex) < EXAMPLES_PER_ITEM:
                    for _, row in df[mask].head(EXAMPLES_PER_ITEM - len(ex)).iterrows():
                        ex.append({
                            "id": row["id"], "country": row["country"], "date": row["date"],
                            "snippet": str(row["text"])[:160].replace("\n", " "),
                        })
        print(f"  {cc}: {len(df):>7,} speeches  ({applied} items checked)")
        del df, text_lower

    # Assemble verified items, apply thresholds.
    print(f"\nVerifying {len(SEED_TOPICS)} seed topics…\n")
    kept, dropped = [], []
    print(f"{'id':<26}{'matches':>9}{'countries':>11}  status")
    print("-" * 72)
    for item in SEED_TOPICS:
        by_country = dict(sorted(acc[item["id"]]["by_country"].items(), key=lambda kv: -kv[1]))
        total = sum(by_country.values())
        n_ctry = len(by_country)
        verified = {**item,
                    "corpus_matches": total,
                    "corpus_matches_by_country": by_country,
                    "corpus_n_countries": n_ctry,
                    "examples": acc[item["id"]]["examples"]}
        is_pan = item.get("countries") is None
        ok = total >= a.min_matches and (not is_pan or n_ctry >= a.min_countries)
        (kept if ok else dropped).append(verified)
        status = "keep" if ok else "DROP (under threshold)"
        print(f"{item['id']:<26}{total:>9,}{n_ctry:>11}  {status}")

    print(f"\n{len(kept)} kept, {len(dropped)} dropped.")
    print("\nCross-lingual coverage (pan-European items) — top countries per topic:")
    for item in kept:
        if item.get("countries") is None:
            top = list(item["corpus_matches_by_country"].items())[:8]
            print(f"  {item['id']:<26} [{item['corpus_n_countries']} ctry] " +
                  ", ".join(f"{c}:{n}" for c, n in top))

    if a.report_only:
        print("\n--report-only: golden_set.jsonl not written.")
        return
    GOLDEN_PATH.write_text(
        "\n".join(json.dumps(it, ensure_ascii=False) for it in kept) + "\n", encoding="utf-8")
    print(f"\nWrote {len(kept)} items -> {GOLDEN_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
