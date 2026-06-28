"""Filter-correctness check — assert metadata filters actually constrain retrieval.

A retrieval system whose country/year/party/domain filters silently leak would corrupt
every scoped query (and every scoped eval item). This is the cheap guard: issue the same
query with each filter and assert every returned speech honours it. Needs the live index.

Usage:  QDRANT_URL=... python eval/filter_check.py
Exit code is non-zero if any check fails, so it can gate CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ask_parliament.config import REPO_ROOT  # noqa: E402
from ask_parliament.retrieval import Retriever  # noqa: E402

QUERY = "economic policy and public spending"  # broad on purpose, to fill k under every filter
K = 20


def _report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    r = Retriever()
    facets = r.facets()
    checks: list[bool] = []

    # Country filter.
    cc = "AT" if "AT" in facets.countries else facets.countries[0]
    hits = r.search(QUERY, k=K, countries=[cc])
    bad = {h.country for h in hits} - {cc}
    checks.append(_report(f"country={cc}", not bad and len(hits) > 0,
                          f"{len(hits)} hits, stray countries={bad or 'none'}"))

    # Year window.
    y0, y1 = 2015, 2018
    hits = r.search(QUERY, k=K, year_from=y0, year_to=y1)
    out = [h.year for h in hits if not (y0 <= h.year <= y1)]
    checks.append(_report(f"year {y0}-{y1}", not out and len(hits) > 0,
                          f"{len(hits)} hits, out-of-range years={out or 'none'}"))

    # CAP domain.
    if facets.cap_domains:
        dom = facets.cap_domains[0]
        hits = r.search(QUERY, k=K, cap_domains=[dom])
        bad = {h.cap_domain for h in hits} - {dom}
        checks.append(_report(f"cap_domain={dom}", not bad,
                              f"{len(hits)} hits, stray domains={bad or 'none'}"))

    # Party.
    if facets.parties:
        party = facets.parties[0]
        hits = r.search(QUERY, k=K, party=party)
        bad = {h.party for h in hits} - {party}
        checks.append(_report(f"party={party}", not bad,
                              f"{len(hits)} hits, stray parties={bad or 'none'}"))

    # Combined country+year, to catch filters that don't compose.
    hits = r.search(QUERY, k=K, countries=[cc], year_from=y0, year_to=y1)
    bad = [(h.country, h.year) for h in hits if h.country != cc or not (y0 <= h.year <= y1)]
    checks.append(_report(f"country={cc} AND year {y0}-{y1}", not bad,
                          f"{len(hits)} hits, violations={bad or 'none'}"))

    ok = all(checks)
    print(f"\n{'ALL FILTERS OK' if ok else 'SOME FILTERS FAILED'} "
          f"({sum(checks)}/{len(checks)} passed)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
