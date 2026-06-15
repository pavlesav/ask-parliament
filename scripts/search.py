"""Search the speech index from the command line (no LLM involved).

For eyeballing retrieval quality before wiring up generation.

Examples:
    python scripts/search.py "immigration during the refugee crisis" --k 5
    python scripts/search.py "pension reform" --year-from 2000 --year-to 2005
    python scripts/search.py "renewable energy" --domain Energy --domain Environment
    python scripts/search.py "budget deficit" --party FPÖ --k 3
"""
from __future__ import annotations

import argparse
import textwrap

from ask_parliament.retrieval import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Top-k semantic search over parliamentary speeches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:")[1],
    )
    parser.add_argument("query", help="natural-language query (any language BGE-m3 knows)")
    parser.add_argument("--k", type=int, default=8, help="number of results (default 8)")
    parser.add_argument("--country", help="country code filter, e.g. AT")
    parser.add_argument("--year-from", type=int, help="earliest year (inclusive)")
    parser.add_argument("--year-to", type=int, help="latest year (inclusive)")
    parser.add_argument("--domain", action="append",
                        help="CAP domain filter, repeatable (e.g. --domain Health)")
    parser.add_argument("--party", help="party code filter, e.g. SPÖ")
    parser.add_argument("--chars", type=int, default=400,
                        help="speech preview length (default 400; 0 = full text)")
    args = parser.parse_args()

    retriever = Retriever()
    hits = retriever.search(
        args.query,
        k=args.k,
        countries=[args.country] if args.country else None,
        year_from=args.year_from,
        year_to=args.year_to,
        cap_domains=args.domain,
        party=args.party,
    )

    t = retriever.last_timings
    print(f"\n{len(hits)} results  (embed {t['embed_s']:.2f}s, search {t['search_s']:.3f}s)\n")
    for rank, h in enumerate(hits, 1):
        status = f", {h.party_status}" if h.party_status not in ("-", "") else ""
        print(f"[{rank}] sim={h.similarity:.3f}  {h.speaker} ({h.party}{status})  "
              f"{h.date}  {h.country}")
        print(f"    {h.cap_domain} | {h.topic_name} | {h.segment_id}")
        text = h.text if args.chars == 0 else h.text[: args.chars]
        ellipsis = "…" if 0 < args.chars < len(h.text) else ""
        print(textwrap.indent(textwrap.fill(text + ellipsis, width=100), "    "))
        print()


if __name__ == "__main__":
    main()
