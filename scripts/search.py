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

from ask_parliament.hybrid import HybridRetriever
from ask_parliament.retrieval import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Top-k search over parliamentary speeches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:")[1],
    )
    parser.add_argument("query", help="natural-language query (any language BGE-m3 knows)")
    parser.add_argument("--k", type=int, default=8, help="number of results (default 8)")
    parser.add_argument("--hybrid", action="store_true",
                        help="fuse BM25 keyword search with vector search (RRF)")
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
    searcher = HybridRetriever(retriever) if args.hybrid else retriever
    hits = searcher.search(
        args.query,
        k=args.k,
        countries=[args.country] if args.country else None,
        year_from=args.year_from,
        year_to=args.year_to,
        cap_domains=args.domain,
        party=args.party,
    )

    t = searcher.last_timings
    timing = ", ".join(f"{name} {secs:.2f}s" for name, secs in t.items())
    mode = "hybrid" if args.hybrid else "semantic"
    print(f"\n{len(hits)} results [{mode}]  ({timing})\n")
    for rank, h in enumerate(hits, 1):
        status = f", {h.party_status}" if h.party_status not in ("-", "") else ""
        # In hybrid mode, similarity 0.0 means the dense side didn't surface it —
        # it's here on the strength of the BM25 keyword match alone.
        sim = "keyword-only" if (args.hybrid and h.similarity == 0.0) else f"sim={h.similarity:.3f}"
        print(f"[{rank}] {sim}  {h.speaker} ({h.party}{status})  {h.date}  {h.country}")
        print(f"    {h.cap_domain} | {h.topic_name} | {h.segment_id}")
        text = h.text if args.chars == 0 else h.text[: args.chars]
        ellipsis = "…" if 0 < args.chars < len(h.text) else ""
        print(textwrap.indent(textwrap.fill(text + ellipsis, width=100), "    "))
        print()


if __name__ == "__main__":
    main()
