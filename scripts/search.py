"""Search the scaled multi-country speech index from the command line (no LLM).

For eyeballing retrieval quality before wiring up generation.

Examples:
    python scripts/search.py "immigration during the refugee crisis" --k 5
    python scripts/search.py "renewable energy" --country GR --year-from 2015
    python scripts/search.py "pension reform" --domain Macroeconomics --domain "Social Welfare"
    python scripts/search.py "budget deficit" --country AT --k 3
"""
from __future__ import annotations

import argparse
import textwrap

from ask_parliament.retrieval import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Top-k search over parliamentary speeches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:")[1],
    )
    parser.add_argument("query", help="natural-language query (any language BGE-m3 knows)")
    parser.add_argument("--k", type=int, default=8, help="number of results (default 8)")
    parser.add_argument("--agentic", action="store_true",
                        help="query transformation + reranking + self-correcting retrieval")
    parser.add_argument("--country", help="country code filter, e.g. AT")
    parser.add_argument("--year-from", type=int, help="earliest year (inclusive)")
    parser.add_argument("--year-to", type=int, help="latest year (inclusive)")
    parser.add_argument("--domain", action="append",
                        help="CAP domain filter, repeatable (e.g. --domain Health)")
    parser.add_argument("--party", help="party code filter, e.g. SPÖ")
    parser.add_argument("--chars", type=int, default=400,
                        help="speech preview length (default 400; 0 = full text)")
    args = parser.parse_args()

    base = Retriever()
    if args.agentic:
        from ask_parliament.agentic import AgenticRetriever
        searcher = AgenticRetriever(base)
    else:
        searcher = base
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
    mode = "agentic" if args.agentic else "semantic"
    print(f"\n{len(hits)} results [{mode}]  ({timing})")
    if args.agentic:
        tr = searcher.last_trace
        print(f"  trace: {tr['n_variants']} query variants, pool {tr['pool_size']}, "
              f"top rerank {tr['top_rerank']}, corrected={tr['corrected']} (rounds {tr['rounds']})")
        if tr["sub_queries"]:
            print("  sub-queries: " + " | ".join(tr["sub_queries"]))
    print()
    for rank, h in enumerate(hits, 1):
        status = f", {h.party_status}" if h.party_status not in ("-", "") else ""
        score = f"sim={h.similarity:.3f}" + (f" rerank={h.rerank_score:.3f}" if h.rerank_score is not None else "")
        print(f"[{rank}] {score}  {h.speaker} ({h.party}{status})  {h.date}  {h.country}")
        print(f"    {h.cap_domain} | {h.segment_id}")
        text = h.text if args.chars == 0 else h.text[: args.chars]
        ellipsis = "…" if 0 < args.chars < len(h.text) else ""
        print(textwrap.indent(textwrap.fill(text + ellipsis, width=100), "    "))
        print()


if __name__ == "__main__":
    main()
