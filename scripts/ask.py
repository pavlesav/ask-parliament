"""Ask a grounded question from the command line (retrieval + generation).

Wires Qdrant retrieval to grounded generation so you can eyeball end-to-end
answers over the scaled multi-country corpus from the terminal.

Examples:
    python scripts/ask.py "What did MPs say about the 2015 refugee crisis?" --country AT
    python scripts/ask.py "How was mandatory vaccination debated?" --domain Health --year-from 2020
    python scripts/ask.py "Positions on pension reform?" --k 10 --model claude-sonnet-4-6
"""
from __future__ import annotations

import argparse
import textwrap

from ask_parliament.config import GENERATION_MODEL
from ask_parliament.generation import generate_answer
from ask_parliament.retrieval import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a grounded question over parliamentary speeches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:")[1],
    )
    parser.add_argument("question", help="natural-language question")
    parser.add_argument("--k", type=int, default=8, help="speeches to retrieve (default 8)")
    parser.add_argument("--agentic", action="store_true",
                        help="query transformation + reranking + self-correcting retrieval")
    parser.add_argument("--country", help="country code filter, e.g. AT")
    parser.add_argument("--year-from", type=int, help="earliest year (inclusive)")
    parser.add_argument("--year-to", type=int, help="latest year (inclusive)")
    parser.add_argument("--domain", action="append", help="CAP domain filter, repeatable")
    parser.add_argument("--party", help="party code filter, e.g. SPÖ")
    parser.add_argument("--model", default=GENERATION_MODEL, help=f"model (default {GENERATION_MODEL})")
    args = parser.parse_args()

    base = Retriever()
    if args.agentic:
        from ask_parliament.agentic import AgenticRetriever
        searcher = AgenticRetriever(base)
    else:
        searcher = base
    hits = searcher.search(
        args.question, k=args.k,
        countries=[args.country] if args.country else None,
        year_from=args.year_from, year_to=args.year_to,
        cap_domains=args.domain, party=args.party,
    )
    if args.agentic:
        tr = searcher.last_trace
        print(f"[agentic] {tr['n_variants']} query variants, pool {tr['pool_size']}, "
              f"top rerank {tr['top_rerank']}, corrected={tr['corrected']} (rounds {tr['rounds']})")
    result = generate_answer(args.question, hits, model=args.model)

    print("\n" + "=" * 100)
    print(textwrap.fill(result.answer, width=100))
    print("=" * 100)

    print("\nSources:")
    for i, h in enumerate(result.sources, 1):
        status = f", {h.party_status}" if h.party_status not in ("-", "") else ""
        score = f"sim={h.similarity:.3f}"
        if h.rerank_score is not None:
            score += f" rerank={h.rerank_score:.3f}"
        print(f"  [{i}] {score}  {h.speaker} ({h.party}{status})  "
              f"{h.date}  {h.country}  {h.cap_domain}")

    retrieval_s = sum(searcher.last_timings.values())
    print(f"\n{result.model} | in {result.input_tokens} tok, out {result.output_tokens} tok "
          f"| retrieval {retrieval_s:.2f}s, generate {result.latency_s:.2f}s")


if __name__ == "__main__":
    main()
