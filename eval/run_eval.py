"""Multi-country retrieval evaluation — synthetic known-item retrieval.

For a sample of speeches drawn from across the parliaments, an LLM writes a natural
question that the speech answers; we then retrieve over the *whole* corpus and check
whether (and at what rank) that source speech comes back. This measures retrieval
quality across all 29 countries without any hand-labeling, and lets us compare plain
vector search against the agentic pipeline (query transform + rerank + self-correct)
head to head.

Known-item caveat: the question is derived from the target speech, so this is an
optimistic, *relative* measure — fair for comparing methods and catching regressions,
not an absolute recall figure. Other speeches may also be relevant but aren't credited.

Usage:
    python eval/run_eval.py                          # 2 speeches/country, both methods, k=10
    python eval/run_eval.py --per-country 3 --method agentic
    python eval/run_eval.py --countries AT,GB,FR,GR --k 20

Needs the index served (QDRANT_URL) and ANTHROPIC_API_KEY in .env.
"""
from __future__ import annotations

import argparse
import time

import anthropic
import pandas as pd
from dotenv import load_dotenv

from ask_parliament.config import GENERATION_MODEL, PARSED_DIR, REPO_ROOT
from ask_parliament.retrieval import Retriever

load_dotenv(REPO_ROOT / ".env")

_QUESTION_SYSTEM = (
    "You are given one parliamentary speech. Write ONE natural, specific question in "
    "English that a person could ask and that THIS speech answers — about its topic and "
    "stance, not its wording. Output only the question, nothing else."
)


def sample_speeches(countries, per_country, seed, min_chars, max_chars) -> list[dict]:
    """Draw `per_country` speeches from each country's parsed parquet (reproducible)."""
    rows: list[dict] = []
    for cc in countries:
        pq = PARSED_DIR / f"{cc}.parquet"
        if not pq.exists():
            print(f"  (skip {cc}: no parquet)")
            continue
        df = pd.read_parquet(pq, columns=["id", "text", "country", "date"])
        df = df[df["text"].str.len().between(min_chars, max_chars)]
        if df.empty:
            continue
        rows += df.sample(min(per_country, len(df)), random_state=seed).to_dict("records")
    return rows


def gen_question(client: anthropic.Anthropic, text: str, model: str) -> str:
    resp = client.messages.create(
        model=model, max_tokens=100, system=_QUESTION_SYSTEM,
        messages=[{"role": "user", "content": text[:3000]}],
    )
    return next((b.text for b in resp.content if b.type == "text"), "").strip()


def rank_of(hits, target_id: str) -> int | None:
    for i, h in enumerate(hits, 1):
        if h.id == target_id:
            return i
    return None


def evaluate(searcher, samples, questions, k: int) -> dict:
    """Mean reciprocal rank + hit@1/5/10 over the samples (relevant set = the source)."""
    rr = hit1 = hit5 = hit10 = found = 0.0
    per_country: dict[str, list[float]] = {}
    for s, q in zip(samples, questions):
        rank = rank_of(searcher.search(q, k=k), s["id"])
        r = 1.0 / rank if rank else 0.0
        rr += r
        hit1 += rank == 1
        hit5 += bool(rank and rank <= 5)
        hit10 += bool(rank and rank <= 10)
        found += bool(rank)
        per_country.setdefault(s["country"], []).append(r)
    n = len(samples)
    return {
        "mrr": rr / n, "hit1": hit1 / n, "hit5": hit5 / n, "hit10": hit10 / n,
        "found": found / n, "n": n,
        "per_country": {c: sum(v) / len(v) for c, v in sorted(per_country.items())},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-country", type=int, default=2, help="speeches sampled per country (default 2)")
    p.add_argument("--countries", help="comma-separated codes (default: all in the index)")
    p.add_argument("--k", type=int, default=10, help="retrieval depth (default 10)")
    p.add_argument("--method", choices=["plain", "agentic", "both"], default="both")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-chars", type=int, default=400)
    p.add_argument("--max-chars", type=int, default=2000)
    a = p.parse_args()

    retriever = Retriever()
    countries = ([c.strip() for c in a.countries.split(",")] if a.countries
                 else retriever.facets().countries)

    print(f"Sampling {a.per_country}/country across {len(countries)} parliaments…")
    samples = sample_speeches(countries, a.per_country, a.seed, a.min_chars, a.max_chars)
    if not samples:
        print("No speeches sampled — is the parsed corpus present?")
        return

    print(f"Generating {len(samples)} questions with {GENERATION_MODEL}…")
    client = anthropic.Anthropic(max_retries=2)
    t0 = time.time()
    questions = [gen_question(client, s["text"], GENERATION_MODEL) for s in samples]
    print(f"  ({time.time() - t0:.0f}s)\n")

    methods = ["plain", "agentic"] if a.method == "both" else [a.method]
    searchers = {"plain": retriever}
    if "agentic" in methods:
        from ask_parliament.agentic import AgenticRetriever
        searchers["agentic"] = AgenticRetriever(retriever)

    results = {}
    for m in methods:
        print(f"Evaluating [{m}] over the full corpus (k={a.k})…")
        t0 = time.time()
        results[m] = evaluate(searchers[m], samples, questions, a.k)
        print(f"  ({time.time() - t0:.0f}s)")

    print(f"\nKnown-item retrieval — {len(samples)} speeches, {len(countries)} countries, k={a.k}\n")
    print(f"{'method':<10}{'MRR':>7}{'hit@1':>8}{'hit@5':>8}{'hit@10':>8}{'found':>8}")
    print("-" * 49)
    for m in methods:
        r = results[m]
        print(f"{m:<10}{r['mrr']:>7.3f}{r['hit1']:>8.2f}{r['hit5']:>8.2f}{r['hit10']:>8.2f}{r['found']:>8.2f}")
    print("\nMRR = mean reciprocal rank of the source speech. hit@k = source in the top k. "
          "found = source\nanywhere in the top k. Higher is better (max 1.0).")


if __name__ == "__main__":
    main()
