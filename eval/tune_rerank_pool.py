"""Sweep the reranker candidate-pool depth and report what it buys — no API calls.

`rerank` over-fetches N vector candidates and lets the cross-encoder pick the top k.
Deeper N gives the cross-encoder more to choose from (potentially higher precision/MRR)
at the cost of more cross-encoder forward passes. This sweeps N over the golden set,
reusing one loaded embedding model + one loaded reranker, and prints the metric curve so
RERANK_CANDIDATES can be set from evidence rather than the conventional 60.

Pure retrieval + reranking (no query transform), so it needs Qdrant but not ANTHROPIC_API_KEY.

Usage:
    QDRANT_URL=... python eval/tune_rerank_pool.py --pools 60,100,150,200
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ask_parliament.rerank import Reranker  # noqa: E402
from ask_parliament.retrieval import Retriever  # noqa: E402
from evallib import metrics as M  # noqa: E402
from evallib.goldenset import load_golden, year_window  # noqa: E402
from evallib.judge import compile_item, grade_hits  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pools", default="60,100,150,200", help="comma-separated candidate-pool sizes")
    p.add_argument("--k", type=int, default=10)
    a = p.parse_args()
    pools = [int(x) for x in a.pools.split(",")]

    items = load_golden()
    compiled = {it["id"]: compile_item(it) for it in items}
    base = Retriever()
    reranker = Reranker()

    # Fetch the deepest pool once per item, then rerank prefixes of it — so we pay the
    # vector search at max depth a single time and the cross-encoder is the only thing
    # that varies. (Reranking the prefix == reranking that pool, since rerank is per-item.)
    max_pool = max(pools)
    print(f"Sweeping rerank pool over {pools} (k={a.k}), {len(items)} golden items…\n")
    candidates: dict[str, list] = {}
    t0 = time.time()
    for it in items:
        yf, yt = year_window(it)
        candidates[it["id"]] = base.search(it["question"], k=max_pool,
                                           countries=it.get("countries"), year_from=yf, year_to=yt)
    print(f"  fetched candidates in {time.time()-t0:.0f}s\n")

    print(f"{'pool':>6}{'MRR':>10}{'prec@k':>10}{'nDCG@k':>10}{'success@k':>12}")
    print("-" * 48)
    for pool in pools:
        rr = pr = nd = sc = 0.0
        for it in items:
            cand = candidates[it["id"]][:pool]
            ranked = reranker.rerank(it["question"], cand)[:a.k]
            grades = grade_hits(ranked, compiled[it["id"]])
            n_rel = max(1, int(it.get("corpus_matches", 1)))
            rr += M.reciprocal_rank(grades)
            pr += M.precision_at_k(grades, a.k)
            nd += M.ndcg_at_k(grades, a.k, n_relevant=n_rel)
            sc += M.success_at_k(grades, a.k)
        n = len(items)
        print(f"{pool:>6}{rr/n:>10.3f}{pr/n:>10.3f}{nd/n:>10.3f}{sc/n:>12.3f}")
    print("\n(no-API rerank-only sweep; pick the pool where the curve plateaus)")


if __name__ == "__main__":
    main()
