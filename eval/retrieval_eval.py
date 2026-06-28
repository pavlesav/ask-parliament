"""Golden-set retrieval evaluation — pattern-judged, with ablations and CIs.

The complement to the known-item eval (`run_eval.py`). Instead of crediting one source
speech, it judges relevance by **topic pattern** (see evallib/judge.py): any retrieved
speech that actually discusses the item's topic counts, in any of the 29 languages. That
turns the eval into a real recall/precision measure and — because each item runs through
every ablation variant — shows what query transformation and reranking each contribute.

For each golden item we retrieve within the item's declared scope (its country whitelist
and date window, mirroring how a user would ask), grade the ranked hits, and compute:

    MRR · success@k (hit@k) · precision@k · nDCG@k

aggregated with bootstrap 95% CIs, plus a cross-lingual breadth figure (how many distinct
parliaments the relevant hits span) for the pan-European items.

Usage:
    python eval/retrieval_eval.py                                  # all methods, k=10
    python eval/retrieval_eval.py --methods plain,agentic --k 20
    python eval/retrieval_eval.py --limit 5 --methods plain        # quick smoke run
    python eval/retrieval_eval.py --no-save

Needs the index served (QDRANT_URL) and, for transform/agentic, ANTHROPIC_API_KEY in .env.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ask_parliament.config import REPO_ROOT  # noqa: E402
from ask_parliament.retrieval import Retriever  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from evallib import metrics as M  # noqa: E402
from evallib import persistence, provenance  # noqa: E402
from evallib.goldenset import load_golden, year_window  # noqa: E402
from evallib.judge import compile_item, grade_hits  # noqa: E402
from evallib.searchers import build_searchers  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

ALL_METHODS = ["plain", "rerank", "transform", "agentic"]


def eval_item(searcher, item: dict, compiled, k: int) -> dict:
    """Run one item through one searcher; return its per-query metric record."""
    yf, yt = year_window(item)
    t0 = time.time()
    hits = searcher.search(
        item["question"], k=k,
        countries=item.get("countries"), year_from=yf, year_to=yt,
    )
    latency = time.time() - t0
    grades = grade_hits(hits, compiled)
    n_rel = max(1, int(item.get("corpus_matches", 1)))
    rel_countries = {h.country for h, g in zip(hits, grades) if g > 0}
    return {
        "id": item["id"],
        "topic_domain": item.get("topic_domain", "-"),
        "is_pan_european": item.get("countries") is None,
        "rr": M.reciprocal_rank(grades),
        "success_at_k": M.success_at_k(grades, k),
        "precision_at_k": M.precision_at_k(grades, k),
        "ndcg_at_k": M.ndcg_at_k(grades, k, n_relevant=n_rel),
        "n_relevant_in_topk": sum(1 for g in grades if g > 0),
        "languages_reached": len(rel_countries),
        "latency_s": latency,
        "grades": grades,
        # retrieved provenance — lets a later judge tweak be re-graded offline
        # (look the ids up in the parquets) without re-querying Qdrant.
        "hit_ids": [h.id for h in hits],
        "hit_countries": [h.country for h in hits],
    }


def aggregate(rows: list[dict], k: int) -> dict:
    """Mean + bootstrap CI per metric, plus pan-European cross-lingual breadth."""
    def col(name):
        return [r[name] for r in rows]
    pan = [r for r in rows if r["is_pan_european"]]
    return {
        "mrr": M.summarize(col("rr")),
        "success_at_k": M.summarize(col("success_at_k")),
        "precision_at_k": M.summarize(col("precision_at_k")),
        "ndcg_at_k": M.summarize(col("ndcg_at_k")),
        "languages_reached_pan": M.summarize([r["languages_reached"] for r in pan]),
        "mean_latency_s": sum(col("latency_s")) / len(rows) if rows else 0.0,
        "k": k,
        "n_items": len(rows),
    }


def fmt_ci(s: dict) -> str:
    return f"{s['mean']:.3f} [{s['ci_low']:.3f},{s['ci_high']:.3f}]"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", default=",".join(ALL_METHODS),
                   help=f"comma-separated subset of {ALL_METHODS} (default all)")
    p.add_argument("--k", type=int, default=10, help="retrieval depth (default 10)")
    p.add_argument("--limit", type=int, help="evaluate only the first N golden items (smoke test)")
    p.add_argument("--no-save", action="store_true", help="don't write to eval/results/")
    a = p.parse_args()

    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    items = load_golden()
    if a.limit:
        items = items[: a.limit]
    compiled = {it["id"]: compile_item(it) for it in items}

    base = Retriever()
    searchers = build_searchers(methods, base)
    print(f"Golden-set retrieval eval — {len(items)} items, methods={methods}, k={a.k}\n")

    results = {}
    per_method_rows = {}
    for m in methods:
        print(f"[{m}] evaluating…", flush=True)
        t0 = time.time()
        rows = [eval_item(searchers[m], it, compiled[it["id"]], a.k) for it in items]
        per_method_rows[m] = rows
        results[m] = aggregate(rows, a.k)
        print(f"  done in {time.time() - t0:.0f}s")

    # --- console report ---
    print(f"\n{'method':<11}{'MRR':>22}{'success@k':>22}{'prec@k':>22}{'nDCG@k':>22}")
    print("-" * 99)
    for m in methods:
        r = results[m]
        print(f"{m:<11}{fmt_ci(r['mrr']):>22}{fmt_ci(r['success_at_k']):>22}"
              f"{fmt_ci(r['precision_at_k']):>22}{fmt_ci(r['ndcg_at_k']):>22}")
    print("\n(value = mean [95% bootstrap CI] over golden items)")
    print(f"\nCross-lingual breadth (distinct parliaments among relevant top-{a.k} hits, "
          f"pan-European items):")
    for m in methods:
        print(f"  {m:<11}{fmt_ci(results[m]['languages_reached_pan'])}  "
              f"(mean latency {results[m]['mean_latency_s']:.2f}s)")

    if not a.no_save:
        doc = {
            **provenance.stamp({"eval": "retrieval_golden", "k": a.k, "methods": methods,
                                "n_items": len(items)}),
            "summary": results,
            "per_item": {m: per_method_rows[m] for m in methods},
        }
        headline = {**provenance.stamp(), "k": a.k,
                    "metrics": {m: {kk: results[m][kk]["mean"]
                                    for kk in ("mrr", "success_at_k", "precision_at_k", "ndcg_at_k")}
                                for m in methods}}
        path = persistence.save_run("retrieval_golden", doc, headline)
        print(f"\nSaved -> {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
