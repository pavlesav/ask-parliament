"""Retrieval evaluation over the golden set: recall@k and MRR.

Ground truth, per question, is defined lexically and embedding-independently:
the speeches matching a distinctive phrase within the anchor debate's date window
(the criterion lives in golden_set.jsonl, so it is fully auditable). We then run
the BGE-m3 semantic retriever and measure how well it surfaces those speeches.

Two conditions are reported:

  unfiltered  — query the whole 27-year corpus with no metadata filter. A hard
                test: the same topic recurs on other dates, and those equally
                relevant speeches (not in our single-debate judged set) displace
                judged ones, so this is a conservative lower bound.
  year-scoped — also pass the debate's year as a filter, mirroring how the app is
                actually used (the sidebar year slider). Isolates the effect of
                metadata filtering on finding a specific debate.

Usage:
    python eval/run_eval.py            # k=20
    python eval/run_eval.py --k 30
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ask_parliament.retrieval import Retriever

GOLDEN_PATH = Path(__file__).parent / "golden_set.jsonl"


def load_golden() -> list[dict]:
    lines = GOLDEN_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def load_corpus(collection) -> tuple[list[str], list[str], list[dict]]:
    """Pull all (id, document, metadata) from the index, paginated."""
    ids, docs, metas = [], [], []
    for offset in range(0, collection.count(), 10_000):
        g = collection.get(include=["documents", "metadatas"], limit=10_000, offset=offset)
        ids += g["ids"]
        docs += g["documents"]
        metas += g["metadatas"]
    return ids, docs, metas


def relevant_ids(item: dict, ids, docs, metas) -> set[str]:
    """Ground-truth relevant set: phrase match within the anchor date window."""
    rx = re.compile(item["pattern"], re.I)
    d0, d1 = item["date_from"], item["date_to"]
    return {
        ids[i]
        for i in range(len(ids))
        if d0 <= metas[i]["date"] <= d1 and rx.search(docs[i])
    }


def metrics_for(hits, relevant: set[str], k: int) -> dict:
    """recall@5/10/k, reciprocal rank of first hit, hit@10 for one query."""
    ranks = [i + 1 for i, h in enumerate(hits) if h.id in relevant]
    first = ranks[0] if ranks else None
    n = len(relevant)
    return {
        "first": first,
        "rr": 1.0 / first if first else 0.0,
        "r5": sum(r <= 5 for r in ranks) / n if n else 0.0,
        "r10": sum(r <= 10 for r in ranks) / n if n else 0.0,
        "rk": len(ranks) / n if n else 0.0,
        "hit10": 1.0 if first and first <= 10 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval eval: recall@k and MRR")
    parser.add_argument("--k", type=int, default=20, help="retrieval depth (default 20)")
    args = parser.parse_args()
    k = args.k

    retriever = Retriever()
    ids, docs, metas = load_corpus(retriever.collection)
    golden = load_golden()

    rows = []
    sums = {c: {"mrr": 0.0, "r10": 0.0, "rk": 0.0, "hit10": 0.0} for c in ("uf", "yr")}
    for item in golden:
        relevant = relevant_ids(item, ids, docs, metas)
        year = int(item["date_from"][:4])

        uf = metrics_for(retriever.search(item["question"], k=k), relevant, k)
        yr = metrics_for(
            retriever.search(item["question"], k=k, year_from=year, year_to=year),
            relevant, k,
        )
        rows.append((item["id"], len(relevant), uf, yr))
        for c, m in (("uf", uf), ("yr", yr)):
            sums[c]["mrr"] += m["rr"]
            sums[c]["r10"] += m["r10"]
            sums[c]["rk"] += m["rk"]
            sums[c]["hit10"] += m["hit10"]

    n = len(golden)
    means = {c: {key: v / n for key, v in d.items()} for c, d in sums.items()}

    # --- table: unfiltered vs year-scoped ---
    print(f"\nRetrieval eval — {n} questions, k={k}\n")
    h = (f"{'question':<24}{'|R|':>5} | {'MRR':>5}{'R@10':>6}{'R@'+str(k):>6}{'hit10':>6}"
         f"  | {'MRR':>5}{'R@10':>6}{'R@'+str(k):>6}{'hit10':>6}")
    print(f"{'':<29} |  --- unfiltered ---    |  --- year-scoped ---")
    print(h)
    print("-" * len(h))
    for qid, nr, uf, yr in rows:
        print(f"{qid:<24}{nr:>5} | {uf['rr']:>5.2f}{uf['r10']:>6.2f}{uf['rk']:>6.2f}{uf['hit10']:>6.0f}"
              f"  | {yr['rr']:>5.2f}{yr['r10']:>6.2f}{yr['rk']:>6.2f}{yr['hit10']:>6.0f}")
    print("-" * len(h))
    print(f"{'MEAN':<24}{'':>5} | {means['uf']['mrr']:>5.2f}{means['uf']['r10']:>6.2f}"
          f"{means['uf']['rk']:>6.2f}{means['uf']['hit10']:>6.2f}"
          f"  | {means['yr']['mrr']:>5.2f}{means['yr']['r10']:>6.2f}"
          f"{means['yr']['rk']:>6.2f}{means['yr']['hit10']:>6.2f}")
    print(f"\nunfiltered:  MRR={means['uf']['mrr']:.3f} · recall@10={means['uf']['r10']:.3f} "
          f"· recall@{k}={means['uf']['rk']:.3f} · hit@10={means['uf']['hit10']:.3f}")
    print(f"year-scoped: MRR={means['yr']['mrr']:.3f} · recall@10={means['yr']['r10']:.3f} "
          f"· recall@{k}={means['yr']['rk']:.3f} · hit@10={means['yr']['hit10']:.3f}")
    print("\n|R| = size of the judged relevant set (one specific debate). MRR = reciprocal "
          "rank of\nthe first relevant speech. hit10 = a relevant speech in the top 10.")


if __name__ == "__main__":
    main()
