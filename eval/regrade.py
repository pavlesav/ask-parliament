"""Re-grade a saved retrieval run with the *current* golden patterns — offline.

Judging is the part of the eval most likely to change (you tighten a pattern, add a
language variant). Re-running retrieval to test that change costs ~20 minutes of GPU +
Qdrant. But the retrieval result files store the `hit_ids` each method returned, so we
can look those speeches up in the parquets and re-score with the new patterns in seconds
— no index needed. This turns judge iteration from a coffee break into a keystroke.

It prints the old vs. new aggregate per method so you can see exactly what a pattern
change did, and (with --save) writes a fresh result document so the change is recorded.

Usage:
    python eval/regrade.py                         # re-grade the newest retrieval run
    python eval/regrade.py --result eval/results/<file>.json --save
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ask_parliament.config import PARSED_DIR, REPO_ROOT  # noqa: E402
from evallib import metrics as M  # noqa: E402
from evallib import persistence, provenance  # noqa: E402
from evallib.goldenset import load_golden  # noqa: E402
from evallib.judge import compile_item, relevance  # noqa: E402
from evallib.persistence import RESULTS_DIR  # noqa: E402


def _newest_retrieval() -> Path:
    files = sorted(RESULTS_DIR.glob("*_retrieval_golden.json"))
    if not files:
        raise FileNotFoundError("no retrieval result files in eval/results/")
    return files[-1]


def _needed_ids(doc: dict) -> dict[str, set[str]]:
    """speech ids to look up, grouped by country (from saved hit_ids/hit_countries)."""
    by_country: dict[str, set[str]] = {}
    for rows in doc.get("per_item", {}).values():
        for row in rows:
            for sid, cc in zip(row.get("hit_ids", []), row.get("hit_countries", [])):
                by_country.setdefault(cc, set()).add(sid)
    return by_country


def _load_lookup(by_country: dict[str, set[str]]) -> dict[str, dict]:
    """id -> {text, country, date} for exactly the needed ids, streaming per parquet."""
    lookup: dict[str, dict] = {}
    for cc, ids in by_country.items():
        pq = PARSED_DIR / f"{cc}.parquet"
        if not pq.exists():
            continue
        df = pd.read_parquet(pq, columns=["id", "text", "country", "date"])
        sub = df[df["id"].isin(ids)]
        for _, r in sub.iterrows():
            lookup[r["id"]] = {"text": r["text"], "country": r["country"], "date": r["date"]}
    return lookup


def _regrade_rows(rows, compiled_by_id, lookup, k):
    out = []
    for row in rows:
        it = compiled_by_id.get(row["id"])
        if it is None:
            out.append(row)  # item no longer in golden set; keep as-is
            continue
        grades = []
        for sid in row.get("hit_ids", []):
            rec = lookup.get(sid)
            grades.append(relevance(rec["text"], rec["country"], rec["date"], it) if rec else 0.0)
        # Same nDCG ideal as retrieval_eval: the true relevant total (corpus_matches),
        # so the ideal top-k is "all k slots relevant" for these broad topics.
        n_rel = max(1, int(it.raw.get("corpus_matches", 1)))
        out.append({
            **row,
            "rr": M.reciprocal_rank(grades),
            "success_at_k": M.success_at_k(grades, k),
            "precision_at_k": M.precision_at_k(grades, k),
            "ndcg_at_k": M.ndcg_at_k(grades, k, n_relevant=n_rel),
            "grades": grades,
        })
    return out


def _summ(rows):
    f = lambda n: M.summarize([r[n] for r in rows])  # noqa: E731
    return {"mrr": f("rr"), "success_at_k": f("success_at_k"),
            "precision_at_k": f("precision_at_k"), "ndcg_at_k": f("ndcg_at_k")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result", help="retrieval result JSON to re-grade (default: newest)")
    p.add_argument("--save", action="store_true", help="write a new retrieval_golden result doc")
    a = p.parse_args()

    path = Path(a.result) if a.result else _newest_retrieval()
    doc = json.loads(path.read_text(encoding="utf-8"))
    k = doc.get("k", 10)
    print(f"Re-grading {path.name} (k={k}) with current golden patterns…\n")

    compiled = {it["id"]: compile_item(it) for it in load_golden()}
    lookup = _load_lookup(_needed_ids(doc))
    print(f"Looked up {len(lookup)} speeches from the parquets.\n")

    new_summary = {}
    print(f"{'method':<11}{'metric':<14}{'old':>10}{'new':>10}{'Δ':>9}")
    print("-" * 54)
    for m, rows in doc.get("per_item", {}).items():
        regraded = _regrade_rows(rows, compiled, lookup, k)
        new_summary[m] = _summ(regraded)
        for metric in ("mrr", "success_at_k", "precision_at_k", "ndcg_at_k"):
            old = doc["summary"][m][metric]["mean"]
            new = new_summary[m][metric]["mean"]
            print(f"{m:<11}{metric:<14}{old:>10.3f}{new:>10.3f}{new-old:>+9.3f}")
        print()

    if a.save:
        out = {**provenance.stamp({"eval": "retrieval_golden", "k": k,
                                   "methods": list(new_summary), "n_items": doc.get("n_items"),
                                   "regraded_from": path.name}),
               "summary": new_summary, "per_item": doc["per_item"]}
        headline = {**provenance.stamp(), "k": k,
                    "metrics": {m: {kk: new_summary[m][kk]["mean"]
                                    for kk in ("mrr", "success_at_k", "precision_at_k", "ndcg_at_k")}
                                for m in new_summary}}
        sp = persistence.save_run("retrieval_golden", out, headline)
        print(f"Saved re-graded run -> {sp.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
