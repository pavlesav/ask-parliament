"""Measure how well the deterministic pattern judge agrees with gold labels.

Reads the sampled pairs (`pairs.jsonl`) and the gold labels (`labels.jsonl`, one
`{"pair_id", "gold"}` per line, gold in {0,1}) and reports, treating the gold label as
truth and the pattern judge as the predictor:

  accuracy · precision · recall · F1 · Cohen's kappa · confusion matrix · every disagreement

This quantifies how much to trust the cheap pattern judge that the whole golden-set
retrieval eval rests on. Offline, no Qdrant/API.

Usage:  python eval/calibration/score.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str) -> list[dict]:
    p = HERE / name
    if not p.exists():
        raise FileNotFoundError(f"{p} missing")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _recompute_labels(pairs: dict) -> None:
    """Recompute each pair's pattern_label from the CURRENT golden patterns applied to the
    stored speech text — so a pattern fix can be re-scored against the same gold labels."""
    sys.path.insert(0, str(HERE.parent))  # eval/ on path
    from evallib.goldenset import load_golden
    from evallib.judge import compile_item, relevance
    compiled = {it["id"]: compile_item(it) for it in load_golden()}
    for p in pairs.values():
        ci = compiled.get(p["query_id"])
        if ci is None:
            continue
        g = relevance(p["speech_text"], p["speech_country"], p["speech_date"], ci)
        p["pattern_label"] = int(g > 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recompute", action="store_true",
                    help="recompute pattern verdicts from current golden patterns (for before/after a fix)")
    a = ap.parse_args()
    pairs = {p["pair_id"]: p for p in _load("pairs.jsonl")}
    if a.recompute:
        _recompute_labels(pairs)
        print("(pattern verdicts recomputed from current golden patterns)\n")
    gold = {g["pair_id"]: int(g["gold"]) for g in _load("labels.jsonl")}

    missing = set(pairs) - set(gold)
    if missing:
        print(f"warning: {len(missing)} pairs not yet labelled: {sorted(missing)[:5]}…")

    tp = fp = fn = tn = 0
    disagreements = []
    for pid, g in gold.items():
        pr = pairs[pid]["pattern_label"]
        if pr == 1 and g == 1:
            tp += 1
        elif pr == 1 and g == 0:
            fp += 1
            disagreements.append((pid, "pattern=REL gold=irrel", pairs[pid]))
        elif pr == 0 and g == 1:
            fn += 1
            disagreements.append((pid, "pattern=irrel gold=REL", pairs[pid]))
        else:
            tn += 1

    n = tp + fp + fn + tn
    if n == 0:
        print("no labelled pairs"); return
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    # Cohen's kappa
    po = acc
    p_pred1, p_true1 = (tp + fp) / n, (tp + fn) / n
    pe = p_pred1 * p_true1 + (1 - p_pred1) * (1 - p_true1)
    kappa = (po - pe) / (1 - pe) if (1 - pe) else 1.0

    print(f"Pattern judge vs gold labels — {n} pairs\n")
    print(f"  accuracy   {acc:.3f}")
    print(f"  precision  {prec:.3f}   (of pattern-RELEVANT calls, how many gold agrees)")
    print(f"  recall     {rec:.3f}   (of gold-RELEVANT pairs, how many the pattern caught)")
    print(f"  F1         {f1:.3f}")
    print(f"  Cohen kappa {kappa:.3f}")
    print(f"\n  confusion (rows=pattern, cols=gold):")
    print(f"            gold=REL  gold=irrel")
    print(f"  pat=REL     {tp:>5}     {fp:>6}")
    print(f"  pat=irrel   {fn:>5}     {tn:>6}")

    if disagreements:
        print(f"\n  {len(disagreements)} disagreement(s):")
        for pid, how, p in disagreements:
            print(f"   - {pid} [{how}] rerank={p['rerank_score']} | {p['speech_text'][:90]}")
    else:
        print("\n  perfect agreement.")


if __name__ == "__main__":
    main()
