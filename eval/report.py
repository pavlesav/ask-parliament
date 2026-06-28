"""Render the latest eval runs into a single human-readable Markdown report.

Reads the JSON documents written under eval/results/ (and the flat history.jsonl) and
produces eval/results/REPORT.md: the method-comparison table with confidence intervals,
the cross-lingual breadth figure, a per-topic-domain breakdown, the ablation deltas
(what reranking and query transformation each add over plain), the generation-quality
and refusal numbers, and a short regression trend from the run history.

Usage:
    python eval/report.py                 # build REPORT.md from the newest results
    python eval/report.py --print         # also echo it to stdout
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ask_parliament.config import REPO_ROOT  # noqa: E402
from evallib.persistence import RESULTS_DIR, load_history  # noqa: E402

REPORT_PATH = RESULTS_DIR / "REPORT.md"
METHOD_ORDER = ["plain", "rerank", "transform", "agentic"]


def _latest(kind: str) -> dict | None:
    """The most recent full result document of a given kind, or None."""
    files = sorted(RESULTS_DIR.glob(f"*_{kind}.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text(encoding="utf-8"))


def _ci(s: dict) -> str:
    return f"{s['mean']:.3f} <sub>[{s['ci_low']:.3f}, {s['ci_high']:.3f}]</sub>"


def _provenance_line(doc: dict) -> str:
    g = doc.get("git", {})
    c = doc.get("config", {})
    dirty = " (dirty)" if g.get("dirty") else ""
    return (f"_{doc.get('timestamp', '?')} · commit `{g.get('sha', '?')}`{dirty} on "
            f"`{g.get('branch', '?')}` · device `{c.get('device', '?')}` · "
            f"embeddings `{c.get('embedding_model', '?')}`_")


def render_retrieval(doc: dict) -> str:
    summ = doc["summary"]
    methods = [m for m in METHOD_ORDER if m in summ] + [m for m in summ if m not in METHOD_ORDER]
    out = ["## Retrieval — golden set (pattern-judged)\n", _provenance_line(doc), ""]
    out.append(f"k = {doc.get('k', '?')} · {doc.get('n_items', '?')} golden items · "
               "value = mean [95% bootstrap CI]\n")
    out.append("| method | MRR | success@k | precision@k | nDCG@k | langs reached | latency |")
    out.append("|---|---|---|---|---|---|---|")
    for m in methods:
        r = summ[m]
        out.append(f"| `{m}` | {_ci(r['mrr'])} | {_ci(r['success_at_k'])} | "
                   f"{_ci(r['precision_at_k'])} | {_ci(r['ndcg_at_k'])} | "
                   f"{r['languages_reached_pan']['mean']:.1f} | {r['mean_latency_s']:.2f}s |")
    out.append("")

    # Ablation deltas vs plain.
    if "plain" in summ:
        base = summ["plain"]
        out.append("**What each stage adds (Δ vs `plain`, nDCG@k):**\n")
        for m in methods:
            if m == "plain":
                continue
            d = summ[m]["ndcg_at_k"]["mean"] - base["ndcg_at_k"]["mean"]
            out.append(f"- `{m}`: {d:+.3f}")
        out.append("")

    # Per-topic-domain success@k, per method.
    per_item = doc.get("per_item", {})
    if per_item:
        domains: dict[str, dict[str, list]] = {}
        for m, rows in per_item.items():
            for row in rows:
                domains.setdefault(row["topic_domain"], {}).setdefault(m, []).append(row["success_at_k"])
        out.append("**success@k by topic domain:**\n")
        out.append("| domain | " + " | ".join(f"`{m}`" for m in methods) + " |")
        out.append("|" + "---|" * (len(methods) + 1))
        for dom in sorted(domains):
            cells = []
            for m in methods:
                vals = domains[dom].get(m, [])
                cells.append(f"{sum(vals)/len(vals):.2f}" if vals else "–")
            out.append(f"| {dom} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def render_generation(doc: dict) -> str:
    summ = doc.get("summary", {})
    out = ["## Generation — LLM-as-judge\n", _provenance_line(doc), ""]
    out.append(f"retrieval = `{doc.get('retrieval','?')}` · generator = `{doc.get('gen_model','?')}` · "
               f"judge = `{doc.get('judge_model','?')}`\n")
    if "quality" in summ:
        s = summ["quality"]["scores"]
        out.append(f"**In-corpus answer quality** (1–5, mean [95% CI], n={summ['quality']['n_judged']}):\n")
        out.append("| groundedness | citation validity | answer relevance |")
        out.append("|---|---|---|")
        out.append("| " + " | ".join(
            f"{s[k]['mean']:.2f} [{s[k]['ci_low']:.2f}, {s[k]['ci_high']:.2f}]"
            for k in ("groundedness", "citation_validity", "answer_relevance")) + " |")
        out.append("")
    if "refusal" in summ:
        r = summ["refusal"]
        ok = r.get("ok_rate", r.get("refusal_rate", 0))
        out.append(f"**Out-of-corpus handling** (n={r['n_judged']}; correct = refused *or* "
                   f"grounded in a cited speech): handled **{ok*100:.0f}%** "
                   f"(refused {r.get('refusal_rate',0)*100:.0f}% + grounded "
                   f"{r.get('grounded_rate',0)*100:.0f}%) · hallucinated "
                   f"**{r['hallucination_rate']*100:.0f}%** (lower is better).\n")
    return "\n".join(out)


def render_history() -> str:
    hist = load_history()
    if not hist:
        return ""
    out = ["## Run history (regression trend)\n",
           "| when | kind | commit | headline |", "|---|---|---|---|"]
    for h in hist[-12:]:
        sha = h.get("git", {}).get("sha", "?")
        head = ""
        if h.get("kind") == "retrieval_golden" and "metrics" in h:
            ag = h["metrics"].get("agentic") or next(iter(h["metrics"].values()), {})
            head = f"agentic nDCG {ag.get('ndcg_at_k', 0):.3f}, MRR {ag.get('mrr', 0):.3f}"
        elif h.get("kind") == "generation":
            s = h.get("summary", {})
            q = s.get("quality", {}).get("scores", {}).get("groundedness", {})
            rf = s.get("refusal", {})
            halluc = rf.get("hallucination_rate", 0) * 100
            head = f"grounded {q.get('mean', 0):.2f}, halluc {halluc:.0f}%"
        out.append(f"| {h.get('timestamp','?')} | {h.get('kind','?')} | `{sha}` | {head} |")
    return "\n".join(out) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--print", action="store_true", dest="echo", help="also print the report to stdout")
    a = p.parse_args()

    parts = ["# Ask Parliament — evaluation report\n",
             "_Generated by `eval/report.py` from the newest runs in `eval/results/`._\n"]
    ret = _latest("retrieval_golden")
    gen = _latest("generation")
    if ret:
        parts.append(render_retrieval(ret))
    if gen:
        parts.append(render_generation(gen))
    hist = render_history()
    if hist:
        parts.append(hist)
    if not ret and not gen:
        parts.append("_No results yet — run `eval/retrieval_eval.py` or `eval/generation_eval.py` first._\n")

    report = "\n".join(parts)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH.relative_to(REPO_ROOT)}")
    if a.echo:
        print("\n" + report)


if __name__ == "__main__":
    main()
