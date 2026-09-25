"""Render the README figures (light + dark variants) into docs/figures/.

  corpus_by_parliament_{light,dark}.png  speeches per parliament, from the parsed parquets
  retrieval_ablation_{light,dark}.png    golden-set retrieval, mean + 95% CI per pipeline stage,
                                         from a retrieval_golden result in eval/results/

Needs the parsed corpus (scripts/scale/parse_parlamint.py) and a golden-set run
(eval/retrieval_eval.py) that covers all four methods. matplotlib comes with the
dev extra: `pip install -e .[dev]`.

Usage:
    python scripts/make_figures.py
    python scripts/make_figures.py --result eval/results/<file>_retrieval_golden.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from ask_parliament.config import PARLAMINT_COUNTRIES, PARSED_DIR, REPO_ROOT  # noqa: E402

OUT_DIR = REPO_ROOT / "docs" / "figures"
RESULTS_DIR = REPO_ROOT / "eval" / "results"

# Chart chrome per theme; one series, so one data colour.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "data": "#2a78d6"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "data": "#3987e5"},
}

NAMES = {
    "AT": "Austria", "BA": "Bosnia and Herzegovina", "BE": "Belgium", "BG": "Bulgaria",
    "CZ": "Czechia", "DK": "Denmark", "EE": "Estonia", "ES": "Spain", "ES-CT": "Catalonia",
    "ES-GA": "Galicia", "ES-PV": "Basque Country", "FI": "Finland", "FR": "France",
    "GB": "United Kingdom", "GR": "Greece", "HR": "Croatia", "HU": "Hungary", "IS": "Iceland",
    "IT": "Italy", "LV": "Latvia", "NL": "Netherlands", "NO": "Norway", "PL": "Poland",
    "PT": "Portugal", "RS": "Serbia", "SE": "Sweden", "SI": "Slovenia", "TR": "Türkiye",
    "UA": "Ukraine",
}

METHODS = [  # (result key, label) in pipeline order
    ("plain", "Dense vector (plain)"),
    ("rerank", "+ cross-encoder rerank"),
    ("transform", "+ query transform (no rerank)"),
    ("agentic", "Full agentic (app default)"),
]
METRICS = [("mrr", "MRR"), ("precision_at_k", "precision@10"), ("ndcg_at_k", "nDCG@10")]


def _style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=9)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def corpus_counts() -> list[tuple[str, int]]:
    rows = [(cc, pq.ParquetFile(PARSED_DIR / f"{cc}.parquet").metadata.num_rows)
            for cc in PARLAMINT_COUNTRIES]
    return sorted(rows, key=lambda r: r[1])


def plot_corpus(counts, theme: str) -> Path:
    t = THEMES[theme]
    total = sum(n for _, n in counts)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150, facecolor=t["surface"])
    _style(ax, t)
    labels = [f"{NAMES[cc]} ({cc})" for cc, _ in counts]
    ax.barh(labels, [n for _, n in counts], height=0.72, color=t["data"])
    ax.tick_params(axis="y", colors=t["ink2"])
    ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:.0f}k")
    ax.margins(y=0.01)
    fig.text(0.02, 0.975, "Indexed speeches per parliament", color=t["ink"], fontsize=13,
             fontweight="bold", va="top")
    fig.text(0.02, 0.945, f"ParlaMint 5.0, {len(counts)} parliaments, {total:,} speeches "
             "(regular speakers, ≥300 characters)", color=t["ink2"], fontsize=9, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT_DIR / f"corpus_by_parliament_{theme}.png"
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)
    return out


def newest_full_run() -> Path:
    for f in sorted(RESULTS_DIR.glob("*_retrieval_golden.json"), reverse=True):
        doc = json.loads(f.read_text(encoding="utf-8"))
        if all(m in doc["summary"] for m, _ in METHODS):
            return f
    raise FileNotFoundError("no retrieval_golden run in eval/results/ covers all four methods")


def plot_ablation(doc: dict, theme: str) -> Path:
    t = THEMES[theme]
    s = doc["summary"]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(10, 3.4), dpi=150, sharey=True,
                             facecolor=t["surface"])
    ys = list(range(len(METHODS)))[::-1]  # plain on top
    for ax, (key, title) in zip(axes, METRICS):
        _style(ax, t)
        for y, (m, _) in zip(ys, METHODS):
            v = s[m][key]
            ax.plot([v["ci_low"], v["ci_high"]], [y, y], color=t["data"], linewidth=2,
                    solid_capstyle="round", alpha=0.45)
            ax.plot(v["mean"], y, "o", color=t["data"], markersize=8,
                    markeredgecolor=t["surface"], markeredgewidth=2)
            ax.annotate(f"{v['mean']:.2f}", (v["mean"], y), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=8.5, color=t["ink2"])
        ax.set_title(title, loc="left", color=t["ink"], fontsize=10, fontweight="bold")
        ax.set_xlim(0.55, 1.0)
        ax.set_ylim(-0.6, len(METHODS) - 0.3)
    axes[0].set_yticks(ys, [label for _, label in METHODS])
    axes[0].tick_params(axis="y", colors=t["ink2"], labelsize=9)
    fig.suptitle(f"Golden-set retrieval by pipeline stage (n={doc['n_items']} topics, "
                 f"k={doc['k']}; dot = mean, bar = 95% bootstrap CI)",
                 x=0.01, ha="left", color=t["ink"], fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / f"retrieval_ablation_{theme}.png"
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result", help="retrieval_golden result JSON (default: newest with all four methods)")
    a = p.parse_args()

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # quiet font fallback
    plt.rcParams["font.family"] = ["Segoe UI", "Arial", "DejaVu Sans"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = corpus_counts()
    result = Path(a.result) if a.result else newest_full_run()
    doc = json.loads(result.read_text(encoding="utf-8"))
    print(f"Ablation figure from {result.name}")
    for theme in THEMES:
        for out in (plot_corpus(counts, theme), plot_ablation(doc, theme)):
            print(f"Wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
