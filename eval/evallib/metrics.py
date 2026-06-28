"""Ranking metrics for retrieval evaluation — pure functions, no heavy deps.

Everything operates on a **graded relevance list in rank order**: `grades[i]` is the
relevance of the speech retrieved at rank `i+1` (1-based). Binary judging gives 0/1;
the helpers also accept graded relevance (e.g. 0/1/2) for nDCG.

Which metric tells you what, in this project:

- **MRR** — how high the *first* relevant speech lands. Good single-number summary;
  the known-item eval lives or dies by it.
- **success@k / hit@k** — did *any* relevant speech reach the top k. The user-facing
  question ("did we surface something useful?").
- **precision@k** — what fraction of the top k are on-topic. The quality of the page;
  the metric reranking is supposed to move.
- **recall@k** — fraction of all relevant speeches found in the top k. Meaningful for
  *known-item* (one relevant doc, so recall@k == hit@k); for broad-topic golden items,
  where thousands of speeches match, it's reported against a capped relevant pool and
  read with care (see eval/README.md).
- **nDCG@k** — rewards putting relevant speeches higher, with graded relevance support.
- **average precision / MAP** — precision averaged over the ranks where relevant docs
  appear; a fuller picture than precision@k for multi-relevant items.

Aggregation helpers add a **bootstrap confidence interval** so we report a number with
an honest error bar instead of a point estimate from a noisy synthetic sample.
"""
from __future__ import annotations

import math
import random
from statistics import mean


def reciprocal_rank(grades: list[float]) -> float:
    """1 / rank of the first relevant (grade > 0) item; 0 if none."""
    for i, g in enumerate(grades, start=1):
        if g > 0:
            return 1.0 / i
    return 0.0


def success_at_k(grades: list[float], k: int) -> float:
    """1.0 if any relevant item is in the top k, else 0.0 (a.k.a. hit@k)."""
    return 1.0 if any(g > 0 for g in grades[:k]) else 0.0


# hit@k is the same thing under a more familiar name.
hit_at_k = success_at_k


def precision_at_k(grades: list[float], k: int) -> float:
    """Fraction of the top k that are relevant. Denominator is k (not len), so a
    short result list is penalised for not filling k — that is the honest reading
    of 'precision of the first k slots'."""
    if k <= 0:
        return 0.0
    topk = grades[:k]
    return sum(1 for g in topk if g > 0) / k


def recall_at_k(grades: list[float], k: int, n_relevant: int) -> float:
    """Fraction of all relevant items retrieved in the top k.

    `n_relevant` is the size of the relevant universe (1 for known-item; a capped
    estimate for broad topics). Returns 0 when nothing is relevant, to avoid 0/0.
    """
    if n_relevant <= 0:
        return 0.0
    found = sum(1 for g in grades[:k] if g > 0)
    return found / n_relevant


def dcg_at_k(grades: list[float], k: int) -> float:
    """Discounted cumulative gain with the standard log2(rank+1) discount."""
    return sum(g / math.log2(i + 1) for i, g in enumerate(grades[:k], start=1))


def ndcg_at_k(grades: list[float], k: int, n_relevant: int | None = None) -> float:
    """nDCG@k. The ideal ranking puts the highest grades first; when `n_relevant`
    is given we assume that many relevant items exist (each graded 1.0 unless the
    observed grades are higher), so the ideal DCG isn't capped by what we happened
    to retrieve. Falls back to sorting the observed grades when `n_relevant` is None.
    """
    dcg = dcg_at_k(grades, k)
    if dcg == 0.0:
        return 0.0
    if n_relevant is None:
        ideal = sorted(grades, reverse=True)
    else:
        # Ideal: the best grades we saw, padded with unit-relevant docs up to
        # n_relevant, then truncated to k by dcg_at_k.
        observed = sorted((g for g in grades if g > 0), reverse=True)
        pad = max(0, n_relevant - len(observed))
        ideal = observed + [1.0] * pad
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def average_precision(grades: list[float], n_relevant: int | None = None) -> float:
    """Average of precision@i over the ranks i where a relevant item occurs,
    normalised by the number of relevant items (capped at `n_relevant` if given).
    With one relevant item this reduces to reciprocal rank.
    """
    hits = 0
    score = 0.0
    for i, g in enumerate(grades, start=1):
        if g > 0:
            hits += 1
            score += hits / i
    total = hits if n_relevant is None else min(n_relevant, max(hits, 1))
    if n_relevant is not None:
        total = n_relevant
    return score / total if total > 0 else 0.0


# --- aggregation + uncertainty --------------------------------------------------

def bootstrap_ci(
    values: list[float], confidence: float = 0.95, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of `values`. Returns (lo, hi).

    Synthetic eval samples are small and noisy; a CI keeps us honest about whether
    a method difference is real or sampling jitter. Deterministic given `seed`.
    """
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(mean(sample))
    means.sort()
    lo_idx = int((1 - confidence) / 2 * n_boot)
    hi_idx = int((1 + confidence) / 2 * n_boot) - 1
    return (means[lo_idx], means[max(lo_idx, hi_idx)])


def summarize(per_query: list[float], **ci_kwargs) -> dict:
    """Mean + bootstrap CI + n for a list of per-query metric values."""
    if not per_query:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    m = mean(per_query)
    lo, hi = bootstrap_ci(per_query, **ci_kwargs)
    return {"mean": m, "ci_low": lo, "ci_high": hi, "n": len(per_query)}
