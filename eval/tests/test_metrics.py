"""Unit tests for the ranking metrics — pure maths, no Qdrant/API needed.

Run: python -m pytest eval/tests -q   (or eval/tests/run_offline.py without pytest)
"""
from evallib import metrics as M


def test_reciprocal_rank():
    assert M.reciprocal_rank([1, 0, 0]) == 1.0
    assert M.reciprocal_rank([0, 1, 0]) == 0.5
    assert M.reciprocal_rank([0, 0, 1]) == 1 / 3
    assert M.reciprocal_rank([0, 0, 0]) == 0.0


def test_success_and_hit_alias():
    assert M.success_at_k([0, 1, 0], 1) == 0.0
    assert M.success_at_k([0, 1, 0], 2) == 1.0
    assert M.hit_at_k is M.success_at_k


def test_precision_at_k():
    assert M.precision_at_k([1, 1, 0, 0], 2) == 1.0
    assert M.precision_at_k([1, 0, 1, 0], 4) == 0.5
    # short list still divides by k (honest "first k slots" reading)
    assert M.precision_at_k([1], 4) == 0.25
    assert M.precision_at_k([], 4) == 0.0


def test_recall_at_k():
    assert M.recall_at_k([1, 0, 0], 10, n_relevant=1) == 1.0  # known-item
    assert M.recall_at_k([1, 1, 0, 0], 4, n_relevant=4) == 0.5
    assert M.recall_at_k([0, 0], 2, n_relevant=0) == 0.0


def test_ndcg_monotonic_in_rank():
    # Same relevant doc, higher rank -> higher nDCG.
    high = M.ndcg_at_k([1, 0, 0, 0], 4, n_relevant=1)
    low = M.ndcg_at_k([0, 0, 0, 1], 4, n_relevant=1)
    assert high == 1.0
    assert 0.0 < low < high


def test_ndcg_graded_prefers_strong_first():
    # Grade-2 doc above a grade-1 doc is the ideal ordering -> nDCG 1.0.
    assert M.ndcg_at_k([2, 1, 0], 3, n_relevant=2) == 1.0
    # Swapped order scores strictly less.
    assert M.ndcg_at_k([1, 2, 0], 3, n_relevant=2) < 1.0


def test_average_precision_reduces_to_rr_for_single_relevant():
    grades = [0, 1, 0, 0]
    assert M.average_precision(grades, n_relevant=1) == M.reciprocal_rank(grades)


def test_average_precision_multi():
    # relevant at ranks 1 and 3: (1/1 + 2/3) / 2
    ap = M.average_precision([1, 0, 1, 0], n_relevant=2)
    assert abs(ap - ((1.0 + 2 / 3) / 2)) < 1e-9


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    vals = [0.0, 0.5, 1.0, 0.5, 0.0, 1.0, 0.5, 0.5]
    lo, hi = M.bootstrap_ci(vals, seed=0)
    m = sum(vals) / len(vals)
    assert lo <= m <= hi
    assert M.bootstrap_ci(vals, seed=0) == M.bootstrap_ci(vals, seed=0)  # reproducible


def test_summarize_shape():
    s = M.summarize([1.0, 0.0, 1.0])
    assert s["n"] == 3 and 0.0 <= s["ci_low"] <= s["mean"] <= s["ci_high"] <= 1.0
    empty = M.summarize([])
    assert empty["n"] == 0 and empty["mean"] == 0.0
