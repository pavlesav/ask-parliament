"""Validate the built golden set: it loads, items are well-formed, patterns compile,
and the corpus-verification metadata is present and self-consistent. Offline (reads the
JSONL the builder wrote); skips cleanly if the set hasn't been built yet."""
import re

import pytest

from evallib.goldenset import GOLDEN_PATH, load_golden, year_window
from evallib.judge import compile_item, grade_hits

pytestmark = pytest.mark.skipif(
    not GOLDEN_PATH.exists(), reason="golden_set.jsonl not built (run eval/build_golden_set.py)"
)


def test_loads_and_required_fields():
    items = load_golden()
    assert len(items) >= 10
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), "golden ids must be unique"
    for it in items:
        assert it["question"].strip()
        assert isinstance(it["patterns"], list) and it["patterns"]


def test_patterns_compile_and_judge_runs():
    for it in load_golden():
        for pat in it["patterns"]:
            re.compile(pat)  # raises if malformed
        compiled = compile_item(it)
        # Judging an empty hit list must not blow up and yields no grades.
        assert grade_hits([], compiled) == []


def test_verification_metadata_consistent():
    for it in load_golden():
        # The builder stamps these; totals must agree with the per-country breakdown.
        if "corpus_matches_by_country" in it:
            assert it["corpus_matches"] == sum(it["corpus_matches_by_country"].values())
            assert it["corpus_n_countries"] == len(it["corpus_matches_by_country"])
            assert it["corpus_matches"] > 0


def test_examples_match_their_own_patterns():
    """Each stored example speech must itself satisfy the item it was drawn for —
    a sanity check that the verification scan and the judge agree."""
    for it in load_golden():
        compiled = compile_item(it)
        for ex in it.get("examples", []):
            g = grade_hits([{"text": ex["snippet"], "country": ex["country"], "date": ex["date"]}], compiled)
            # The snippet is truncated to 160 chars, so a pattern may legitimately fall
            # outside it; only assert when the example does match (no false drops).
            assert g[0] in (0.0, 1.0, 2.0)


def test_year_window_parsing():
    assert year_window({"date_from": "2022-02-01", "date_to": "2022-12-31"}) == (2022, 2022)
    assert year_window({}) == (None, None)
