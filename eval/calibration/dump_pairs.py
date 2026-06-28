"""Sample query–speech pairs for human (or strong-model) calibration of the pattern judge.

The golden-set retrieval eval trusts a deterministic *pattern* judge to decide whether a
retrieved speech is relevant. This dumps a balanced sample of (query, speech) pairs with
the full speech text, the pattern judge's verdict, and the reranker score — so a careful
reader can assign a gold label and we can measure how often the cheap pattern judge agrees
with considered judgment (and in which direction it errs).

Balance matters: for each sampled query we take some speeches the pattern marks RELEVANT
(to catch false positives) and some it marks IRRELEVANT but that the reranker ranked highly
(to catch false negatives). All retrieval is within the item's own scope, so the only thing
being judged is topical relevance of the text.

Pure retrieval + reranking — no API. Writes eval/calibration/pairs.jsonl.

Usage:  QDRANT_URL=... python eval/calibration/dump_pairs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # eval/ on path

from ask_parliament.config import REPO_ROOT  # noqa: E402
from ask_parliament.rerank import Reranker  # noqa: E402
from ask_parliament.retrieval import Retriever  # noqa: E402
from evallib.goldenset import load_golden, year_window  # noqa: E402
from evallib.judge import compile_item, relevance  # noqa: E402

OUT = Path(__file__).resolve().parent / "pairs.jsonl"
# A spread of domains and corpus languages (English-heavy and native-heavy topics).
QUERIES = ["ukraine_invasion_2022", "greek_debt_crisis", "migration_asylum", "nuclear_energy",
           "pension_reform", "healthcare_system", "abortion", "taxation", "corruption",
           "housing", "rule_of_law", "fishing_policy"]
POOL = 40
SNIPPET = 700


def main() -> None:
    by_id = {it["id"]: it for it in load_golden()}
    base, rr = Retriever(), Reranker()
    _ = rr.model
    pairs = []
    for qid in QUERIES:
        item = by_id.get(qid)
        if item is None:
            print(f"  (skip {qid}: not in golden set)")
            continue
        ci = compile_item(item)
        yf, yt = year_window(item)
        ranked = rr.rerank(item["question"],
                           base.search(item["question"], k=POOL,
                                       countries=item.get("countries"), year_from=yf, year_to=yt))
        rel, irrel = [], []
        for h in ranked:
            g = relevance(h.text, h.country, h.date, ci)
            (rel if g > 0 else irrel).append((h, g))
        # 2 highest-reranked pattern-relevant + 2 highest-reranked pattern-irrelevant
        chosen = rel[:2] + irrel[:2]
        for h, g in chosen:
            pairs.append({
                "pair_id": f"{qid}::{h.id[-12:]}",
                "query_id": qid,
                "question": item["question"],
                "topic_domain": item.get("topic_domain", "-"),
                "speech_country": h.country,
                "speech_date": h.date,
                "rerank_score": round(float(h.rerank_score), 3),
                "pattern_grade": g,            # 0 = pattern says irrelevant, >0 = relevant
                "pattern_label": int(g > 0),
                "speech_text": h.text[:SNIPPET].replace("\n", " ").strip(),
            })
    OUT.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs) + "\n",
                   encoding="utf-8")
    npos = sum(p["pattern_label"] for p in pairs)
    print(f"Wrote {len(pairs)} pairs ({npos} pattern-relevant, {len(pairs)-npos} pattern-irrelevant) "
          f"-> {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
