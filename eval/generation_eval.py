"""End-to-end generation quality eval — LLM-as-judge over the full RAG pipeline.

Retrieval metrics stop at "did we fetch the right speeches". This measures the part the
user actually reads: the grounded, cited answer. It runs real questions through the
production path (retrieve -> generate) and has a stronger model judge each answer.

Two halves:

  quality (in-corpus):  the golden-set questions -> retrieve -> generate -> judge for
      groundedness, citation validity, and answer relevance (evallib/llm_judge).
  refusal (out-of-corpus):  eval/refusal_set.jsonl questions whose answers are NOT in any
      parliamentary speech -> the system should DECLINE, not confabulate. We measure the
      refusal rate and (its mirror) the hallucination rate.

The generator defaults to the cheap model (Haiku) and the judge to the quality model
(Sonnet), so the judge isn't grading its own output.

Usage:
    python eval/generation_eval.py                      # all golden Qs + refusal set, agentic retrieval
    python eval/generation_eval.py --n 8 --retrieval plain
    python eval/generation_eval.py --skip-refusal --gen-model claude-sonnet-4-6
    python eval/generation_eval.py --skip-quality       # only the refusal test

Needs QDRANT_URL and ANTHROPIC_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ask_parliament.config import GENERATION_MODEL, REPO_ROOT  # noqa: E402
from ask_parliament.generation import generate_answer  # noqa: E402
from ask_parliament.retrieval import Retriever  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from evallib import metrics as M  # noqa: E402
from evallib import llm_judge, persistence, provenance  # noqa: E402
from evallib.goldenset import load_golden, year_window  # noqa: E402
from evallib.llm_judge import JUDGE_MODEL  # noqa: E402
from evallib.searchers import build_searchers  # noqa: E402

load_dotenv(REPO_ROOT / ".env")
REFUSAL_PATH = REPO_ROOT / "eval" / "refusal_set.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


THROTTLE_S = 1.5  # gentle pacing between items to stay under API rate limits


def _safe(fn, *args, **kwargs):
    """Run an API-touching call; on a transient failure return None instead of
    aborting the whole eval (one rate-limited item shouldn't kill a 24-item run)."""
    for attempt in range(4):
        try:
            return fn(*args, **kwargs)
        except Exception as err:  # noqa: BLE001
            if attempt < 3:
                time.sleep(8 * (attempt + 1))  # 8s, 16s, 24s backoff
                continue
            print(f"    (call failed after retries: {err})", flush=True)
            return None


def run_quality(searcher, items, gen_model, judge_model, k) -> list[dict]:
    rows = []
    for it in items:
        yf, yt = year_window(it)
        hits = searcher.search(it["question"], k=k, countries=it.get("countries"),
                               year_from=yf, year_to=yt)
        res = _safe(generate_answer, it["question"], hits, model=gen_model)
        j = _safe(llm_judge.judge_answer, it["question"], res.answer, res.sources,
                  model=judge_model) if res else None
        rows.append({
            "id": it["id"], "question": it["question"],
            "n_sources": len(res.sources) if res else 0,
            "answer": res.answer if res else None, "judged": j is not None, **(j or {}),
        })
        ok = "ok" if j else "JUDGE/GEN-FAILED"
        g = j["groundedness"] if j else "-"
        print(f"  {it['id']:<26} grounded={g} ({ok})", flush=True)
        time.sleep(THROTTLE_S)
    return rows


def run_refusal(searcher, items, gen_model, judge_model, k) -> list[dict]:
    rows = []
    for it in items:
        hits = searcher.search(it["question"], k=k)
        res = _safe(generate_answer, it["question"], hits, model=gen_model)
        j = _safe(llm_judge.judge_refusal, it["question"], res.answer, res.sources,
                  model=judge_model) if res else None
        rows.append({"id": it["id"], "question": it["question"],
                     "answer": res.answer if res else None,
                     "judged": j is not None, **(j or {})})
        verdict = ("JUDGE/GEN-FAILED" if not j else
                   "HALLUCINATED" if j["hallucinated"] else
                   "refused" if j["refused"] else "grounded")
        print(f"  {it['id']:<26} {verdict}", flush=True)
        time.sleep(THROTTLE_S)
    return rows


def _mean_scores(rows, keys):
    out = {}
    for kname in keys:
        vals = [r[kname] for r in rows if r.get("judged") and r.get(kname) is not None]
        out[kname] = M.summarize(vals)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, help="limit the in-corpus quality questions (default: all golden)")
    p.add_argument("--retrieval", default="agentic", choices=["plain", "rerank", "transform", "agentic"],
                   help="retrieval method feeding generation (default agentic = production path)")
    p.add_argument("--k", type=int, default=8, help="speeches retrieved for generation (default 8)")
    p.add_argument("--gen-model", default=GENERATION_MODEL, help=f"answer model (default {GENERATION_MODEL})")
    p.add_argument("--judge-model", default=JUDGE_MODEL, help=f"judge model (default {JUDGE_MODEL})")
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--skip-refusal", action="store_true")
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args()

    base = Retriever()
    searcher = build_searchers([a.retrieval], base)[a.retrieval]
    print(f"Generation eval — retrieval={a.retrieval}, gen={a.gen_model}, judge={a.judge_model}, k={a.k}\n")

    doc = provenance.stamp({"eval": "generation", "retrieval": a.retrieval,
                            "gen_model": a.gen_model, "judge_model": a.judge_model, "k": a.k})
    summary = {}

    if not a.skip_quality:
        items = load_golden()
        if a.n:
            items = items[: a.n]
        print(f"Quality (in-corpus) — {len(items)} questions:")
        t0 = time.time()
        q_rows = run_quality(searcher, items, a.gen_model, a.judge_model, a.k)
        q_scores = _mean_scores(q_rows, ["groundedness", "citation_validity", "answer_relevance"])
        n_judged = sum(r["judged"] for r in q_rows)
        summary["quality"] = {"scores": q_scores, "n": len(q_rows), "n_judged": n_judged}
        doc["quality_rows"] = q_rows
        print(f"  ({time.time() - t0:.0f}s, {n_judged}/{len(q_rows)} judged)\n")

    if not a.skip_refusal:
        ref = _load_jsonl(REFUSAL_PATH)
        print(f"Refusal (out-of-corpus) — {len(ref)} questions:")
        t0 = time.time()
        r_rows = run_refusal(searcher, ref, a.gen_model, a.judge_model, a.k)
        judged = [r for r in r_rows if r["judged"]]
        nj = len(judged) or 1
        refusal_rate = sum(r["refused"] for r in judged) / nj
        grounded_rate = sum(r.get("grounded_answer") for r in judged) / nj
        halluc_rate = sum(r["hallucinated"] for r in judged) / nj
        ok_rate = sum(not r["hallucinated"] for r in judged) / nj  # refused OR grounded
        summary["refusal"] = {"refusal_rate": refusal_rate, "grounded_rate": grounded_rate,
                              "hallucination_rate": halluc_rate, "ok_rate": ok_rate,
                              "n": len(r_rows), "n_judged": len(judged)}
        doc["refusal_rows"] = r_rows
        print(f"  ({time.time() - t0:.0f}s)\n")

    # --- report ---
    print("=" * 60)
    if "quality" in summary:
        s = summary["quality"]["scores"]
        print("In-corpus answer quality (1-5, mean [95% CI]):")
        for kname in ("groundedness", "citation_validity", "answer_relevance"):
            v = s[kname]
            print(f"  {kname:<18} {v['mean']:.2f} [{v['ci_low']:.2f},{v['ci_high']:.2f}]  (n={v['n']})")
    if "refusal" in summary:
        r = summary["refusal"]
        print(f"\nOut-of-corpus handling (refused OR grounded = correct): "
              f"ok {r['ok_rate']*100:.0f}% (refused {r['refusal_rate']*100:.0f}% + "
              f"grounded {r['grounded_rate']*100:.0f}%) · hallucinated "
              f"{r['hallucination_rate']*100:.0f}%  ({r['n_judged']}/{r['n']} judged) "
              "— lower hallucination is better")

    if not a.no_save:
        doc["summary"] = summary
        headline = {**provenance.stamp(), "retrieval": a.retrieval, "gen_model": a.gen_model,
                    "summary": summary}
        path = persistence.save_run("generation", doc, headline)
        print(f"\nSaved -> {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
