"""Run the whole evaluation suite end to end, then render the report.

A thin orchestrator over the individual scripts (each still runnable on its own):

    build_golden_set.py  ->  retrieval_eval.py  ->  generation_eval.py  ->  report.py
    (+ filter_check.py as a gate)

so a single command reproduces every number in eval/results/REPORT.md. Steps shell out to
the same entrypoints a human would call, so there's no hidden second code path.

Usage:
    python eval/run_all.py                 # full suite (needs Qdrant + ANTHROPIC_API_KEY)
    python eval/run_all.py --quick         # tiny subset, for a fast smoke of the wiring
    python eval/run_all.py --skip-build    # reuse the existing golden_set.jsonl
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PY = sys.executable


def run(name: str, args: list[str]) -> int:
    """Run one eval script, streaming its output; return its exit code."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}  # native-language text in stdout
    print(f"\n{'='*70}\n# {name}\n{'='*70}", flush=True)
    return subprocess.run([PY, "-u", str(EVAL_DIR / name), *args], env=env).returncode


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="tiny subset, smoke the wiring (fast)")
    p.add_argument("--skip-build", action="store_true", help="reuse the existing golden_set.jsonl")
    p.add_argument("--skip-filter-check", action="store_true")
    p.add_argument("--k", type=int, default=10)
    a = p.parse_args()

    if "QDRANT_URL" not in os.environ:
        print("warning: QDRANT_URL is not set — retrieval/generation will use the embedded store.")

    steps: list[tuple[str, list[str]]] = []
    if not a.skip_build:
        steps.append(("build_golden_set.py", []))
    if not a.skip_filter_check:
        steps.append(("filter_check.py", []))
    ret_args = ["--k", str(a.k)] + (["--limit", "4", "--methods", "plain,agentic"] if a.quick else [])
    steps.append(("retrieval_eval.py", ret_args))
    gen_args = (["--n", "4", "--retrieval", "plain"] if a.quick else ["--retrieval", "agentic"])
    steps.append(("generation_eval.py", gen_args))
    steps.append(("report.py", []))

    for name, args in steps:
        code = run(name, args)
        if code != 0 and name != "filter_check.py":  # filter_check is a gate, but don't abort the suite on it
            print(f"\n{name} exited {code} — stopping.")
            return code
    print("\nSuite complete. See eval/results/REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
