"""Persist eval runs to disk so results are comparable over time.

Each run writes a full JSON document to `eval/results/<timestamp>_<kind>.json`
(provenance + config + every per-query record + the aggregate summary) and appends
one flat line to `eval/results/history.jsonl` (provenance + headline metrics only),
which is what the regression view and the report generator read. Results are
gitignored — they're machine-specific artefacts, not source.
"""
from __future__ import annotations

import json
from pathlib import Path

from ask_parliament.config import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "eval" / "results"
HISTORY_PATH = RESULTS_DIR / "history.jsonl"


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in s)[:60]


def save_run(kind: str, document: dict, headline: dict) -> Path:
    """Write the full result JSON and append a headline line to history.jsonl.

    `kind` is a short tag (e.g. "retrieval", "generation"). `document` is the full
    record (provenance, config, per-query rows, summary). `headline` is the compact
    subset for the history log (the same provenance stamp + top-line metrics).
    Returns the path of the full result file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = document.get("timestamp", "").replace(":", "").replace("-", "")
    path = RESULTS_DIR / f"{ts or 'run'}_{_slug(kind)}.json"
    # encoding is explicit: results hold native-language speech text, and Windows'
    # default cp1252 would crash on the first non-latin-1 character.
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")

    with HISTORY_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": kind, **headline}, ensure_ascii=False) + "\n")
    return path


def load_history() -> list[dict]:
    """Every headline record ever written, oldest first. [] if none yet."""
    if not HISTORY_PATH.exists():
        return []
    rows = []
    for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
