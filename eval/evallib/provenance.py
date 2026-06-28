"""Run provenance: stamp every eval result with enough context to reproduce it.

A metric without its conditions is noise. Each run records the git commit (and
whether the tree was dirty), the timestamp, the model/knob config the pipeline was
using, and the host device — so a number in `eval/results/` can always be traced
back to exactly what produced it, and regressions can be attributed to a change.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone

from ask_parliament import config


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=config.REPO_ROOT, timeout=10
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 — provenance is best-effort, never block a run
        return ""


def git_state() -> dict:
    sha = _git("rev-parse", "--short", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    return {"sha": sha, "branch": branch, "dirty": dirty}


def config_snapshot() -> dict:
    """The knobs that actually move eval numbers, captured by value."""
    return {
        "embedding_model": config.EMBEDDING_MODEL,
        "rerank_model": config.RERANK_MODEL,
        "generation_model": config.GENERATION_MODEL,
        "query_transform_model": config.QUERY_TRANSFORM_MODEL,
        "per_query_limit": config.PER_QUERY_LIMIT,
        "rerank_candidates": config.RERANK_CANDIDATES,
        "rerank_score_floor": config.RERANK_SCORE_FLOOR,
        "agentic_max_rounds": config.AGENTIC_MAX_ROUNDS,
        "qdrant_url": config.QDRANT_URL or "embedded",
        "device": config.resolve_device(),
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp(extra: dict | None = None) -> dict:
    """A provenance block to embed at the top of every result file."""
    block = {
        "timestamp": now_iso(),
        "git": git_state(),
        "config": config_snapshot(),
        "python": sys.version.split()[0],
    }
    if extra:
        block.update(extra)
    return block
