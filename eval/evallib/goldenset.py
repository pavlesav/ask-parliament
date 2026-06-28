"""Load and validate the golden set (eval/golden_set.jsonl)."""
from __future__ import annotations

import json
from pathlib import Path

from ask_parliament.config import REPO_ROOT

GOLDEN_PATH = REPO_ROOT / "eval" / "golden_set.jsonl"
REQUIRED = ("id", "question", "patterns")


def load_golden(path: Path | None = None) -> list[dict]:
    """Return the golden items (raw dicts). Raises if the file is missing or malformed."""
    path = path or GOLDEN_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — build it first: python eval/build_golden_set.py"
        )
    items = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        missing = [k for k in REQUIRED if k not in item]
        if missing:
            raise ValueError(f"golden item on line {n} missing {missing}: {item.get('id', '?')}")
        items.append(item)
    if not items:
        raise ValueError(f"{path} is empty")
    return items


def year_window(item: dict) -> tuple[int | None, int | None]:
    """Year bounds (for the Qdrant year filter) from the item's ISO date window."""
    yf = int(item["date_from"][:4]) if item.get("date_from") else None
    yt = int(item["date_to"][:4]) if item.get("date_to") else None
    return yf, yt
