"""Persistence must round-trip non-ASCII (native-language speech text) on any platform.

Regression guard for a real bug: `save_run` once used `Path.write_text` without an
encoding, so on Windows (cp1252) it crashed the moment a result held a Cyrillic/Greek/
Latvian character — which every real run does."""
import json

from evallib import persistence


def test_save_run_handles_non_ascii(tmp_path, monkeypatch):
    monkeypatch.setattr(persistence, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(persistence, "HISTORY_PATH", tmp_path / "history.jsonl")

    doc = {
        "timestamp": "2026-01-01T00:00:00Z",
        "answer": "Sākotnēji — Ελλάδα — Україна — Łódź — Köln",  # lv/gr/ua/pl/de
    }
    path = persistence.save_run("unit", doc, {"timestamp": doc["timestamp"], "ok": True})

    assert path.exists()
    back = json.loads(path.read_text(encoding="utf-8"))
    assert back["answer"] == doc["answer"]

    hist = persistence.load_history()
    assert hist and hist[-1]["kind"] == "unit" and hist[-1]["ok"] is True
