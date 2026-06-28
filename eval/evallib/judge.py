"""Pattern-based relevance judging for the golden set.

The known-item eval credits exactly one speech (its own source), so it can't measure
real recall — every *other* speech that genuinely answers the question is scored as a
miss. The golden set fixes that with cheap, transparent, language-agnostic judging:
a retrieved speech is **relevant** if its text matches the topic's patterns (and any
country/date constraints the item declares). No LLM call, fully deterministic, and it
credits *any* on-topic speech from *any* of the 29 parliaments.

A golden item (one JSONL line) looks like:

    {
      "id": "ukraine_invasion_2022",
      "question": "How did parliaments respond to Russia's 2022 invasion of Ukraine?",
      "topic_domain": "International Affairs",
      "patterns": ["ukrain", "invasion|aggression|war|troops"],   # ALL must hit (each is a regex, OR inside)
      "strong_patterns": ["invasion of ukraine|russian aggression"],  # optional -> grade 2 (highly on-point)
      "countries": null,            # optional whitelist; null = any parliament counts
      "date_from": "2022-02-01",    # optional ISO window (helps event-anchored items)
      "date_to": "2022-12-31"
    }

Grade is 2 when a strong pattern also matches (and date/country pass), 1 for a plain
match, 0 otherwise — so nDCG has graded signal while success@k/precision treat >0 as hit.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def fold(s: str) -> str:
    """Strip Latin combining diacritics so accented spellings match unaccented stems
    (Grécia→grecia, réforme→reforme, Ucrânia→ucrania, nucléaire→nucleaire). Greek and
    Cyrillic are scripts, not accents, so they survive NFKD unchanged. This removed a
    whole class of false negatives the calibration found (see eval/calibration/)."""
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


@dataclass
class CompiledItem:
    """A golden item with its regexes compiled once, ready to judge many speeches."""

    id: str
    question: str
    patterns: list[re.Pattern]
    strong_patterns: list[re.Pattern]
    countries: set[str] | None
    date_from: str | None
    date_to: str | None
    topic_domain: str
    raw: dict


def compile_item(item: dict) -> CompiledItem:
    flags = re.IGNORECASE
    # Patterns are matched against diacritic-folded text, so fold the patterns too.
    pats = [re.compile(fold(p), flags) for p in item.get("patterns", [])]
    strong = [re.compile(fold(p), flags) for p in item.get("strong_patterns", [])]
    countries = item.get("countries")
    return CompiledItem(
        id=item["id"],
        question=item["question"],
        patterns=pats,
        strong_patterns=strong,
        countries=set(countries) if countries else None,
        date_from=item.get("date_from"),
        date_to=item.get("date_to"),
        topic_domain=item.get("topic_domain", "-"),
        raw=item,
    )


def _all_match(text: str, patterns: list[re.Pattern]) -> bool:
    folded = fold(text)
    return all(p.search(folded) for p in patterns)


def _date_ok(date: str | None, lo: str | None, hi: str | None) -> bool:
    if not date:
        return lo is None and hi is None
    if lo and date < lo:
        return False
    if hi and date > hi:
        return False
    return True


def relevance(text: str, country: str | None, date: str | None, item: CompiledItem) -> float:
    """Graded relevance (0 / 1 / 2) of one speech to a compiled golden item.

    Text must match every pattern; country (if constrained) and date window (if set)
    must also pass. A matching strong pattern lifts a relevant hit to grade 2.
    """
    if item.countries is not None and (country or "") not in item.countries:
        return 0.0
    if not _date_ok(date, item.date_from, item.date_to):
        return 0.0
    if not item.patterns or not _all_match(text, item.patterns):
        return 0.0
    if item.strong_patterns and _all_match(text, item.strong_patterns):
        return 2.0
    return 1.0


def grade_hits(hits, item: CompiledItem) -> list[float]:
    """Graded relevance for a ranked list of RetrievedSpeech (or dicts), in order."""
    grades = []
    for h in hits:
        text = getattr(h, "text", None) if not isinstance(h, dict) else h.get("text")
        country = getattr(h, "country", None) if not isinstance(h, dict) else h.get("country")
        date = getattr(h, "date", None) if not isinstance(h, dict) else h.get("date")
        grades.append(relevance(text or "", country, date, item))
    return grades
