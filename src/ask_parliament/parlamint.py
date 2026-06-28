"""Parse a ParlaMint 5.0 country corpus (plain-text + TSV) into speech records.

ParlaMint 5.0 (CLARIN.SI, CC BY 4.0) ships, per country, a `ParlaMint-{CC}.txt/`
tree of per-session files:

  - `<session>.txt`         — one line per utterance: ``<utterance_id>\\t<native text>``
  - `<session>-meta-en.tsv` — one row per utterance, 24 columns, with canonical
                              English ``Speaker_role`` (Regular/Chairperson/Guest)
                              and the per-speech CAP ``Topic``.
  - `<session>-meta.tsv`    — same columns, but role/labels localized; used as a
                              fallback when no English meta exists (e.g. GB).

We embed the NATIVE text (BGE-m3 is multilingual and cross-lingual) and take the
structured metadata from the English meta, so roles and CAP topics are uniform
across all 29 corpora. Text and metadata align on the utterance ``ID``.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import pandas as pd

from ask_parliament.config import RAW_DIR, SCALE_MIN_TEXT_CHARS, SCALE_SPEAKER_ROLE

log = logging.getLogger("parlamint")

# Source -meta column -> output field. Order defines the parsed-frame columns.
_META_FIELDS = {
    "Text_ID": "text_id",       # session id — lets us regroup a debate later
    "Date": "date",
    "Title": "title",
    "Term": "term",
    "Body": "body",
    "Subcorpus": "subcorpus",   # Reference / COVID / war — free era filter
    "Speaker_role": "role",     # filtered on; kept for provenance
    "Speaker_party": "party",
    "Speaker_party_name": "party_name",
    "Party_status": "party_status",
    "Party_orientation": "party_orientation",
    "Speaker_ID": "speaker_id",
    "Speaker_name": "speaker",
    "Speaker_gender": "gender",
    "Topic": "cap_topic",       # ParlaMint's automatic CAP top-level topic
}

# Fields where an empty cell should read as "-" (index-friendly, matches the app).
_TIDY_EMPTY = [
    "text_id", "title", "term", "body", "subcorpus", "party", "party_name",
    "party_status", "party_orientation", "speaker_id", "speaker", "gender", "cap_topic",
]


def _read_text(txt_path: Path) -> dict[str, str]:
    """Map utterance ID -> text from a session `.txt` (``ID\\ttext`` per line)."""
    out: dict[str, str] = {}
    with txt_path.open(encoding="utf-8") as fh:
        for line in fh:
            uid, sep, text = line.rstrip("\n").partition("\t")
            if sep and text:
                out[uid] = text
    return out


def _meta_path_for(txt_path: Path) -> Path | None:
    """Prefer English meta (uniform role/topic); fall back to native, else None."""
    en = txt_path.with_name(txt_path.stem + "-meta-en.tsv")
    if en.exists():
        return en
    native = txt_path.with_name(txt_path.stem + "-meta.tsv")
    return native if native.exists() else None


def _iter_session_files(country_dir: Path):
    """Yield (txt_path, meta_path|None) for every session under ParlaMint-{CC}.txt/."""
    txt_root = next(country_dir.glob("ParlaMint-*.txt"), None)
    if txt_root is None or not txt_root.is_dir():
        raise FileNotFoundError(f"No 'ParlaMint-*.txt' tree under {country_dir}")
    for txt_path in sorted(txt_root.rglob("*.txt")):
        if txt_path.name == "00README.txt":
            continue
        yield txt_path, _meta_path_for(txt_path)


def parse_country(country: str, raw_dir: Path | None = None) -> pd.DataFrame:
    """Parse one country into a tidy, filtered speech-level DataFrame.

    Columns: id, text, country, then the mapped metadata fields, plus year.
    Filtered to Regular speakers with text >= SCALE_MIN_TEXT_CHARS.
    """
    raw_dir = raw_dir or RAW_DIR
    country_dir = raw_dir / country
    rows: list[dict] = []
    n_sessions = n_missing_meta = 0

    for txt_path, meta_path in _iter_session_files(country_dir):
        n_sessions += 1
        if meta_path is None:
            n_missing_meta += 1
            continue
        texts = _read_text(txt_path)
        meta = pd.read_csv(
            meta_path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
            keep_default_na=False, na_filter=False, on_bad_lines="warn",
        )
        for d in meta.to_dict("records"):
            text = texts.get(d.get("ID", ""))
            if not text:
                continue
            rec = {"id": d["ID"], "text": text, "country": country}
            for col, field in _META_FIELDS.items():
                rec[field] = d.get(col, "")
            rows.append(rec)

    if not rows:
        log.warning("%s: parsed 0 rows from %d sessions", country, n_sessions)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["year"] = pd.to_numeric(df["date"].str.slice(0, 4), errors="coerce").fillna(0).astype(int)

    n0 = len(df)
    df = df[df["role"] == SCALE_SPEAKER_ROLE]
    n1 = len(df)
    df = df[df["text"].str.len() >= SCALE_MIN_TEXT_CHARS]
    n2 = len(df)

    for col in _TIDY_EMPTY:
        df[col] = df[col].replace("", "-")

    log.info(
        "%s: %d sessions (%d w/o meta), %s utterances -> %s Regular -> %s >=%d chars",
        country, n_sessions, n_missing_meta, f"{n0:,}", f"{n1:,}", f"{n2:,}", SCALE_MIN_TEXT_CHARS,
    )
    return df.reset_index(drop=True)
