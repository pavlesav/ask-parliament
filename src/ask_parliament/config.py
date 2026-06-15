"""Central configuration: paths, model names, corpus-selection constants.

Everything the pipeline treats as a decision lives here, so each script reads
the same values and the choices are documented in one place.
"""
import os
from pathlib import Path

# --- Paths ---------------------------------------------------------------
# Repo root = two levels above this file (src/ask_parliament/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

# The thesis repo is a sibling of this repo; its data folder really is named
# "data folder" (with a space) — see DATA_MAP.md §2.
THESIS_DATA_DIR = REPO_ROOT.parent / "master-thesis" / "code" / "data folder"
AT_FINAL_PKL = THESIS_DATA_DIR / "AT" / "AT_final.pkl"

# Persistent Chroma store (gitignored).
CHROMA_DIR = REPO_ROOT / "chroma_db"
COLLECTION_NAME = "speeches_at"

# --- Corpus selection (Phase 1 decision: Austria, full range 1996-2022) ---
# Country code stored in every record's metadata (the store is single-country
# today; the field keeps the schema honest for a future HR ingest).
COUNTRY = "AT"
# Regular speakers only: drops ~125k Chairperson procedural turns and 440 Guests.
SPEAKER_ROLE = "Regular"
# Drop trivially short interventions (interjection fragments, one-liners).
MIN_TEXT_CHARS = 300
# Expected row count after filtering, verified in Phase 0 (DATA_MAP.md §9).
# The build script warns if the actual count drifts from this.
EXPECTED_SPEECH_COUNT = 99_089

# --- Embeddings ------------------------------------------------------------
# Must match the thesis: BAAI/bge-m3, 1024-d, cosine space. Some stored vectors
# are chunk-averaged and not unit-norm, so cosine (not dot product) is required.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Generation -------------------------------------------------------------
# Cheap-by-default; switch to the quality model when polishing answers.
GENERATION_MODEL = "claude-haiku-4-5"
GENERATION_MODEL_QUALITY = "claude-sonnet-4-6"

# --- Scaling: full ParlaMint 5.0 corpus (recompute embeddings on GPU) --------
# ParlaMint 5.0 lives on CLARIN.SI (CC BY 4.0, direct download, no login): one
# .tgz per country/region. The scale pipeline (scripts/scale/) uses the native
# plain-text + English TSV metadata and recomputes BGE-m3 vectors from scratch,
# so the scaled index is a uniform space independent of the thesis vectors.
PARLAMINT_VERSION = "5.0"
PARLAMINT_HANDLE = "11356/2004"
PARLAMINT_PAGE = f"https://www.clarin.si/repository/xmlui/handle/{PARLAMINT_HANDLE}"

# The 29 corpora in ParlaMint 5.0 (national parliaments + Spanish regional ones).
PARLAMINT_COUNTRIES = [
    "AT", "BA", "BE", "BG", "CZ", "DK", "EE", "ES", "ES-CT", "ES-GA", "ES-PV",
    "FI", "FR", "GB", "GR", "HR", "HU", "IS", "IT", "LV", "NL", "NO", "PL",
    "PT", "RS", "SE", "SI", "TR", "UA",
]

# Data lives outside git (data/ is gitignored). Set PARLAMINT_DATA_DIR to point
# the whole pipeline elsewhere (e.g. a sample dir while testing, or a big disk).
DATA_DIR = Path(os.environ.get("PARLAMINT_DATA_DIR", str(REPO_ROOT / "data" / "parlamint")))
RAW_DIR = DATA_DIR / "raw"          # extracted ParlaMint-{CC}.txt/ trees
PARSED_DIR = DATA_DIR / "parsed"    # {CC}.parquet — one tidy row per speech
EMB_DIR = DATA_DIR / "embeddings"   # {CC}/shard_*.npy + manifest.json

# Speech selection at scale (mirrors the Austria demo: Regular speakers only,
# drop trivially short turns). Role is the canonical English value in -meta-en.tsv.
SCALE_SPEAKER_ROLE = "Regular"
SCALE_MIN_TEXT_CHARS = 300

# --- Embedding compute (GPU batch job) --------------------------------------
EMBED_BATCH_SIZE = 64          # per-forward batch; raise on a big GPU, lower on OOM
EMBED_SHARD_SIZE = 50_000      # speeches per .npy shard = the resume/parallel unit
EMBED_MAX_SEQ_LENGTH = 1024    # truncate long speeches for throughput (raise for fidelity)
EMBED_DTYPE = "float16"        # stored vector dtype (half the disk of float32)
