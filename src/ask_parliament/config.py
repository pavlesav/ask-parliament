"""Central configuration: paths, model names, corpus-selection constants.

Everything the pipeline treats as a decision lives here, so each script reads
the same values and the choices are documented in one place.
"""
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
