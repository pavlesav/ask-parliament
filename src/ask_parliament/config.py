"""Central configuration: paths, model names, and pipeline constants.

Every decision the pipeline treats as a knob lives here, so each script reads the
same values and the choices are documented in one place.
"""
import os
from pathlib import Path


def resolve_device() -> str:
    """'cuda' when a GPU is visible, else 'cpu'. torch is imported lazily so the
    config module stays cheap to import for code that doesn't load a model."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# --- Paths ---------------------------------------------------------------
# Repo root = two levels above this file (src/ask_parliament/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Qdrant store for the multi-country corpus (gitignored). Embedded on-disk by
# default; set QDRANT_URL to point the same client at a running Qdrant server
# instead (the production path) — no other code changes.
QDRANT_PATH = REPO_ROOT / "qdrant_db"
QDRANT_URL = os.environ.get("QDRANT_URL")  # e.g. "http://localhost:6333"; None -> embedded
QDRANT_COLLECTION = "speeches"

# --- Embeddings ------------------------------------------------------------
# BAAI/bge-m3: 1024-d, multilingual/cross-lingual. Vectors are L2-normalised at
# embed time, and search uses cosine.
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Generation -------------------------------------------------------------
# Cheap-by-default; switch to the quality model when polishing answers.
GENERATION_MODEL = "claude-haiku-4-5"
GENERATION_MODEL_QUALITY = "claude-sonnet-4-6"

# --- Retrieval quality: reranking + agentic query transformation ------------
# Cross-encoder reranker — same family as bge-m3, multilingual, so it scores the
# native-language speeches directly. Loaded lazily; runs on GPU if torch sees one.
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_MAX_LENGTH = 512        # query+speech truncation for the cross-encoder (lead is enough)
# Agentic pipeline knobs.
QUERY_TRANSFORM_MODEL = GENERATION_MODEL  # cheap model for query rewriting / HyDE
PER_QUERY_LIMIT = 40           # vector hits fetched per query variant
# Pool fed into the cross-encoder (and the RRF pool size in the agentic path). Tuned
# on the golden set (eval/tune_rerank_pool.py): going 60 -> 150 improved rerank MRR,
# precision@10 and nDCG@10 together; beyond ~150 precision keeps creeping but MRR falls
# off as the cross-encoder wades through more near-duplicates. 150 is the balance point.
RERANK_CANDIDATES = 150
RERANK_SCORE_FLOOR = 0.30      # if the top reranked score is below this, the loop self-corrects
AGENTIC_MAX_ROUNDS = 1         # corrective re-retrieval rounds after the first pass

# --- Corpus: full ParlaMint 5.0 (29 parliaments) ----------------------------
# ParlaMint 5.0 lives on CLARIN.SI (CC BY 4.0, direct download, no login): one
# .tgz per country/region. The pipeline (scripts/scale/) uses the native
# plain-text + English TSV metadata and computes BGE-m3 vectors, so the index is
# one uniform cross-lingual space.
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
# Facets sidecar written by the index build, read by the app (so the filter
# widgets don't require sweeping millions of payloads).
FACETS_PATH = DATA_DIR / "facets.json"

# Speech selection: Regular speakers only (drop Chairperson/Guest procedural
# turns), and drop trivially short turns. Role is the canonical English value in
# the -meta-en.tsv files.
SCALE_SPEAKER_ROLE = "Regular"
SCALE_MIN_TEXT_CHARS = 300

# --- Embedding compute (GPU batch job) --------------------------------------
EMBED_BATCH_SIZE = 64          # per-forward batch; raise on a big GPU, lower on OOM
EMBED_SHARD_SIZE = 50_000      # speeches per .npy shard = the resume/parallel unit
EMBED_MAX_SEQ_LENGTH = 1024    # truncate long speeches for throughput (raise for fidelity)
EMBED_DTYPE = "float16"        # stored vector dtype (half the disk of float32)
