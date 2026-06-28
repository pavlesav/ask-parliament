"""FastAPI backend for Ask Parliament.

Owns the heavy machinery — the Qdrant client, the BGE-m3 embedder, the cross-encoder
reranker, and the Anthropic generation call — behind a small HTTP API so the Streamlit
frontend can stay a thin, model-free client. The retrievers are built once at startup
and warmed so the first user request isn't penalised by model loading.

Endpoints:
  GET  /health  — liveness/readiness
  GET  /meta    — filter facets + model choices (everything the sidebar needs)
  POST /ask     — retrieve (plain or agentic) + grounded generation

Run:  uvicorn ask_parliament.api:app --host 0.0.0.0 --port 8000
Env:  QDRANT_URL (Qdrant server), ANTHROPIC_API_KEY (.env or environment).
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel

from ask_parliament.agentic import AgenticRetriever
from ask_parliament.config import GENERATION_MODEL, GENERATION_MODEL_QUALITY
from ask_parliament.generation import generate_answer
from ask_parliament.retrieval import Retriever

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("api")


class AskRequest(BaseModel):
    question: str
    k: int = 8
    countries: list[str] | None = None
    year_from: int | None = None
    year_to: int | None = None
    cap_domains: list[str] | None = None
    party: str | None = None
    model: str = GENERATION_MODEL
    agentic: bool = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build retrievers once and warm the models so the first request is fast."""
    log.info("Loading retriever + reranker …")
    app.state.retriever = Retriever()
    app.state.agentic = AgenticRetriever(app.state.retriever)
    try:
        app.state.agentic.search("warmup", k=1)  # loads embedder + reranker
        log.info("Warmup complete — models loaded.")
    except Exception as err:  # noqa: BLE001 — warmup is best-effort; serve anyway
        log.warning("Warmup skipped (%s); models will load on first request.", err)
    yield


app = FastAPI(title="Ask Parliament API", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/meta")
def meta() -> dict:
    """Filter facets + model choices — one call populates the whole sidebar."""
    f = app.state.retriever.facets()
    return {
        "countries": f.countries,
        "cap_domains": f.cap_domains,
        "parties": f.parties,
        "year_min": f.year_min,
        "year_max": f.year_max,
        "models": [GENERATION_MODEL, GENERATION_MODEL_QUALITY],
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    """Retrieve (plain or agentic) and generate a grounded, cited answer."""
    searcher = app.state.agentic if req.agentic else app.state.retriever
    hits = searcher.search(
        req.question, k=req.k, countries=req.countries, year_from=req.year_from,
        year_to=req.year_to, cap_domains=req.cap_domains, party=req.party,
    )
    result = generate_answer(req.question, hits, model=req.model)
    return {
        "answer": result.answer,
        "sources": [asdict(s) for s in result.sources],
        "model": result.model,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "latency_s": result.latency_s,
        "retrieval_s": sum(searcher.last_timings.values()),
        "trace": getattr(searcher, "last_trace", None) if req.agentic else None,
    }
