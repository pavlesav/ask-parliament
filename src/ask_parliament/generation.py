"""Grounded answer generation over retrieved speeches.

Assembles a prompt from numbered retrieved speeches and the user's question,
asks Claude to answer using ONLY that context with [n] citations, and returns a
structured result (answer + the sources it was built from + usage/latency).

The model is told to say so explicitly when the context doesn't contain the
answer — the whole point of RAG here is that answers stay grounded in the corpus.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import anthropic
from dotenv import load_dotenv

from ask_parliament.config import GENERATION_MODEL, REPO_ROOT
from ask_parliament.retrieval import RetrievedSpeech

# Load ANTHROPIC_API_KEY from the repo-root .env (never hardcoded, gitignored).
load_dotenv(REPO_ROOT / ".env")

MAX_TOKENS = 1024  # answers are meant to be concise; well under any timeout

SYSTEM_PROMPT = """You answer questions about parliamentary debates using ONLY \
the numbered speeches provided in the user's message. These are real speeches \
from the Austrian Parliament (Nationalrat); non-English speeches have been \
machine-translated to English, so quote their substance rather than exact wording.

Rules:
- Base every statement only on the provided speeches. Do not use outside knowledge \
or assumptions about what was said.
- Cite the speeches you draw on with bracketed numbers like [1] or [2], matching \
the numbers in the context. Attach a citation to each claim.
- Speeches from one debate are grouped under a "Debate:" header in the order they \
were spoken; turns marked "· context" are the surrounding exchange, given for \
background. Cite whichever speech a statement actually comes from.
- If the provided speeches do not contain enough information to answer, say so \
plainly (e.g. "The retrieved speeches don't address this") instead of guessing.
- Be concise and neutral. Attribute views to the speakers, not to yourself.
- When speakers disagree, represent the different positions and cite each."""


@dataclass
class GenerationResult:
    """A grounded answer plus the sources and cost it was produced from."""

    answer: str
    sources: list[RetrievedSpeech]
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    stop_reason: str | None = None
    timings: dict[str, float] = field(default_factory=dict)


def format_context(hits: list[RetrievedSpeech]) -> str:
    """Render speeches as numbered blocks with citation headers.

    Consecutive speeches from the same segment (set up by expand_to_segments) are
    introduced with a "Debate:" header so the model can see the exchange as one
    conversation; context turns are flagged. Numbering follows list order, so [n]
    matches the n-th speech everywhere it is shown (prompt and Sources panel).
    """
    blocks, num, i, n = [], 0, 0, len(hits)
    while i < n:
        seg = hits[i].segment_id
        j = i
        while j < n and hits[j].segment_id == seg:
            j += 1
        run = hits[i:j]
        if seg not in ("-", "") and len(run) > 1:
            blocks.append(f"--- Debate: {run[0].topic_name} · {run[0].date} ---")
        for h in run:
            num += 1
            status = f", {h.party_status}" if h.party_status not in ("-", "") else ""
            tag = "" if h.is_anchor else " · context"
            header = f"[{num}] {h.speaker} ({h.party}{status}) — {h.date} — {h.cap_domain}{tag}"
            blocks.append(f"{header}\n{h.text.strip()}")
        i = j
    return "\n\n".join(blocks)


def build_user_message(question: str, hits: list[RetrievedSpeech]) -> str:
    return (
        "Here are the retrieved parliamentary speeches:\n\n"
        f"{format_context(hits)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the speeches above, with [n] citations."
    )


def _is_transient(err: Exception) -> bool:
    """Transient errors are worth one retry; client errors (auth, 400) are not."""
    if isinstance(err, (anthropic.RateLimitError, anthropic.APIConnectionError,
                        anthropic.InternalServerError)):
        return True
    # OverloadedError (529) and any other 5xx surface as APIStatusError.
    if isinstance(err, anthropic.APIStatusError):
        return err.status_code >= 500 or err.status_code == 529
    return False


def generate_answer(
    question: str,
    hits: list[RetrievedSpeech],
    model: str = GENERATION_MODEL,
) -> GenerationResult:
    """Generate a grounded answer. Retries once on transient API errors."""
    if not hits:
        return GenerationResult(
            answer="No speeches matched the query and filters, so there is nothing "
            "to answer from. Try broadening the filters or rephrasing.",
            sources=[], model=model, input_tokens=0, output_tokens=0, latency_s=0.0,
        )

    # max_retries=0: our explicit single retry below is the only retry, so the
    # behaviour matches what this code shows rather than hidden SDK retries.
    client = anthropic.Anthropic(max_retries=0)
    user_message = build_user_message(question, hits)

    last_err: Exception | None = None
    for attempt in range(2):  # one initial try + one retry
        t0 = time.time()
        try:
            response = client.messages.create(
                model=model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as err:  # noqa: BLE001 — classify, then re-raise with context
            last_err = err
            if attempt == 0 and _is_transient(err):
                time.sleep(1.0)
                continue
            raise RuntimeError(_error_message(err)) from err

        latency = time.time() - t0
        answer = next((b.text for b in response.content if b.type == "text"), "")
        return GenerationResult(
            answer=answer,
            sources=hits,
            model=model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_s=latency,
            stop_reason=response.stop_reason,
        )

    raise RuntimeError(_error_message(last_err))  # pragma: no cover — loop always returns/raises


def _error_message(err: Exception | None) -> str:
    """Map an SDK exception to a clear, user-facing message."""
    if isinstance(err, anthropic.AuthenticationError):
        return "Authentication failed — check ANTHROPIC_API_KEY in your .env."
    if isinstance(err, anthropic.PermissionDeniedError):
        return "The API key lacks permission for this model."
    if isinstance(err, anthropic.NotFoundError):
        return "Model not found — check the model name in config.py."
    if isinstance(err, anthropic.RateLimitError):
        return "Rate limited by the API after a retry. Wait a moment and try again."
    if isinstance(err, anthropic.APIStatusError):
        return f"The API returned an error ({err.status_code}) after a retry. Try again shortly."
    if isinstance(err, anthropic.APIConnectionError):
        return "Could not reach the Anthropic API after a retry. Check your connection."
    return f"Generation failed: {err}"
