"""LLM-as-judge for grounded generation quality.

Retrieval metrics say nothing about the *answer*. This module scores the generated
answer against the speeches it was built from, using a stronger model as judge
(Sonnet by default) than the one that writes the answers (Haiku) — so the judge isn't
just grading its own homework. Three axes, the standard RAG triad plus citations:

  * groundedness / faithfulness — is every claim supported by the provided speeches,
    with no outside knowledge or invention? (the core RAG property)
  * citation validity — does each [n] marker point to a source that actually backs
    the sentence it's attached to?
  * answer relevance — does the answer address the question that was asked?

A separate `judge_refusal` checks the negative set: when the retrieved speeches don't
contain the answer, a good system *declines* instead of confabulating.

All judging asks for strict JSON and parses defensively; on any API/parse failure the
caller gets a `None`/empty result and can record the item as unjudged rather than crash.
"""
from __future__ import annotations

import json
import re

import anthropic
from dotenv import load_dotenv

from ask_parliament.config import GENERATION_MODEL_QUALITY, REPO_ROOT
from ask_parliament.generation import format_context

load_dotenv(REPO_ROOT / ".env")

JUDGE_MODEL = GENERATION_MODEL_QUALITY  # Sonnet: stronger than the Haiku generator

_JUDGE_SYSTEM = """You are a strict, fair evaluator of a retrieval-augmented answer about \
European parliamentary debates. You are given the QUESTION, the numbered SPEECHES that were \
retrieved (each may be in its original language), and the ANSWER a system produced from them.

Judge ONLY against the provided speeches — treat them as the sole ground truth, even if you \
believe otherwise. Assess three things:

1. groundedness (1-5): is every factual claim in the answer supported by the speeches? \
5 = fully supported; 3 = mostly, with a minor unsupported detail; 1 = substantial claims have \
no support in the speeches (hallucination). List any unsupported claims.
2. citation_validity (1-5): the answer cites speeches with [n] markers. 5 = every [n] points to \
a speech that genuinely supports the attached statement; 1 = citations are largely wrong or \
missing where needed. List any clearly wrong [n] markers.
3. answer_relevance (1-5): does the answer actually address the question? 5 = directly and \
completely; 1 = off-topic or non-responsive. A correct, well-grounded refusal ("the speeches \
don't address this") when the speeches truly lack the answer should score 5 on relevance and \
groundedness.

Return ONLY a JSON object, no prose:
{"groundedness": int, "citation_validity": int, "answer_relevance": int,
 "unsupported_claims": [str], "bad_citations": [str], "comment": str}"""

_REFUSAL_SYSTEM = """You evaluate a retrieval-augmented answer to a question that is OUT OF SCOPE \
for a corpus of European parliamentary speeches. You are given the QUESTION, the numbered SPEECHES \
that were retrieved, and the ANSWER.

A correct system does ONE of two things: (a) it DECLINES, saying the speeches don't address the \
question; or (b) it answers using ONLY facts actually stated in the provided speeches, cited with \
[n]. Both are acceptable — grounding an answer in a retrieved speech is fine even when the topic \
seems like general knowledge. The ONLY failure is asserting a substantive factual answer that is \
NOT supported by any provided speech (i.e. drawing on outside/general knowledge) — a hallucination.

Judge ONLY against the provided speeches. Decide:
- refused (bool): did the answer decline / state the speeches don't contain the answer?
- grounded_answer (bool): did it give a substantive answer that IS supported by the cited speeches?
- hallucinated (bool): did it assert a substantive answer NOT supported by any provided speech?

Return ONLY JSON: {"refused": bool, "grounded_answer": bool, "hallucinated": bool, "comment": str}"""


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(max_retries=2)


def _complete(system: str, user: str, model: str) -> str:
    resp = _client().messages.create(
        model=model, max_tokens=700, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return next((b.text for b in resp.content if b.type == "text"), "")


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _clamp(v, lo=1, hi=5):
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except (TypeError, ValueError):
        return None


def judge_answer(question: str, answer: str, sources, model: str = JUDGE_MODEL) -> dict | None:
    """Score one grounded answer. Returns None on API/parse failure (record as unjudged)."""
    user = (
        f"QUESTION:\n{question}\n\nSPEECHES:\n{format_context(sources)}\n\n"
        f"ANSWER:\n{answer}\n\nReturn the JSON judgment."
    )
    try:
        data = _parse_json(_complete(_JUDGE_SYSTEM, user, model))
    except Exception:  # noqa: BLE001 — judging is best-effort
        return None
    if not data:
        return None
    return {
        "groundedness": _clamp(data.get("groundedness")),
        "citation_validity": _clamp(data.get("citation_validity")),
        "answer_relevance": _clamp(data.get("answer_relevance")),
        "unsupported_claims": data.get("unsupported_claims") or [],
        "bad_citations": data.get("bad_citations") or [],
        "comment": str(data.get("comment", ""))[:500],
    }


def judge_refusal(question: str, answer: str, sources, model: str = JUDGE_MODEL) -> dict | None:
    """For the negative set: did the system decline OR ground its answer in the sources?

    The only failure is a substantive answer unsupported by any provided speech. Seeing the
    sources lets the judge credit a correctly-cited answer (the corpus is vast, so an
    out-of-scope-looking question sometimes is genuinely answerable from a speech) instead
    of mislabelling it a hallucination. None on API/parse failure.
    """
    user = (f"QUESTION:\n{question}\n\nSPEECHES:\n{format_context(sources)}\n\n"
            f"ANSWER:\n{answer}\n\nReturn the JSON judgment.")
    try:
        data = _parse_json(_complete(_REFUSAL_SYSTEM, user, model))
    except Exception:  # noqa: BLE001
        return None
    if not data:
        return None
    return {
        "refused": bool(data.get("refused")),
        "grounded_answer": bool(data.get("grounded_answer")),
        "hallucinated": bool(data.get("hallucinated")),
        "comment": str(data.get("comment", ""))[:500],
    }
