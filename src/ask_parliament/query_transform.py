"""LLM query transformation for retrieval.

A user question and a parliamentary speech rarely share surface form, so embedding
the raw question alone leaves recall on the table. Two cheap, well-known transforms
fix most of it, both done in one small Haiku call:

  * multi-query   — rephrase / decompose the question into a few alternative search
                    queries, retrieve for each, and fuse. Widens coverage.
  * HyDE          — write a short *hypothetical speech* that would answer the question
                    and embed that too. A fake answer sits much closer to real speeches
                    in embedding space than the question does. (Gao et al., 2022.)

`corrective_rewrite` is the agentic loop's recovery move: when the first pass reranks
poorly, it proposes broader / differently-angled queries from what we learned.

All calls degrade gracefully — on any API/parse error we return empty extras so the
caller still has the original question to retrieve with.
"""
from __future__ import annotations

import json
import re

import anthropic
from dotenv import load_dotenv

from ask_parliament.config import QUERY_TRANSFORM_MODEL, REPO_ROOT

load_dotenv(REPO_ROOT / ".env")

_TRANSFORM_SYSTEM = """You expand a user's question into search queries for a \
retrieval system over European parliamentary speeches (the ParlaMint corpus, many \
languages). Return ONLY a JSON object, no prose, with exactly these keys:
  "sub_queries": a list of 3 short alternative search queries — paraphrases and \
narrower decompositions of the question (English; the retriever is cross-lingual).
  "hypothetical": one or two sentences of a plausible parliamentary speech that would \
answer the question, as if spoken by an MP. Concrete, on-topic, no preamble.
Keep it tight. Output the JSON object only."""

_CORRECTIVE_SYSTEM = """A first retrieval pass for the user's question returned only \
weakly relevant parliamentary speeches (shown below). Propose better search queries to \
find stronger evidence: broaden the framing, try synonyms and related policy terms, and \
decompose into sub-topics. Return ONLY a JSON object with one key "sub_queries": a list \
of 4 short English search queries different from an obvious restatement of the question."""


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(max_retries=1)


def _complete(system: str, user: str, model: str) -> str:
    resp = _client().messages.create(
        model=model, max_tokens=400, system=system,
        messages=[{"role": "user", "content": user}],
    )
    return next((b.text for b in resp.content if b.type == "text"), "")


def _parse_json(text: str) -> dict:
    """Pull a JSON object out of the model's reply, tolerating ```json fences."""
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


def _clean_list(value, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out = [str(v).strip() for v in value if str(v).strip()]
    return out[:limit]


def transform_query(question: str, model: str = QUERY_TRANSFORM_MODEL) -> dict:
    """Return {"sub_queries": [...], "hypothetical": str}. Empty fields on failure."""
    try:
        data = _parse_json(_complete(_TRANSFORM_SYSTEM, f"Question: {question}", model))
    except Exception:  # noqa: BLE001 — transformation is best-effort; never block retrieval
        return {"sub_queries": [], "hypothetical": ""}
    hypo = data.get("hypothetical", "")
    return {
        "sub_queries": _clean_list(data.get("sub_queries"), 3),
        "hypothetical": str(hypo).strip() if isinstance(hypo, str) else "",
    }


def corrective_rewrite(
    question: str, weak_texts: list[str], model: str = QUERY_TRANSFORM_MODEL
) -> list[str]:
    """Propose broader/alternative queries given weakly-relevant first-pass results."""
    snippets = "\n".join(f"- {t[:300]}" for t in weak_texts[:3])
    user = f"Question: {question}\n\nWeakly relevant results:\n{snippets}"
    try:
        return _clean_list(_parse_json(_complete(_CORRECTIVE_SYSTEM, user, model)).get("sub_queries"), 4)
    except Exception:  # noqa: BLE001
        return []
