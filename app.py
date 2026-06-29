"""Ask Parliament — Streamlit chat UI (thin client over the FastAPI backend).

This frontend holds no models and no vector store: it calls the API
(`ask_parliament.api`) over HTTP for everything — filter facets via /meta and
grounded answers via /ask — so it stays light enough to run in a tiny container.
Point it at the backend with the API_URL env var (default http://localhost:8000).

Run the API first, then:  streamlit run app.py
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Ask Parliament", page_icon="🏛️", layout="wide")


@st.cache_data(show_spinner="Reading filter options from the API…")
def get_meta() -> dict:
    r = requests.get(f"{API_URL}/meta", timeout=30)
    r.raise_for_status()
    return r.json()


def ask_api(payload: dict) -> dict:
    r = requests.post(f"{API_URL}/ask", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


def render_sources(sources: list[dict], show_english: bool = False) -> None:
    """Expandable list of the speeches an answer was grounded in.

    Sources arrive as plain dicts from the API; wrap each in a namespace so the
    rendering reads as attribute access. Each shows its similarity (and the
    cross-encoder rerank score when present); numbering matches the [n] citations.
    When `show_english` is on, the speech body is swapped for ParlaMint's English
    machine translation (`text_en`, returned with the source), falling back to the
    original if a translation isn't indexed.
    """
    if not sources:
        return
    hits = [SimpleNamespace(**s) for s in sources]
    with st.expander(f"Sources ({len(hits)})"):
        if show_english:
            st.caption("🌐 Showing ParlaMint's machine translation to English — faithful, "
                       "not official; consult the original for exact wording.")
        for i, h in enumerate(hits, 1):
            score = f"similarity {h.similarity:.3f}"
            if h.rerank_score is not None:
                score += f" · rerank {h.rerank_score:.3f}"
            status = f", {h.party_status}" if h.party_status not in ("-", "") else ""
            st.markdown(
                f"**[{i}]** {h.speaker} ({h.party}{status}) · {h.date} · "
                f"{h.country} · *{h.cap_domain}* · {score}"
            )
            text = (getattr(h, "text_en", None) or h.text) if show_english else h.text
            st.markdown(f"> {text.strip()}")
            if i < len(hits):
                st.divider()


try:
    meta = get_meta()
except Exception as err:  # noqa: BLE001 — surface a clear message if the backend is down
    st.error(f"Can't reach the API at {API_URL} — is the backend running? ({err})")
    st.stop()

# --- Sidebar filters (apply to the next question asked) --------------------
with st.sidebar:
    st.header("Filters")
    st.caption("Filters apply to the next question you ask.")

    sel_countries = st.multiselect("Country", meta["countries"], default=meta["countries"])
    year_from, year_to = st.slider(
        "Year range", meta["year_min"], meta["year_max"],
        (meta["year_min"], meta["year_max"]),
    )
    sel_domains = st.multiselect(
        "Policy domain (optional)", meta["cap_domains"],
        help="Leave empty (the default) to search all speeches. Domain labels are "
        "approximate — assigned per debate segment — so filtering may miss relevant "
        "speeches. Semantic search already finds on-topic speeches without it.",
    )
    top_k = st.slider("Speeches to retrieve (top-k)", 3, 20, 8)
    agentic = st.checkbox(
        "🧠 Agentic retrieval", value=True,
        help="Query transformation (paraphrases + a hypothetical-answer / HyDE query) + "
        "cross-encoder reranking + self-correcting re-retrieval when the top results look "
        "weak. Better recall and precision; adds an LLM call so it's slower. Turn off for "
        "fast plain semantic search.",
    )

    st.divider()
    model = st.selectbox(
        "Generation model", meta["models"],
        help="Haiku is cheaper/faster; Sonnet is higher quality.",
    )
    translate_sources = st.checkbox(
        "🌐 Show sources in English", value=False,
        help="Show the cited speeches in English. They're stored in each parliament's "
        "original language; this displays ParlaMint's machine translation (ParlaMint-en). "
        "The answer itself is always in English.",
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# --- Header ----------------------------------------------------------------
st.title("🏛️ Ask Parliament")
st.caption(
    f"Grounded question-answering over {len(meta['countries'])} European parliaments "
    f"(ParlaMint, {meta['year_min']}–{meta['year_max']}). Speeches are in their original "
    "language; answers are in English and cite only retrieved speeches."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay history (Streamlit reruns the whole script on every interaction).
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []), translate_sources)
            if msg.get("caption"):
                st.caption(msg["caption"])

# --- Handle a new question -------------------------------------------------
if prompt := st.chat_input("Ask about parliamentary debates…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Selecting every country is the same as no country filter — send null.
    countries = None if set(sel_countries) == set(meta["countries"]) else (sel_countries or None)
    spinner_msg = (
        "Transforming the query, retrieving, reranking, self-correcting…"
        if agentic else "Retrieving speeches and generating an answer…"
    )

    with st.chat_message("assistant"):
        with st.spinner(spinner_msg):
            try:
                result = ask_api({
                    "question": prompt, "k": top_k, "countries": countries,
                    "year_from": year_from, "year_to": year_to,
                    "cap_domains": sel_domains or None, "model": model, "agentic": agentic,
                })
            except Exception as err:  # noqa: BLE001
                st.error(f"Request failed: {err}")
                st.stop()

        st.markdown(result["answer"])
        render_sources(result["sources"], translate_sources)

        caption = (
            f"{result['model']} · {result['input_tokens']} in / {result['output_tokens']} out tokens · "
            f"retrieval {result['retrieval_s']:.2f}s · generate {result['latency_s']:.2f}s"
        )
        if agentic and result.get("trace"):
            tr = result["trace"]
            caption += f" · 🧠 {tr['n_variants']} query variants, top rerank {tr['top_rerank']}"
            if tr["corrected"]:
                caption += f", self-corrected ×{tr['rounds']}"
        st.caption(caption)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "caption": caption,
    })
