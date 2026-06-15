"""Ask Parliament — Streamlit chat UI.

Run from the repo root:
    streamlit run app.py

Sidebar filters constrain retrieval; each answer shows an expandable Sources
section and a token/latency caption. The retriever (and the BGE-m3 model it
loads) is cached for the whole app run, so the model loads once, not per question.
"""
from __future__ import annotations

import streamlit as st

from ask_parliament.config import GENERATION_MODEL, GENERATION_MODEL_QUALITY
from ask_parliament.generation import generate_answer
from ask_parliament.hybrid import HybridRetriever
from ask_parliament.retrieval import Retriever

st.set_page_config(page_title="Ask Parliament", page_icon="🏛️", layout="wide")


@st.cache_resource(show_spinner="Loading retriever and embedding model…")
def get_retriever() -> Retriever:
    """One Retriever per app run — loads BGE-m3 once, not per question."""
    return Retriever()


@st.cache_resource(show_spinner="Building keyword (BM25) index…")
def get_hybrid(_retriever: Retriever) -> HybridRetriever:
    """Hybrid retriever, built once per app run (BM25 index over all speeches)."""
    return HybridRetriever(_retriever)


@st.cache_data(show_spinner="Reading filter options…")
def get_facets(_retriever: Retriever):
    # _retriever is prefixed with "_" so Streamlit doesn't try to hash it.
    return _retriever.facets()


def render_sources(sources) -> None:
    """Expandable list of the speeches an answer was grounded in."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, h in enumerate(sources, 1):
            status = f", {h.party_status}" if h.party_status not in ("-", "") else ""
            st.markdown(
                f"**[{i}]** {h.speaker} ({h.party}{status}) · {h.date} · "
                f"{h.country} · *{h.cap_domain}* · similarity {h.similarity:.3f}"
            )
            st.markdown(f"> {h.text.strip()}")
            if i < len(sources):
                st.divider()


retriever = get_retriever()
facets = get_facets(retriever)

# --- Sidebar filters (apply to the next question asked) --------------------
with st.sidebar:
    st.header("Filters")
    st.caption("Filters apply to the next question you ask.")

    sel_countries = st.multiselect("Country", facets.countries, default=facets.countries)
    year_from, year_to = st.slider(
        "Year range", facets.year_min, facets.year_max,
        (facets.year_min, facets.year_max),
    )
    sel_domains = st.multiselect(
        "Policy domain (optional)", facets.cap_domains,
        help="Leave empty (the default) to search all speeches. Domain labels are "
        "approximate — assigned per debate segment — so filtering may miss relevant "
        "speeches. Semantic search already finds on-topic speeches without it.",
    )
    top_k = st.slider("Speeches to retrieve (top-k)", 3, 20, 8)
    retrieval_mode = st.radio(
        "Retrieval", ["Hybrid (semantic + keyword)", "Semantic only"],
        help="Hybrid fuses BM25 keyword matching with dense vector search via "
        "reciprocal rank fusion — better on exact terms (names, specific phrases).",
    )

    st.divider()
    model = st.selectbox(
        "Generation model", [GENERATION_MODEL, GENERATION_MODEL_QUALITY],
        help="Haiku is cheaper/faster; Sonnet is higher quality.",
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# --- Header ----------------------------------------------------------------
st.title("🏛️ Ask Parliament")
st.caption(
    "Grounded question-answering over Austrian parliamentary speeches "
    f"({facets.year_min}–{facets.year_max}). Answers cite only retrieved speeches."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay history (Streamlit reruns the whole script on every interaction).
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))
            if msg.get("caption"):
                st.caption(msg["caption"])

# --- Handle a new question -------------------------------------------------
if prompt := st.chat_input("Ask about parliamentary debates…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    searcher = retriever if retrieval_mode == "Semantic only" else get_hybrid(retriever)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving speeches and generating an answer…"):
            hits = searcher.search(
                prompt,
                k=top_k,
                countries=sel_countries or None,
                year_from=year_from,
                year_to=year_to,
                cap_domains=sel_domains or None,
            )
            result = generate_answer(prompt, hits, model=model)

        st.markdown(result.answer)
        render_sources(result.sources)

        retrieval_s = sum(searcher.last_timings.values())  # works for both retrievers
        caption = (
            f"{result.model} · {result.input_tokens} in / {result.output_tokens} out tokens · "
            f"retrieval {retrieval_s:.2f}s · generate {result.latency_s:.2f}s"
        )
        st.caption(caption)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result.answer,
        "sources": result.sources,
        "caption": caption,
    })
