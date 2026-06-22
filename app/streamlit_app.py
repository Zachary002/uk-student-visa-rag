"""Streamlit chat UI for the UK Student Visa RAG assistant.

A clean, native Streamlit interface (no JS frameworks) that wraps the RAG
pipeline: chat bubbles, an expandable Sources panel with clickable official
links and relevance scores, and visually distinct handling for the
"no reliable answer" (refusal) case.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make ``src`` importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import streamlit as st  # noqa: E402

from src.config import settings  # noqa: E402
from src.generation.rag_pipeline import RagResponse, get_pipeline  # noqa: E402

EXAMPLE_QUESTIONS = [
    "How many hours can I work per week during term time?",
    "How do I open a UK bank account as an international student?",
    "How do I register with a GP?",
    "Can I extend my Student visa from inside the UK?",
    "How long can I stay in the UK on a Graduate visa?",
]

st.set_page_config(
    page_title="UK Student Visa & Study-Life Assistant",
    page_icon="🎓",
    layout="centered",
)

# --- Light custom CSS (no JavaScript): header banner + source cards ---
st.markdown(
    """
    <style>
      #MainMenu, footer {visibility: hidden;}
      .app-header {
        background: linear-gradient(135deg, #1d70b8 0%, #003078 100%);
        color: #ffffff; padding: 1.3rem 1.5rem; border-radius: 14px;
        margin-bottom: 1.1rem;
      }
      .app-header h1 { color: #fff; margin: 0; font-size: 1.55rem; }
      .app-header p  { color: #d6e4f2; margin: .35rem 0 0; font-size: .95rem; }
      .src-score { color: #1d70b8; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading models & index (first run takes a moment)…")
def load_pipeline():
    """Build the RAG pipeline once and reuse it across reruns/sessions.

    On a fresh deploy (e.g. Hugging Face Spaces) this also builds the vector
    index from the committed data/processed corpus — no scraping on the server.
    """
    from src.indexing.vector_store import ensure_index

    ensure_index()
    return get_pipeline()


def render_sources(response: RagResponse) -> None:
    """Render the expandable Sources panel for a grounded answer."""
    if not response.sources:
        return
    confidence = int(round(response.top_score * 100))
    with st.expander(f"📄 Sources ({len(response.sources)}) · top match {confidence}%"):
        for src in response.sources:
            score = int(round(src.score * 100))
            link = f"[{src.title}]({src.source_url})" if src.source_url else src.title
            st.markdown(
                f"**[{src.index}]** {link} &nbsp;·&nbsp; "
                f"<span class='src-score'>relevance {score}%</span>",
                unsafe_allow_html=True,
            )
            st.caption(src.snippet + "…")


def render_assistant(message: dict) -> None:
    """Render one stored assistant turn (answer + sources, or a refusal)."""
    if message.get("has_answer", True):
        st.markdown(message["content"])
        render_sources(message["response"])
    else:
        st.warning(message["content"], icon="⚠️")


# --- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("🎓 About")
    st.markdown(
        "A Retrieval-Augmented Generation assistant for **international students "
        "in the UK** — answering questions on the Student/Graduate visa, the right "
        "to work, banking, and NHS/GP registration.\n\n"
        "Answers are grounded in official sources (**gov.uk, UKCISA, NHS**) with "
        "inline citations, and the assistant **refuses rather than guesses** when "
        "it can't find a reliable answer."
    )

    st.subheader("Try an example")
    for question in EXAMPLE_QUESTIONS:
        if st.button(question, use_container_width=True):
            st.session_state.pending_query = question

    st.divider()
    st.caption(f"LLM: `{settings.llm.model}`  ·  Embeddings: `bge-small-en-v1.5`")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.warning(
        "Educational demo — **not official immigration advice**. Always verify "
        "with gov.uk or your university's international office.",
        icon="ℹ️",
    )

# --- Header ---------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
      <h1>🎓 UK Student Visa & Study-Life Assistant</h1>
      <p>Grounded answers with sources — visas, work rights, banking & NHS.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load the pipeline (clear error if the API key isn't set yet).
try:
    pipeline = load_pipeline()
except Exception as exc:  # noqa: BLE001 - surface any startup failure to the user
    st.error(f"Could not start the assistant.\n\n**{type(exc).__name__}:** {exc}")
    st.stop()

# --- Chat state & history --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant(message)
        else:
            st.markdown(message["content"])

# --- Handle new input (typed or example button) ----------------------------
typed = st.chat_input("Ask about visas, work, banking, or the NHS…")
user_query = typed or st.session_state.pop("pending_query", None)

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base…"):
            response = pipeline.answer(user_query)
        if response.has_answer:
            st.markdown(response.answer)
            render_sources(response)
        else:
            st.warning(response.answer, icon="⚠️")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.answer,
            "has_answer": response.has_answer,
            "response": response,
        }
    )
