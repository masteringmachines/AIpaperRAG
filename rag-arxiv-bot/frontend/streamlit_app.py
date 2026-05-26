"""
Streamlit frontend for the RAG Bot.

Run with:
    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make sure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Research Paper Q&A",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 AI Research Paper Q&A")
st.caption(
    "Ask questions about ingested arXiv papers or local PDFs. "
    "Answers are grounded in retrieved context with full citations."
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, answer) tuples

if "chain" not in st.session_state:
    st.session_state.chain = None

# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    openai_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Required to generate answers.",
    )
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    use_few_shot = st.toggle("Few-shot prompt", value=False)
    retriever_mode = st.selectbox(
        "Retriever mode",
        ["Hybrid + Reranking (recommended)", "Vector only (fast)"],
    )

    st.divider()
    st.markdown("**Index path**")
    index_path = st.text_input(
        "FAISS index path",
        value=os.getenv("FAISS_INDEX_PATH", "data/faiss_index"),
    )
    os.environ["FAISS_INDEX_PATH"] = index_path

    if st.button("🔄 Load / Reload Chain"):
        with st.spinner("Loading vector store and building chain …"):
            try:
                import pickle
                from src.retrieval.vectorstore import load_vectorstore
                from src.retrieval.retrievers import (
                    build_hybrid_retriever,
                    build_reranking_retriever,
                    build_simple_retriever,
                )
                from src.chain.rag_chain import build_rag_chain

                vs = load_vectorstore()
                chunks_path = Path(index_path) / "chunks.pkl"

                if retriever_mode.startswith("Hybrid") and chunks_path.exists():
                    with open(chunks_path, "rb") as f:
                        chunks = pickle.load(f)
                    hybrid = build_hybrid_retriever(chunks, vs)
                    retriever = build_reranking_retriever(hybrid)
                else:
                    retriever = build_simple_retriever(vs)

                st.session_state.chain = build_rag_chain(retriever, few_shot=use_few_shot)
                st.success("✅ Chain loaded!")
            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.history = []

# ---------------------------------------------------------------------------
# Main — chat interface
# ---------------------------------------------------------------------------

# Display history
for q, a in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

# Input box
question = st.chat_input("Ask a question about AI research papers …")

if question:
    if st.session_state.chain is None:
        st.warning("⚠️ Please load the chain first using the sidebar button.")
    elif not os.getenv("OPENAI_API_KEY"):
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar.")
    else:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_answer = ""
            try:
                for token in st.session_state.chain.stream(question):
                    full_answer += token
                    placeholder.markdown(full_answer + "▌")
                placeholder.markdown(full_answer)
                st.session_state.history.append((question, full_answer))
            except Exception as e:
                st.error(f"Error generating answer: {e}")
