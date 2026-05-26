"""
Advanced retrieval: hybrid search (BM25 + vector) + cross-encoder reranking.
"""

from __future__ import annotations

import os
from typing import List

from langchain_core.documents import Document


def _top_k() -> int:
    return int(os.getenv("TOP_K_RETRIEVAL", "10"))


def _top_n() -> int:
    return int(os.getenv("TOP_N_RERANK", "3"))


def build_hybrid_retriever(chunks: List[Document], vectorstore):
    """Ensemble BM25 + vector retriever.

    BM25 handles keyword precision; vector handles semantic similarity.
    Weighted 40% BM25 / 60% vector by default.

    Parameters
    ----------
    chunks:
        The same chunks used to build the vector store (needed for BM25).
    vectorstore:
        A FAISS vectorstore.

    Returns
    -------
    EnsembleRetriever
    """
    from langchain.retrievers import BM25Retriever, EnsembleRetriever

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = _top_k()

    vector = vectorstore.as_retriever(search_kwargs={"k": _top_k()})

    ensemble = EnsembleRetriever(
        retrievers=[bm25, vector],
        weights=[0.4, 0.6],
    )
    print("[retrievers] Hybrid retriever (BM25 + vector) ready.")
    return ensemble


def build_reranking_retriever(base_retriever):
    """Wrap a retriever with cross-encoder reranking.

    Uses ``BAAI/bge-reranker-base`` to score and reorder retrieved chunks.

    Parameters
    ----------
    base_retriever:
        Any LangChain retriever (e.g., the hybrid retriever).

    Returns
    -------
    ContextualCompressionRetriever with reranking.
    """
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=_top_n())

    reranking_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )
    print(f"[retrievers] Reranking retriever ready (top_n={_top_n()}).")
    return reranking_retriever


def build_simple_retriever(vectorstore):
    """Basic vector-only retriever.  Useful for testing without BM25/reranking."""
    return vectorstore.as_retriever(search_kwargs={"k": _top_n()})
