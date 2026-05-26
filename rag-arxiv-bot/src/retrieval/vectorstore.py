"""
FAISS vector store — build, save, and load.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document


def _index_path() -> str:
    return os.getenv("FAISS_INDEX_PATH", "data/faiss_index")


def build_vectorstore(chunks: List[Document]):
    """Create a FAISS vector store from document chunks.

    Parameters
    ----------
    chunks:
        List of chunked Document objects.

    Returns
    -------
    FAISS vectorstore object.
    """
    from langchain_community.vectorstores import FAISS
    from src.retrieval.embeddings import get_embeddings

    embeddings = get_embeddings()
    print(f"[vectorstore] Embedding {len(chunks)} chunks into FAISS …")
    vs = FAISS.from_documents(chunks, embeddings)
    print("[vectorstore] FAISS index built.")
    return vs


def save_vectorstore(vs) -> None:
    """Persist the FAISS index to disk."""
    path = _index_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    vs.save_local(path)
    print(f"[vectorstore] Index saved to: {path}")


def load_vectorstore():
    """Load a previously saved FAISS index from disk.

    Returns
    -------
    FAISS vectorstore object.

    Raises
    ------
    FileNotFoundError if the index directory does not exist.
    """
    from langchain_community.vectorstores import FAISS
    from src.retrieval.embeddings import get_embeddings

    path = _index_path()
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No FAISS index found at '{path}'. "
            "Run `python scripts/ingest.py` first."
        )
    embeddings = get_embeddings()
    vs = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    print(f"[vectorstore] Index loaded from: {path}")
    return vs
