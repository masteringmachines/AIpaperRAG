"""
Embedding model factory.
"""

from __future__ import annotations

import os


def get_embeddings():
    """Return the configured embedding model.

    Uses HuggingFace ``BAAI/bge-small-en-v1.5`` by default (local, no API key).
    Set ``EMBEDDING_MODEL`` env var to override.

    Returns
    -------
    LangChain Embeddings object.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings

    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    print(f"[embeddings] Loading embedding model: {model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
