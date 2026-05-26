"""
Chunking strategies for research paper documents.
"""

from __future__ import annotations

import os
from typing import List

from langchain_core.documents import Document


def _chunk_size() -> int:
    return int(os.getenv("CHUNK_SIZE", "512"))


def _chunk_overlap() -> int:
    return int(os.getenv("CHUNK_OVERLAP", "50"))


def recursive_chunk(documents: List[Document]) -> List[Document]:
    """Split documents using RecursiveCharacterTextSplitter.

    Respects paragraph and sentence boundaries.  Best for most use cases.

    Parameters
    ----------
    documents:
        Raw list of Document objects (from loaders).

    Returns
    -------
    List of smaller Document chunks with preserved metadata.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_chunk_size(),
        chunk_overlap=_chunk_overlap(),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    _enrich_metadata(chunks)
    print(f"[recursive_chunk] {len(documents)} docs → {len(chunks)} chunks.")
    return chunks


def semantic_chunk(documents: List[Document]) -> List[Document]:
    """Split documents using SemanticChunker (embedding-based).

    Requires ``langchain-experimental`` and an OpenAI API key.
    Slower but better for dense technical content.

    Parameters
    ----------
    documents:
        Raw list of Document objects.

    Returns
    -------
    Semantically coherent chunks.
    """
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_openai import OpenAIEmbeddings

    splitter = SemanticChunker(OpenAIEmbeddings())
    chunks = splitter.split_documents(documents)
    _enrich_metadata(chunks)
    print(f"[semantic_chunk] {len(documents)} docs → {len(chunks)} chunks.")
    return chunks


def _enrich_metadata(chunks: List[Document]) -> None:
    """Add extra metadata fields useful for citation generation."""
    for chunk in chunks:
        # Preserve the original title for citation
        chunk.metadata.setdefault("original_title", chunk.metadata.get("title", "Unknown"))
        # Preserve authors
        chunk.metadata.setdefault("original_authors", chunk.metadata.get("authors", "Unknown"))
        # Truncate long page_content previews for debugging
        chunk.metadata["preview"] = chunk.page_content[:80].replace("\n", " ")
