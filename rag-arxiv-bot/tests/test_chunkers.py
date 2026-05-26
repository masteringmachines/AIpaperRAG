"""Unit tests for chunking utilities."""

import pytest
from langchain_core.documents import Document


def make_doc(text: str, **meta) -> Document:
    return Document(page_content=text, metadata=meta)


def test_recursive_chunk_basic():
    from src.ingestion.chunkers import recursive_chunk

    docs = [make_doc("A " * 600, title="Test Paper", authors="Smith")]
    chunks = recursive_chunk(docs)
    assert len(chunks) >= 2, "Long document should produce multiple chunks"
    for c in chunks:
        assert "original_title" in c.metadata
        assert len(c.page_content) <= 600  # chunk_size default


def test_metadata_enrichment():
    from src.ingestion.chunkers import recursive_chunk

    docs = [make_doc("Short text.", title="My Paper", authors="Jane Doe")]
    chunks = recursive_chunk(docs)
    assert chunks[0].metadata["original_title"] == "My Paper"
    assert "preview" in chunks[0].metadata
