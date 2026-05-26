"""
Document loaders for arXiv papers and local PDFs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_core.documents import Document


def load_arxiv(query: str, max_docs: int = 10) -> List[Document]:
    """Load papers from arXiv matching *query*.

    Parameters
    ----------
    query:
        Search query string, e.g. "large language model reasoning".
    max_docs:
        Maximum number of papers to retrieve.

    Returns
    -------
    List of LangChain Document objects with metadata:
        title, authors, Published, Summary, entry_id, source.
    """
    from langchain_community.document_loaders import ArxivLoader

    print(f"[ArxivLoader] Fetching up to {max_docs} papers for: '{query}'")
    loader = ArxivLoader(query=query, load_max_docs=max_docs)
    docs = loader.load()
    print(f"[ArxivLoader] Loaded {len(docs)} documents.")
    return docs


def load_pdf(path: str | Path) -> List[Document]:
    """Load all pages from a single PDF file.

    Parameters
    ----------
    path:
        Path to the PDF file.

    Returns
    -------
    List of Document objects, one per page.
    """
    from langchain_community.document_loaders import PyPDFLoader

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    print(f"[PyPDFLoader] Loading: {path.name}")
    loader = PyPDFLoader(str(path))
    pages = loader.load_and_split()
    print(f"[PyPDFLoader] {len(pages)} pages loaded from {path.name}.")
    return pages


def load_pdf_directory(directory: str | Path) -> List[Document]:
    """Recursively load all PDFs from a directory.

    Parameters
    ----------
    directory:
        Path to a folder containing .pdf files.

    Returns
    -------
    Combined list of Documents from all PDFs found.
    """
    directory = Path(directory)
    pdf_files = sorted(directory.rglob("*.pdf"))
    if not pdf_files:
        print(f"[load_pdf_directory] No PDFs found in {directory}.")
        return []

    all_docs: List[Document] = []
    for pdf in pdf_files:
        try:
            all_docs.extend(load_pdf(pdf))
        except Exception as exc:
            print(f"[load_pdf_directory] Skipping {pdf.name}: {exc}")

    print(f"[load_pdf_directory] Total documents loaded: {len(all_docs)}")
    return all_docs


def combine_sources(*source_lists: List[Document]) -> List[Document]:
    """Merge multiple document lists into one.

    Usage
    -----
    >>> docs = combine_sources(arxiv_docs, pdf_docs)
    """
    combined = []
    for lst in source_lists:
        combined.extend(lst)
    print(f"[combine_sources] Total combined documents: {len(combined)}")
    return combined
