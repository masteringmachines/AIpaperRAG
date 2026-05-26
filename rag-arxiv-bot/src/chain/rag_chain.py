"""
Assemble the full RAG pipeline using LangChain Expression Language (LCEL).
"""

from __future__ import annotations

import os
from typing import Iterator

from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.chain.prompts import RAG_PROMPT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_docs(docs: list[Document]) -> str:
    """Convert retrieved documents into a single context string with metadata."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        title = doc.metadata.get("original_title") or doc.metadata.get("title", "Unknown")
        authors = doc.metadata.get("original_authors") or doc.metadata.get("authors", "Unknown")
        year = doc.metadata.get("Published", doc.metadata.get("year", ""))[:4] if doc.metadata.get("Published") else ""
        header = f"[{i}] {title} — {authors} {year}".strip(" —")
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _get_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        streaming=True,
    )


# ---------------------------------------------------------------------------
# Chain factory
# ---------------------------------------------------------------------------

def build_rag_chain(retriever, few_shot: bool = False):
    """Build a complete LCEL RAG chain.

    Parameters
    ----------
    retriever:
        A LangChain retriever (simple, hybrid, or reranking).
    few_shot:
        If True, use the few-shot prompt variant.

    Returns
    -------
    A Runnable that accepts a question string and returns an answer string.

    Usage
    -----
    >>> chain = build_rag_chain(retriever)
    >>> print(chain.invoke("What is self-attention?"))
    """
    if few_shot:
        from src.chain.prompts import FEW_SHOT_RAG_PROMPT as prompt
    else:
        prompt = RAG_PROMPT

    llm = _get_llm()

    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def stream_answer(chain, question: str) -> Iterator[str]:
    """Stream answer tokens from the RAG chain.

    Parameters
    ----------
    chain:
        A compiled RAG chain from ``build_rag_chain``.
    question:
        The user's question.

    Yields
    ------
    str tokens as they arrive.
    """
    for token in chain.stream(question):
        yield token
