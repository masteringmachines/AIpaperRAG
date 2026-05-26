#!/usr/bin/env python3
"""
Run RAGAS evaluation on the RAG bot.

Usage
-----
    python scripts/evaluate.py

Edit the ``SAMPLE_QA`` list below to add your own test questions.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Sample QA pairs — add your own!
# ---------------------------------------------------------------------------

SAMPLE_QA = [
    {"question": "What is chain-of-thought prompting and why does it work?"},
    {"question": "How does RAG (Retrieval-Augmented Generation) reduce hallucinations?"},
    {"question": "What is the difference between LoRA and full fine-tuning?"},
    {"question": "How does self-attention work in transformer models?"},
    {"question": "What are the main benchmarks for evaluating LLM reasoning?"},
]


def main():
    # -----------------------------------------------------------------------
    # Load chain
    # -----------------------------------------------------------------------
    from src.retrieval.vectorstore import load_vectorstore
    from src.retrieval.retrievers import (
        build_hybrid_retriever,
        build_reranking_retriever,
        build_simple_retriever,
    )
    from src.chain.rag_chain import build_rag_chain
    from src.evaluation.evaluator import evaluate_from_chain

    print("[evaluate] Loading vector store …")
    vs = load_vectorstore()

    index_path = Path(os.getenv("FAISS_INDEX_PATH", "data/faiss_index"))
    chunks_path = index_path / "chunks.pkl"

    if chunks_path.exists():
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        hybrid = build_hybrid_retriever(chunks, vs)
        retriever = build_reranking_retriever(hybrid)
    else:
        print("[evaluate] Chunks not found, using simple retriever.")
        retriever = build_simple_retriever(vs)

    chain = build_rag_chain(retriever)

    # -----------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------
    print(f"[evaluate] Evaluating {len(SAMPLE_QA)} questions with RAGAS …\n")
    results = evaluate_from_chain(chain, retriever, SAMPLE_QA)

    print("\n" + "=" * 50)
    print("RAGAS Evaluation Summary")
    print("=" * 50)
    for metric, score in results.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {metric:<25} {bar} {score:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
