#!/usr/bin/env python3
"""
CLI script to ingest papers and build/update the FAISS index.

Examples
--------
# Ingest 10 papers from arXiv on LLM reasoning
python scripts/ingest.py --source arxiv --query "large language model reasoning" --max-docs 10

# Ingest a single PDF
python scripts/ingest.py --source pdf --path data/papers/my_paper.pdf

# Ingest all PDFs in a directory
python scripts/ingest.py --source pdf --path data/papers/

# Use semantic chunking instead of recursive
python scripts/ingest.py --source arxiv --query "RAG retrieval augmented" --chunker semantic
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Ingest papers and build FAISS index.")
    parser.add_argument("--source", choices=["arxiv", "pdf"], required=True)
    parser.add_argument("--query", default="large language model", help="arXiv search query")
    parser.add_argument("--max-docs", type=int, default=10, help="Max papers from arXiv")
    parser.add_argument("--path", help="Path to PDF or directory of PDFs")
    parser.add_argument(
        "--chunker",
        choices=["recursive", "semantic"],
        default="recursive",
        help="Chunking strategy",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Step 1: Load documents
    # -----------------------------------------------------------------------
    from src.ingestion.loaders import load_arxiv, load_pdf, load_pdf_directory

    if args.source == "arxiv":
        docs = load_arxiv(query=args.query, max_docs=args.max_docs)
    else:
        if not args.path:
            print("ERROR: --path is required for --source pdf")
            sys.exit(1)
        p = Path(args.path)
        if p.is_dir():
            docs = load_pdf_directory(p)
        else:
            docs = load_pdf(p)

    if not docs:
        print("No documents loaded. Exiting.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 2: Chunk
    # -----------------------------------------------------------------------
    from src.ingestion.chunkers import recursive_chunk, semantic_chunk

    if args.chunker == "semantic":
        chunks = semantic_chunk(docs)
    else:
        chunks = recursive_chunk(docs)

    # -----------------------------------------------------------------------
    # Step 3: Build and save vector store
    # -----------------------------------------------------------------------
    from src.retrieval.vectorstore import build_vectorstore, save_vectorstore

    vs = build_vectorstore(chunks)
    save_vectorstore(vs)

    # Also save chunks for BM25 (hybrid retrieval)
    index_path = Path(os.getenv("FAISS_INDEX_PATH", "data/faiss_index"))
    chunks_file = index_path / "chunks.pkl"
    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)
    print(f"[ingest] Chunks saved to: {chunks_file}")

    print("\n✅ Ingestion complete!")
    print(f"   Documents: {len(docs)}")
    print(f"   Chunks:    {len(chunks)}")
    print(f"   Index:     {index_path}")
    print("\nNext step: streamlit run frontend/streamlit_app.py")


if __name__ == "__main__":
    main()
