"""
FastAPI backend exposing the RAG chain via LangServe.

Run with:
    uvicorn api.serve:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Startup: load vector store and build chain once
# ---------------------------------------------------------------------------

_chain = None
_retriever = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chain, _retriever

    from src.retrieval.vectorstore import load_vectorstore
    from src.retrieval.retrievers import build_hybrid_retriever, build_reranking_retriever
    from src.chain.rag_chain import build_rag_chain

    # We need the chunks for BM25 — load them from a saved pickle if available
    chunks_path = Path(os.getenv("FAISS_INDEX_PATH", "data/faiss_index")) / "chunks.pkl"

    vs = load_vectorstore()

    if chunks_path.exists():
        import pickle
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        hybrid = build_hybrid_retriever(chunks, vs)
        _retriever = build_reranking_retriever(hybrid)
    else:
        # Fall back to simple vector retriever if chunks not cached
        _retriever = vs.as_retriever(search_kwargs={"k": int(os.getenv("TOP_N_RERANK", "3"))})

    _chain = build_rag_chain(_retriever)
    print("[serve] RAG chain ready.")
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Bot for AI Research Papers",
    description="Ask questions about academic papers and receive cited answers.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "chain_ready": _chain is not None}


# ---------------------------------------------------------------------------
# LangServe routes — provides /rag/invoke, /rag/stream, /rag/batch
# ---------------------------------------------------------------------------

# We defer route addition until chain is ready; use a lambda to resolve lazily
class _LazyChain:
    """Proxy that forwards calls to the globally loaded chain."""
    def invoke(self, input, config=None):
        if _chain is None:
            raise HTTPException(status_code=503, detail="Chain not loaded yet.")
        return _chain.invoke(input, config)

    def stream(self, input, config=None):
        if _chain is None:
            raise HTTPException(status_code=503, detail="Chain not loaded yet.")
        return _chain.stream(input, config)

    def batch(self, inputs, config=None):
        if _chain is None:
            raise HTTPException(status_code=503, detail="Chain not loaded yet.")
        return _chain.batch(inputs, config)

    # Required by LangServe
    def get_output_schema(self, *args, **kwargs):
        return _chain.get_output_schema(*args, **kwargs)

    def get_input_schema(self, *args, **kwargs):
        return _chain.get_input_schema(*args, **kwargs)


# ---------------------------------------------------------------------------
# Direct /ask endpoint (simpler alternative to LangServe)
# ---------------------------------------------------------------------------

class QuestionRequest(BaseModel):
    question: str
    few_shot: bool = False


class AnswerResponse(BaseModel):
    answer: str


@app.post("/ask", response_model=AnswerResponse)
async def ask(req: QuestionRequest):
    """Ask a question and receive a cited answer."""
    if _chain is None:
        raise HTTPException(status_code=503, detail="Chain not loaded yet.")
    answer = _chain.invoke(req.question)
    return AnswerResponse(answer=answer)


# Optionally mount LangServe (requires langserve installed)
try:
    add_routes(app, _LazyChain(), path="/rag")
except Exception:
    pass  # LangServe optional


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.serve:app", host="0.0.0.0", port=8000, reload=True)
