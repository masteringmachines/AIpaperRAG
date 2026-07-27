🔬 RAG Bot for AI Research Papers
A production-ready Retrieval-Augmented Generation system that answers questions about academic papers from arXiv or local PDFs — with full citations.

Python LangChain License

✨ Features
📥 Ingest papers from arXiv API or local PDFs
🧩 Smart chunking with semantic boundaries
🔍 Hybrid search (BM25 + vector) + cross-encoder reranking
💬 Citation-grounded answers from GPT-4o-mini
📊 Evaluation with RAGAS (faithfulness, relevancy, recall)
🚀 FastAPI backend + Streamlit frontend
🔭 LangSmith tracing for observability
🏗️ Architecture
arXiv / PDFs
     │
     ▼
[Data Ingestion]          ← ArxivLoader / PyPDFLoader
     │
     ▼
[Smart Chunking]          ← RecursiveCharacterTextSplitter / SemanticChunker
     │
     ▼
[Embeddings]              ← BAAI/bge-small-en-v1.5 (HuggingFace)
     │
     ▼
[Vector Store]            ← FAISS (local) or Chroma
     │
     ▼
[Hybrid Retrieval]        ← BM25 + Vector EnsembleRetriever
     │
     ▼
[Reranking]               ← CrossEncoderReranker (bge-reranker-base)
     │
     ▼
[RAG Chain (LCEL)]        ← Prompt + GPT-4o-mini + StrOutputParser
     │
     ▼
[FastAPI + Streamlit]     ← REST API + Web UI
🚀 Quickstart
1. Clone and install
git clone https://github.com/yourusername/rag-arxiv-bot.git
cd rag-arxiv-bot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
2. Configure environment
cp .env.example .env
# Edit .env and add your API keys
3. Ingest papers
# Ingest from arXiv (10 papers on LLM reasoning)
python scripts/ingest.py --source arxiv --query "large language model reasoning" --max-docs 10

# Or ingest local PDFs
python scripts/ingest.py --source pdf --path data/papers/
4. Run Streamlit UI
streamlit run frontend/streamlit_app.py
5. Run FastAPI server
uvicorn api.serve:app --host 0.0.0.0 --port 8000 --reload
🐳 Docker
# API only
docker build -t rag-bot .
docker run -p 8000:8000 --env-file .env rag-bot

# Full stack (API + UI)
docker-compose up
📊 Evaluation
python scripts/evaluate.py
Outputs RAGAS scores: faithfulness, answer_relevancy, context_recall.

📁 Project Structure
rag-arxiv-bot/
├── src/
│   ├── ingestion/        # Data loading & chunking
│   │   ├── loaders.py
│   │   └── chunkers.py
│   ├── retrieval/        # Embeddings, vector store, retrievers
│   │   ├── embeddings.py
│   │   ├── vectorstore.py
│   │   └── retrievers.py
│   ├── chain/            # RAG chain & prompts
│   │   ├── prompts.py
│   │   └── rag_chain.py
│   └── evaluation/       # RAGAS evaluation
│       └── evaluator.py
├── api/
│   └── serve.py          # FastAPI + LangServe
├── frontend/
│   └── streamlit_app.py  # Streamlit UI
├── scripts/
│   ├── ingest.py         # CLI ingestion script
│   └── evaluate.py       # CLI evaluation script
├── tests/                # Unit & integration tests
├── data/papers/          # Local PDFs (gitignored)
├── .env.example
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
⚙️ Configuration
Variable	Default	Description
OPENAI_API_KEY	required	OpenAI API key
LANGCHAIN_API_KEY	optional	LangSmith tracing
EMBEDDING_MODEL	BAAI/bge-small-en-v1.5	HuggingFace model
LLM_MODEL	gpt-4o-mini	OpenAI chat model
FAISS_INDEX_PATH	data/faiss_index	Where to save the index
CHUNK_SIZE	512	Characters per chunk
CHUNK_OVERLAP	50	Overlap between chunks
TOP_K_RETRIEVAL	10	Docs fetched before reranking
TOP_N_RERANK	3	Docs after reranking
🧪 API Endpoints
Method	Path	Description
POST	/rag/invoke	Ask a question, get answer
POST	/rag/stream	Streaming answer
GET	/health	Health check
