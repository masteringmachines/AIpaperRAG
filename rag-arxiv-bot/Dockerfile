FROM python:3.11-slim

WORKDIR /app

# System deps for unstructured / pypdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directories
RUN mkdir -p data/faiss_index data/papers

EXPOSE 8000

CMD ["uvicorn", "api.serve:app", "--host", "0.0.0.0", "--port", "8000"]
