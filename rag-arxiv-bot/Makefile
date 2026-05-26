.PHONY: install ingest ui api test docker-up

install:
	pip install -r requirements.txt

ingest-arxiv:
	python scripts/ingest.py --source arxiv --query "large language model reasoning" --max-docs 10

ingest-pdf:
	python scripts/ingest.py --source pdf --path data/papers/

ui:
	streamlit run frontend/streamlit_app.py

api:
	uvicorn api.serve:app --host 0.0.0.0 --port 8000 --reload

evaluate:
	python scripts/evaluate.py

test:
	pytest tests/ -v

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down
