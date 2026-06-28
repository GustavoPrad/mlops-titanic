.PHONY: install eda pipeline api app watch test docker-up docker-down clean

install:
	pip install -r requirements.txt --break-system-packages

eda:
	python -m src.eda

pipeline:
	python -m src.pipeline

pipeline-force:
	python -m src.pipeline --force-promote

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

app:
	streamlit run app/streamlit_app.py

watch:
	python -m src.auto_retrain --watch

retrain-once:
	python -m src.auto_retrain --once

test:
	pytest tests/ -v

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	rm -rf data/processed data/incoming models reports __pycache__ .pytest_cache
	mkdir -p data/processed models reports
