PYTHON    := .venv/bin/python
UV        := uv
DATA      := data/svb_news_sample.csv
DASHBOARD := src/dashboard

.PHONY: setup run test replay evaluate sample download dashboard dashboard-build clean

## setup: create venv and install all Python dependencies
setup:
	$(UV) venv --python 3.11
	$(UV) pip install -r requirements.txt

## run: start the FastAPI engine (API + ingestor) on :8000
run:
	PYTHONPATH=. $(PYTHON) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

## dashboard: start the Vite dev server on :5173 (proxies /api/* to :8000)
dashboard:
	cd $(DASHBOARD) && npm run dev

## dashboard-build: build the dashboard for production
dashboard-build:
	cd $(DASHBOARD) && npm run build

## test: run the full pytest suite
test:
	PYTHONPATH=. $(PYTHON) -m pytest tests/ -v

## replay: replay SVB March-2023 sample data at 60× through the NLP pipeline
replay:
	PYTHONPATH=. $(PYTHON) scripts/run_pipeline.py \
		--news   data/svb_news_sample.csv \
		--tweets data/svb_tweets_sample.csv \
		--start  2023-03-08 \
		--end    2023-03-17 \
		--speed  1000000

## evaluate: score pipeline outputs against gold labels (stub)
evaluate:
	PYTHONPATH=. $(PYTHON) scripts/evaluate.py

## sample: regenerate small sample files from full datasets (or synthetic)
sample:
	PYTHONPATH=. $(PYTHON) scripts/make_sample_data.py

## download: download full public datasets via Kaggle CLI
download:
	PYTHONPATH=. $(PYTHON) scripts/download_data.py --all

## train: calibrate the impact model with real prices (requires network)
train:
	PYTHONPATH=. $(PYTHON) scripts/train_impact.py

## train-offline: calibrate the impact model with synthetic prices (no network)
train-offline:
	PYTHONPATH=. $(PYTHON) scripts/train_impact.py --offline

## clean: remove venv and Python caches
clean:
	rm -rf .venv __pycache__ .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
