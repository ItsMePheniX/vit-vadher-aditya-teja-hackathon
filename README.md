# vitbhopal-vadher-aditya-teja-hackathon

> **Author**: Vadher Aditya Teja | **Deadline**: 12 Oct 2026
> **Tested on**: macOS 14+, Python 3.11, Node.js v18.20+

## Project: Financial Risk NLP Engine

A real-time financial risk intelligence platform that ingests news / market events, runs **FinBERT** sentiment analysis, classifies events, estimates market impact and signal half-life, and surfaces live alerts through a React dashboard.

> **Demo Video Link**: [Insert YouTube/Loom Link Here]

## Datasets
All data is **public or synthetically generated**. No proprietary S&P Global / Crisil client data is used.
- **News**: Subset of the public All-the-News dataset and synthetic samples.
- **Social Media**: Synthetic tweets replicating the SVB 2023 crash.
- **Prices**: Fetched dynamically via `yfinance` caching.
See [`data/DATA_SOURCES.md`](data/DATA_SOURCES.md) for full provenance.

---

## Architecture

```
News Feed (synthetic / NewsAPI)
        │
        ▼
┌─────────────────────────────┐
│   src/engine/               │
│   ├── ingestor.py  (asyncio queue, JSONL append-only)
│   ├── nlp.py       (FinBERT sentiment)
│   ├── classifier.py (event_type)
│   ├── scorer.py    (impact 1-10, confidence, half_life)
│   └── pipeline.py  (orchestrates above)
└───────────┬─────────────────┘
            │  Signal (frozen schema)
            ▼
┌─────────────────────────────┐
│   src/api/   FastAPI        │
│   ├── main.py               │
│   ├── routes/signals.py     │
│   └── routes/health.py      │
└───────────┬─────────────────┘
            │  REST / WebSocket
            ▼
┌─────────────────────────────┐
│   src/dashboard/  React+Vite│
│   Recharts live charts       │
└─────────────────────────────┘
            │
│   src/modules/              │
│   ├── stress_test/          │
│   └── rebalancer/           │
└─────────────────────────────┘
```

## Signal Schema (frozen)

```json
{
  "id": "str",
  "ts": "ISO8601",
  "entity": "ticker or MARKET",
  "sentiment": "float [-1, 1]",
  "event_type": "Geopolitical|Macroeconomic|Credit Event|Merger/Acquisition|Earnings|Regulatory|Product Launch|Other",
  "impact": "int [1, 10]",
  "confidence": "float [0, 1]",
  "half_life_hours": "float",
  "evidence": ["str"],
  "sources": ["str"]
}
```

## Quick Start (Clean Machine)

Follow these exact steps to run the engine from a fresh clone:

```bash
# 1. Clone & enter
git clone <repo-url>
cd vit-vadher-aditya-teja-hackathon

# 2. Create venv and install dependencies
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Generate sample data (or use the committed synthetic samples)
python scripts/make_sample_data.py

# 4. Warm up the ML cache (Downloads FinBERT & NLI models for offline demo use)
python scripts/warm_cache.py

# 5. Run the Engine & trigger the SVB Replay in one command
#    Alternatively, use `bash scripts/demo.sh`
python -m src.run --mode replay --episode svb_march_2023 --speed 60
```

Open a **separate terminal** to start the dashboard:

```bash
cd src/dashboard
npm install
npm run dev
```

API docs → http://localhost:8000/docs  
Dashboard → http://localhost:5173

## Using Docker (Recommended for Jury)

You can launch the entire stack (Backend + Dashboard) with a single command:

```bash
docker compose up --build
```
- The API will be available at `http://localhost:8000/docs`
- The Dashboard will be available at `http://localhost:5173`

## Running Tests

```bash
pytest tests/ -v
```

## Project Structure

```
.
├── src/
│   ├── engine/       # NLP pipeline core
│   ├── api/          # FastAPI server
│   ├── modules/      # Stress test and NLP Rebalancer
│   └── dashboard/    # React + Vite frontend
├── data/             # Small sample / synthetic datasets
├── docs/             # Architecture, assumptions, presentation
├── tests/            # pytest test suite
└── scripts/          # Data download, model training, backtesting, & report generation
```

## Data & Compliance

- All data is **public or synthetically generated**
- No proprietary S&P Global / Crisil client data
- See [`data/DATA_SOURCES.md`](data/DATA_SOURCES.md) for full provenance
- See [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md) for design decisions

## License

MIT — see [LICENSE](LICENSE)
