# vitbhopal-vadher-aditya-teja-hackathon

> **S&P Global & Crisil Campus Hackathon 2026** | Individual Submission  
> **Author**: Vadher Aditya Teja | **Deadline**: 12 Oct 2026

## 🚀 Project: Financial Risk NLP Engine

A real-time financial risk intelligence platform that ingests news / market events, runs **FinBERT** sentiment analysis, classifies events, estimates market impact and signal half-life, and surfaces live alerts through a React dashboard.

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
┌─────────────────────────────┐
│   src/modules/              │
│   ├── portfolio_risk.py     │
│   ├── alert_engine.py       │
│   └── report_gen.py         │
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

## Quick Start

```bash
# 1. Clone & enter
git clone <repo-url>
cd vitbhopal-vadher-aditya-teja-hackathon

# 2. Create venv with uv and install deps
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# 3. Copy env and configure
cp .env.example .env
# Edit .env if you have API keys (optional)

# 4. Run the engine + API
python -m src.api.main

# 5. Dashboard (separate terminal)
cd src/dashboard
npm install && npm run dev
```

API docs → http://localhost:8000/docs  
Dashboard → http://localhost:5173

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
│   ├── modules/      # Portfolio risk, alerts, reports
│   └── dashboard/    # React + Vite frontend
├── data/             # Small sample / synthetic datasets
├── docs/             # Architecture, assumptions, presentation
├── tests/            # pytest test suite
└── scripts/          # Data download & utility scripts
```

## Data & Compliance

- All data is **public or synthetically generated**
- No proprietary S&P Global / Crisil client data
- See [`data/DATA_SOURCES.md`](data/DATA_SOURCES.md) for full provenance
- See [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md) for design decisions

## License

MIT — see [LICENSE](LICENSE)
