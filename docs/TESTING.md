# Feature Testing Guide

Complete manual and automated testing reference for the Financial Risk NLP Engine.

---

## Prerequisites

```bash
# Activate your virtual environment
source .venv/bin/activate   # or: .venv\Scripts\activate on Windows

# Ensure the backend is running (in a separate terminal)
python -m src.run --mode replay --episode svb_march_2023 --speed 60

# Ensure the dashboard is running (in another separate terminal)
cd src/dashboard && npm run dev
```

---

## 1. Automated Unit Tests (pytest)

The test suite has **251 tests** across 11 modules. Run all of them:

```bash
pytest tests/ -v
```

Run a specific module:

```bash
pytest tests/test_schema.py -v           # Signal schema validation
pytest tests/test_scorer.py -v           # Impact scorer
pytest tests/test_classifier.py -v      # Zero-shot event classifier
pytest tests/test_alert_engine.py -v    # Alert threshold logic
pytest tests/test_linker.py -v          # Entity→ticker linker
pytest tests/test_rebalancer.py -v      # Index rebalancer weights
pytest tests/test_stress.py -v          # Stress test engine (Module B)
pytest tests/test_replay.py -v          # Replay streamer
pytest tests/test_nlp_pipeline.py -v    # Full NLP pipeline
pytest tests/test_impact.py -v          # Impact calibration
pytest tests/test_api.py -v             # REST API endpoints
```

Run only fast unit tests (skip slow model-loading tests):

```bash
pytest tests/ -v -m "not slow"
```

Expected output: **251 passed** — no failures on a clean install.

---

## 2. NLP Pipeline Features

### 2.1 Sentiment Analysis (FinBERT)

```bash
# Test a single headline via the API
curl -s -X POST http://localhost:8000/signals/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "SVB collapses as depositors withdraw $42B in a single day", "source": "Reuters"}' \
  | python -m json.tool
```

**Expected**: `sentiment` ≈ `-0.8 to -1.0`, `event_type`: `Credit Event`, `impact` ≥ 7

```bash
# Positive headline
curl -s -X POST http://localhost:8000/signals/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple reports record earnings, beating consensus by 15%", "source": "Bloomberg"}' \
  | python -m json.tool
```

**Expected**: `sentiment` ≈ `+0.6 to +1.0`, `event_type`: `Earnings`, `impact` ≥ 5

---

### 2.2 Event Classification (Zero-Shot NLI)

Test each event type the classifier knows about:

| Test Headline | Expected `event_type` |
|---|---|
| `"Fed raises rates by 50bps"` | `Macroeconomic` |
| `"MSFT acquires Activision for $68B"` | `Merger/Acquisition` |
| `"SEC charges Goldman Sachs with insider trading"` | `Regulatory` |
| `"SVB bank run: depositors pull $42B"` | `Credit Event` |
| `"AAPL reports record quarterly earnings"` | `Earnings` |
| `"US-China trade war escalates with new tariffs"` | `Geopolitical` |
| `"Tesla unveils next-gen Model S with 600mi range"` | `Product Launch` |

```bash
# Quick classification test
curl -s -X POST http://localhost:8000/signals/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "Federal Reserve unexpectedly cuts rates by 75bps", "source": "WSJ"}' \
  | python -m json.tool | grep event_type
```

---

## 3. API Endpoints

### 3.1 Health Check

```bash
curl http://localhost:8000/health
```

**Expected response:**
```json
{
  "status": "ok",
  "mode": "replay",
  "live": {"available": false, "reason": "..."},
  "replay": {"available": true, "episodes": ["svb_march_2023"]}
}
```

### 3.2 List Signals

```bash
# All signals
curl "http://localhost:8000/signals"

# Filter by entity
curl "http://localhost:8000/signals?entity=JPM"

# Filter by minimum impact
curl "http://localhost:8000/signals?min_impact=7"

# Filter by time
curl "http://localhost:8000/signals?since=2023-03-10T00:00:00Z"
```

### 3.3 Entity State (Decayed Sentiment)

```bash
# All entities with their current decayed sentiment
curl http://localhost:8000/state

# Single entity
curl http://localhost:8000/state/SVB
```

**Expected**: `decayed_sentiment` value between -1 and +1 per entity.

### 3.4 Portfolio Weights

```bash
# Current rebalancer weights
curl http://localhost:8000/weights
```

**Expected**: 15 tickers with weights summing to ≈ 1.0, respecting min/max bounds from `config.yaml`.

### 3.5 Stress Test

```bash
# Manually trigger a stress test
curl -s -X POST http://localhost:8000/stress/run \
  -H "Content-Type: application/json" \
  -d '{"event_type": "Credit Event", "sentiment": -0.9, "confidence": 0.85, "signal_id": "test-001"}' \
  | python -m json.tool
```

**Expected response fields**: `loss`, `loss_pct`, `breakdown` (bonds / loans / derivatives), `historical_analog`, `triggering_signal`.

---

## 4. Replay Engine (Module A)

### 4.1 Start SVB Replay

```bash
# Start replay at 60x speed
curl -s -X POST http://localhost:8000/replay/start \
  -H "Content-Type: application/json" \
  -d '{"episode": "svb_march_2023", "speed": 60}' \
  | python -m json.tool
```

**Expected**: `"status": "ok"`, mode switches to `"replay"`.

### 4.2 Check Replay Status

```bash
curl http://localhost:8000/health | python -m json.tool | grep -A3 '"mode"'
```

### 4.3 Stop Replay

```bash
curl -s -X POST http://localhost:8000/replay/stop
```

---

## 5. Live Mode

```bash
# Switch to Live (synthetic feed, no API key needed)
curl -s -X POST http://localhost:8000/live/start | python -m json.tool

# Confirm mode is live
curl http://localhost:8000/health | grep mode

# Stop live mode
curl -s -X POST http://localhost:8000/live/stop
```

**Verify**: New signals should appear in `GET /signals` every ~5 seconds while live mode is running.

---

## 6. WebSocket Streams

Test WebSocket endpoints using a tool like [websocat](https://github.com/vi/websocat) or the browser console.

```bash
# Install websocat (if not available)
brew install websocat

# Stream live signals
websocat ws://localhost:8000/ws/signals

# Stream stress test results
websocat ws://localhost:8000/ws/stress

# Stream portfolio weight updates
websocat ws://localhost:8000/ws/weights
```

**Expected**: Each WebSocket emits JSON messages in real time as signals are processed.

---

## 7. Index Rebalancer (Module A)

### 7.1 Verify Initial Equal Weights

```bash
curl http://localhost:8000/weights | python -m json.tool
```

**Expected**: All 15 tickers start near `0.0667` (= 1/15). Weights sum to `1.0`.

### 7.2 Verify Weight Drift After Signals

After the replay runs for ~30 seconds, weights for negatively-impacted tickers (SVB, banking sector) should decrease:

```bash
# Wait ~30s, then check again
curl http://localhost:8000/weights | python -m json.tool
```

**Expected**: `JPM`, `BAC`, `C` weights drift down; safer tickers like `AAPL`, `MSFT` drift up.

---

## 8. Dashboard UI Checks

Open [http://localhost:5173](http://localhost:5173) and manually verify:

| Feature | How to verify |
|---|---|
| **Connection status** | Green indicator top-right when backend is reachable |
| **Live signal feed** | Cards appear with entity name, sentiment bar, event chip, impact badge |
| **Stress test panel** | Populates automatically when a signal hits impact ≥ 7 |
| **Weights chart** | Stacked area chart shifts as replay progresses |
| **Replay mode indicator** | "REPLAY" badge is visible in the top bar |
| **Live/Replay toggle** | Switching shows appropriate mode banner |
| **Empty state** | Load the page before starting replay — should show a clear empty state |

---

## 9. Evaluation Script

```bash
# Run the full NLP evaluation (generates results/results.md)
python scripts/evaluate.py
```

**Expected outputs**:
- Prints sentiment accuracy and macro-F1 to console
- Generates `docs/figures/confusion_matrix.png`
- Writes `results/results.md` with full backtest comparison table

---

## 10. Backtest Script

```bash
python scripts/backtest_rebalancer.py
```

**Expected**: Prints a table comparing NLP Engine vs Equal Weight strategy metrics (Return, Volatility, Sharpe, Max Drawdown) and saves `data/backtest_results.png`.

---

## Known Limitations to Note During Testing

- **Impact model**: Runs in rule-based fallback mode (not the trained XGBoost model). You will see a `WARNING: using rule-based fallback` log line — this is expected.
- **yfinance calls**: If testing on a machine with no internet, `make_sample_data.py` will fail to fetch fresh prices but the committed CSV fallback ensures the demo still works.
- **Zero-shot accuracy**: The event classifier is probabilistic — on ambiguous single-sentence inputs it may occasionally classify into `Other`.
