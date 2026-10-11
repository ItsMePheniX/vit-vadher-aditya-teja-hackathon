# Dashboard UI Testing & Demo Guide

A complete walkthrough for testing the Financial Risk NLP Engine end-to-end directly through the web dashboard.

---

## 1. Quick Launch

Before opening the browser, start the backend and frontend services.

### Option A: One-Command Launch (Recommended)
```bash
bash scripts/demo.sh
```
This automatically boots both the FastAPI backend (`http://localhost:8000`) and the Vite React dashboard (`http://localhost:5173`).

### Option B: Separate Terminals

**Terminal 1 — Backend API Engine:**
```bash
source .venv/bin/activate
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — React Dashboard:**
```bash
cd src/dashboard
npm run dev
```

Open your browser to: **`http://localhost:5173`**

---

## 2. Interface Layout Overview

| Panel | Location | Purpose |
|---|---|---|
| **Top Bar** | Header | Shows connection status (`WS LIVE`), total ingested signal count, and active engine mode. |
| **KPI Grid** | Top Row | High-level metrics: Total Signals, Average Sentiment, Entity Count, 95% Value at Risk (VaR), and High-Risk Alerts count. |
| **Signal Stream** | Top Left | Real-time tabular feed showing incoming signals with entity badges, event type tags, sentiment bar, impact score (1–10), and corroboration indicators. |
| **Engine Mode Controls** | Top Right | Switch between **Live** and **Replay** modes, select crisis episodes, adjust playback speed (1x to 300x), and view simulated UTC crisis time. |
| **Stress Test Panel** | Top Right (below mode) | Hero risk shock analysis: displays portfolio before/after comparison bar chart, total dollar drawdown, asset class loss breakdown, and exact NLP evidence citations. |
| **Sentiment Over Time** | Bottom Left | Real-time time-series chart tracking market sentiment trajectories. |
| **Entity Sentiment Heatmap** | Bottom Left | Entity-level breakdown showing net sentiment per tracked organization/ticker. |
| **Index Weights Chart** | Bottom Left | Module A dynamic rebalancing visualization displaying portfolio asset weight adjustments in response to incoming sentiment. |
| **Live Event Feed & Ingest Box** | Bottom Center | Interactive terminal-style feed with an inline headline test input (`Paste a headline to analyse…`). |
| **Risk Alerts Panel** | Bottom Right | Live feed of critical alerts triggered when impact ≥ 7 or sentiment ≤ −0.40. |

---

## 3. Step-by-Step UI Test Scenarios

### Test Scenario 1: Historical Crisis Replay (SVB March 2023)
**Objective**: Demonstrate that the engine processes streaming news and social media asynchronously without external network dependencies.

1. Navigate to the **Engine Mode** card in the top right.
2. Select the **Replay** tab.
3. Select Episode: **`svb_march_2023`**.
4. Adjust the **Speed Slider**:
   - Set to **`60x`** (1 minute of real time = 1 hour of crisis time) or **`120x`** for rapid demonstration.
5. Click **Start Replay**.
6. **What to Observe**:
   - The status badge changes to **`REPLAY`**.
   - The **Simulated Clock** appears showing UTC timestamps advancing in March 2023.
   - The **Signal Stream** table immediately starts streaming rows with animated flash highlights.
   - The **KPI Grid** updates dynamically: signal count increments, average sentiment turns negative.
   - The **Live Event Feed** displays ticker items as news headlines arrive.

---

### Test Scenario 2: Interactive Headline Analysis (NLP Pipeline Test)
**Objective**: Test zero-shot classification, FinBERT sentiment scoring, and impact calibration directly from the UI input.

1. Scroll to the **Live Event Feed** card in the bottom center.
2. Locate the text input: `"Paste a headline to analyse…"`.
3. Try each of the following test headlines:

#### Test 2A: Credit Crisis (High Impact Negative)
- **Input**:
  ```text
  SVB collapses as depositors withdraw $42B in a single day amid liquidity crisis
  ```
- Click **Analyse**.
- **What to Observe**:
  - Notification displays: `Success: 1 signal(s) produced`.
  - The signal instantly appears at the top of the **Signal Stream** table.
  - Entity: `SIVB` / `SVB`.
  - Event Type: `Credit Event`.
  - Sentiment: Strongly negative (`−0.85` to `−1.00`, styled in red).
  - Impact Score: `≥ 7` (amber/red badge).
  - High-risk alert is triggered in the **Risk Alerts** panel.

#### Test 2B: Positive Earnings (Low Impact Positive)
- **Input**:
  ```text
  Apple reports record quarterly revenue beating Wall Street expectations by 15%
  ```
- Click **Analyse**.
- **What to Observe**:
  - Entity: `AAPL`.
  - Event Type: `Earnings`.
  - Sentiment: Strongly positive (`+0.60` to `+0.95`, styled in green).
  - Impact Score: Low/moderate (`3` to `5`).

#### Test 2C: Macroeconomic / Central Bank
- **Input**:
  ```text
  Federal Reserve raises interest rates by 50 basis points to fight inflation
  ```
- Click **Analyse**.
- **What to Observe**:
  - Event Type: `Macroeconomic`.
  - Sentiment: Neutral-to-negative.
  - System captures interest rate shock attributes.

---

### Test Scenario 3: Module B — Automated Stress Testing
**Objective**: Verify that high-severity signals trigger portfolio stress tests with full explainability.

1. High-severity signals (`impact ≥ 7` and `confidence ≥ 55%`, e.g., the SVB headline from Scenario 2A or during the SVB replay) automatically trigger Module B.
2. Observe the **Stress Test** panel on the right:
   - Header badge indicates **`TRIGGERED`**.
   - **Portfolio Before vs. After Bars**: Visual bar comparison showing the dollar decline (e.g., −$1.2M or −1.2%).
   - **Loss by Asset Class**: Breakdown across:
     - **Bond**: Duration risk and price declines.
     - **Loan**: Credit spread widening and default probability.
     - **Derivative**: Volatility shifts.
   - **Triggering Signal Card**:
     - Shows the exact triggering entity and event type.
     - Displays the **Evidence Quote** highlighting the specific text snippet responsible for the shock.
     - Displays applied shock parameters: Equity %, Rate bp, Credit spread bp, FX %, and Shock Multiplier.
   - **Recent Stress Events**: History log tracking previously triggered stress scenarios.

---

### Test Scenario 4: Module A — Dynamic Portfolio Rebalancing
**Objective**: Verify that sentiment shifts trigger automated risk adjustments.

1. Observe the **Index Weights** chart in the bottom-left column.
2. During continuous replay or after ingesting negative signals for banking entities:
   - Entity sentiment decays over time.
   - The engine automatically reduces index exposure for impaired entities (e.g., SIVB, KRE index constituents).
   - Capital is reallocated toward safer assets while respecting max-turnover risk constraints.
3. Observe the **Entity Sentiment Heatmap**:
   - Entities with negative sentiment display distinct red shades.
   - Entities with neutral/positive sentiment display gray/green shades.

---

### Test Scenario 5: Real-Time Risk Alerts & Threshold Verification
**Objective**: Verify the alert triggering logic.

1. Inspect the **Risk Alerts** panel in the bottom right corner.
2. Verify that alerts appear only when:
   - Signal Impact Score is **≥ 7**, OR
   - Signal Sentiment Score is **≤ −0.40**.
3. Each alert card displays:
   - Severity level (`HIGH` or `MED`).
   - Ticker badge and event type tag.
   - Summary headline and time elapsed.
   - Corroboration badge if multiple sources reported the event.

---

## 4. Verification Checklist for Evaluators

- [ ] **Real-Time WebSockets**: Top Bar shows `WS LIVE` with green status dot.
- [ ] **No Mock Data**: When the backend is stopped, all cards cleanly show empty states rather than hardcoded numbers.
- [ ] **Offline Resilience**: Historical Replay operates fully offline from local crisis episode logs without external API keys.
- [ ] **Interactive Ingestion**: Any custom headline can be analyzed live through the UI within <100ms.
- [ ] **Explainability**: Stress test shock results clearly cite the exact headline/tweet text evidence and show asset-class breakdowns.
- [ ] **Two Integrated Modules**:
  - **Module A**: Dynamic index rebalancing visible in the Index Weights chart.
  - **Module B**: Portfolio stress testing visible in the Stress Test panel.
