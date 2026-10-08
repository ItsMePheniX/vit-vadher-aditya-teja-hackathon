# Design Assumptions

> **Author**: Vadher Aditya Teja  
> **Last updated**: 2026-10-02  
> This document records every design decision and assumption that the jury may question.

---

## 1. NLP Model Choice — FinBERT

**Decision**: Use `ProsusAI/finbert` (pre-trained, no fine-tuning).

**Rationale**:
- FinBERT is trained specifically on financial communication (10-K filings, earnings calls, Bloomberg), giving domain-adapted embeddings vs. general BERT.
- Pre-trained model avoids the need for labelled training data which we cannot commit (proprietary constraint).
- Apache-2.0 licence is compatible with hackathon open-source submission.

**Limitation**: No fine-tuning on event-type labels. Classification is rule-based (see §3). A fine-tuned multi-label classifier would improve precision — noted as future work.

---

## 2. Event Type Classification — Zero-Shot NLI

**Decision**: Zero-shot natural language inference using `typeform/distilbert-base-uncased-mnli`.

**Rationale**:
- Pre-trained on MultiNLI for entailment, allowing us to classify text against dynamic label descriptions without labeled training data.
- Handles nuanced contexts better than keyword regex.
- Avoids domain-specific fine-tuning limitations.

**Limitation**: Computationally heavier than regex, though manageable on CPU with a DistilBERT backbone. Can fall back to a "catch-all" if inference fails.

---

## 3. Impact Score Formula — Gradient Boosting Regressor

**Decision**: An ML event-study calibration predicting absolute abnormal returns, then bucketed into 1–10 impact scores.
(Rule-based fallback used if the model is missing: `impact = clamp(⌈|sentiment| × confidence × event_weight × 10⌉, 1, 10)`)

**Rationale**:
- Maps sentiment and velocity to actual historical market price movements (abnormal returns vs SPY).
- Gradient Boosting Regressor effectively handles nonlinear relationships between sentiment magnitude, source count, and velocity.
- Mapping predicted moves to quantiles provides interpretable 1–10 scores based on real market data.

**Assumption**: The historical data used for calibration (e.g. SVB window) generalizes sufficiently to current market environments. The fallback rules are approximate and based on published academic research on event-driven market volatility.

---

## 4. Half-Life Values

| Event Type | Half-Life (hours) | Justification |
|------------|-------------------|---------------|
| Geopolitical | 72 | Sustained uncertainty; market adjusts slowly |
| Macroeconomic | 48 | Macro data releases have multi-day effects |
| Credit Event | 24 | Rapid credit repricing but then stabilizes |
| Merger/Acquisition | 168 | Deal uncertainties last weeks |
| Earnings | 12 | Price discovery typically within one session |
| Regulatory | 96 | Investigations/fines play out over days |
| Product Launch | 36 | Consumer/B2B sentiment peaks then fades |
| Other | 24 | Conservative default |

Values inspired by: Tetlock (2007) — "Giving Content to Investor Sentiment: The Role of Media in the Stock Market", Journal of Finance.

---

## 5. No Kafka — asyncio Queue

**Decision**: Use Python `asyncio.Queue` instead of Kafka.

**Rationale**: The problem spec explicitly states "No Kafka". An in-process queue satisfies the streaming requirement for a single-instance hackathon demo. For production, the same interface could be backed by Kafka with a one-line change to the producer/consumer functions.

---

## 6. Portfolio VaR — Equal-Weight Assumption

**Decision**: Assume equal-weight portfolio unless user provides weights.

**Rationale**: Default tickers (AAPL, MSFT, JPM, GS, XOM) represent diverse sectors. Equal-weight is a common benchmark assumption. VaR uses 252-day historical window (1 trading year) at 95% confidence.

**Limitation**: Does not account for correlations between assets in CVaR (uses simple average of individual VaRs). Full covariance matrix would be more accurate.

---

## 7. Signal ID — ULID

**Decision**: Use ULID (Universally Unique Lexicographically Sortable Identifier).

**Rationale**: ULIDs are time-sortable (unlike UUIDs), making JSONL queries and time-series operations efficient. The `python-ulid` library is Apache-2.0.

---

## 8. Data Persistence — Append-Only JSONL

**Decision**: Persist signals to a single JSONL file.

**Rationale**: Simple, auditable, human-readable. Append-only ensures no data loss. For production, replace with TimescaleDB or ClickHouse. JSONL also allows `tail -f` monitoring.

---

## 9. Synthetic Data

**Decision**: All committed data is synthetic (hand-crafted headlines).

**Rationale**: Avoids proprietary data licensing issues. Synthetic data still exercises all 8 event types and covers the full pipeline.

---

## 10. Dashboard Technology

**Decision**: React + Vite + Recharts.

**Rationale**: Lightweight, no SSR needed for a demo, Recharts integrates cleanly with real-time WebSocket data. Minimal bundle footprint.

---

## 11. Impact Score Calibration — Event-Study Method

### Method

The impact score [1–10] is now derived from a **gradient boosting regressor** trained
to predict |abnormal return| on the day after the event.

**Abnormal Return (AR)**:

```
AR(t+1) = R_stock(t+1) − R_SPY(t+1)
```

where R = log(P_t / P_{t-1}).  SPY is the market benchmark (beta = 1 approximation).
A full market-model regression (estimating alpha, beta over a 120-day estimation window)
would be more precise; this is left as future work.

**Targets**:
- `abs_ar1` — |AR| at t+1 (default)
- `abs_ar3` — |CAR[1,3]| cumulative over t+1 … t+3

**Features (all pre-event — no leakage)**:

| Feature | Description |
|---------|-------------|
| `sentiment_abs` | |FinBERT score| — magnitude proxy |
| `sentiment` | Signed FinBERT score |
| `sentiment_sq` | sentiment², captures non-linearity |
| `is_positive` | 1 if sentiment > 0.05 |
| `source_count` | Distinct news sources on event day |
| `velocity` | Signals/day in 24h window before event |
| `log_velocity` | log1p(velocity), reduces skew |
| `event_*` | One-hot encoding of 8 event types |

### Leakage Controls

> **Critical**: the train/test split is **chronological** (events before
> `SPLIT_DATE` → train; after → test).  Random splits would leak future market
> structure into the training set.

Enforced controls:
1. `leakage_check(df)` scans the DataFrame for any column whose name starts
   with `ar`, `ret`, `price`, `future`, `next_day` and raises a `UserWarning`.
2. `build_feature_matrix()` only ever returns columns listed in the frozen
   `FEATURE_COLUMNS` list; any extra column is dropped with a warning.
3. Prices are fetched only *after* the event date for labels; no price at t
   or earlier is used as a feature.
4. The quantile bucket mapping is fit **only on training-set predictions**,
   never on test predictions.

### Impact ↔ Abnormal Return Mapping

Impact level = quantile decile of predicted |AR| over the training set:

| Impact | Interpretation |
|--------|---------------|
| 1–2 | < 0.3% abnormal move (noise-level) |
| 3–4 | 0.3–0.8% abnormal move |
| 5–6 | 0.8–1.8% abnormal move |
| 7–8 | 1.8–3.5% abnormal move |
| 9–10 | > 3.5% abnormal move |

*(Exact thresholds are data-driven; run `make train` to see them.)*

### Fallback

When `data/impact_model/gb_model.joblib` is missing, `scorer.py` uses
the rule-based fallback:

```
raw   = |sentiment| × event_weight × 10
impact = clamp(⌈raw⌉, 1, 10)
```

where `event_weight` ∈ {0.90, 1.00, 1.10, 1.15, 1.20, 1.35, 1.40}.

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean absolute error on |AR| (in fraction, e.g. 0.012 = 1.2%) |
| R² | Coefficient of determination on held-out test set |
| Spearman ρ | Rank correlation between predicted and actual |AR| |
| Bucket accuracy | Fraction of test events where predicted impact = label impact |

### References

- Fama et al. (1969) — "Adjustment of Stock Prices to New Information"
- Mitchell & Mulherin (1994) — "The Impact of Public Information on the Stock Market"
- MacKinlay (1997) — "Event Studies in Economics and Finance", *JEL*
