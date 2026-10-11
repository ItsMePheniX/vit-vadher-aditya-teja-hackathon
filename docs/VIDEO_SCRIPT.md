# Demo Video Script

**Presenter**: Aditya Teja  
**Project**: Financial Risk NLP Engine  
**Duration**: ~5 to 7 minutes  

---

### 1. Introduction & The Problem (0:00 – 0:45)
**[On screen: Webcam / Title Slide]**

"Hi everyone, I'm Aditya. Today I'm presenting the **Financial Risk NLP Engine**.

Think about what happened in March 2023 when Silicon Valley Bank collapsed. Within just a few hours, thousands of tweets and breaking news articles flooded the market. Traders were panicking, depositors were pulling funds, and regional bank stocks were crashing.

The reality in wholesale banking is that human risk teams cannot read and analyze thousands of text posts every minute. By the time someone notices the trend and updates a risk spreadsheet, the portfolio has already taken massive losses.

I built this engine to solve that problem. It automatically monitors unstructured news and social feeds, scores the risk in milliseconds using FinBERT and zero-shot NLP, and immediately takes action: stress-testing the portfolio and dynamically rebalancing index weights to protect capital."

---

### 2. Quick Setup & Run (0:45 – 1:15)
**[On screen: Terminal window showing project root]**

"Let's see how easy it is to run. 

You can launch the entire stack in one command using Docker:
```bash
docker compose up --build
```
Or natively using bash:
```bash
bash scripts/demo.sh
```

Both our FastAPI backend and React dashboard start up together. Everything runs locally, and all models are pre-cached, so the entire demo works completely offline without needing third-party API keys or external dependencies."

---

### 3. Live Dashboard & Interactive Headline Test (1:15 – 3:00)
**[On screen: Browser showing dashboard at http://localhost:5173]**

"Here is our live dashboard. 

Across the top, you can see our real-time metrics: total signals processed, the market sentiment split, high-impact alerts, average model confidence, and our portfolio Value at Risk.

Every number you see here is fed directly from the backend over live WebSockets—there is no fake or hardcoded data.

Let's test the NLP engine directly. In the bottom feed, there's an input box where we can paste any headline. Let's test a real breaking crisis headline:

> *'SVB collapses as depositors withdraw $42B in a single day amid liquidity crisis'*

When I click **Analyse**, watch what happens:
In less than 80 milliseconds, the engine processes the text:
- It identifies the entity as **Silicon Valley Bank**.
- It categorizes the event as a **Credit Event**.
- FinBERT assigns a deeply negative sentiment score of **-0.95**.
- And our impact engine flags this at maximum severity: **Impact 10 out of 10**.

Now let's compare that with regular corporate earnings:

> *'Apple reports record quarterly revenue beating consensus by 15%'*

Instantly: entity is **AAPL**, event type is **Earnings**, sentiment is a solid **+0.85** in green, and impact drops down to a modest **3**. 

The engine clearly knows the difference between normal corporate earnings and a systemic banking panic."

---

### 4. Module B — Automated Stress Testing (3:00 – 4:15)
**[On screen: Zoom in on the Stress Test card on the right]**

"This brings us to **Module B: The Automated Stress Tester**.

Usually, running a stress test requires a risk analyst to manually define shock scenarios hours after an event happens. Our engine does this continuously in the background.

Whenever a signal hits high severity—impact of 7 or above with strong confidence—it automatically triggers a live portfolio shock.

Looking at the Stress Test card:
The SVB event automatically triggered a shock on our $500 million portfolio:
- **Before Value**: $499.5 Million.
- **After Value**: $484.5 Million.
- **Estimated Drawdown**: An instant **$15.0 Million loss (3.0%)**.

Below that, you can see the breakdown across asset classes:
- **Bonds** lost **$9.5M**, reflecting duration risk and unrealized bond portfolio losses.
- **Loans** lost **$5.6M** from credit spread widening.
- **Derivatives** gained **$195K**, acting as a partial hedge.

And the best part is **explainability**: the card shows the exact headline quote that triggered the shock, along with the specific equity, rate, and credit shock parameters applied. It's completely transparent, not a black box."

---

### 5. Module A — Dynamic Portfolio Rebalancer (4:15 – 5:15)
**[On screen: Pan to Index Weights chart on bottom-left]**

"Now let's look at **Module A: Dynamic Rebalancing**.

Measuring risk is good, but institutional systems need to take action.

In the bottom left, the Index Weights chart tracks portfolio allocations across our asset universe. As negative sentiment accumulates for distressed banks, the engine automatically trims exposure to those tickers and shifts capital toward safer assets.

We built two key financial safeguards into the math:
1. **Exponential time decay**: News gets old quickly. A tweet from three days ago shouldn't dictate today's trades, so signals decay smoothly over time using a half-life model.
2. **Turnover limits**: In real trading, you can't dump half your portfolio at once without paying massive transaction costs. We cap turnover at 5% per rebalance to keep trading realistic and profitable."

---

### 6. SVB Crisis Replay & Results (5:15 – 6:15)
**[On screen: Engine Mode toggle in dashboard, set to Replay at 300x]**

"To prove this works on real market data, we built a crisis replay engine. 

Under Engine Mode on the top right, I can select the **SVB March 2023** episode and speed it up to **300x**. 

Watch the simulated clock: it replays actual news and social posts from March 8th to 17th, 2023. As panic spreads, you see signals streaming in, sentiment turning heavily bearish, and alerts firing in real time.

We backtested this strategy against both an Equal-Weight portfolio and a naive keyword baseline:
- Our NLP engine achieved a **Sharpe ratio of 1.59**, compared to **1.10** for the baseline—that's a **44% improvement**.
- It held volatility down to **15.3%**.
- And it reduced maximum drawdown from **-4.0%** to **-3.58%**, proving that automated NLP risk protection genuinely preserves capital."

---

### 7. Conclusion (6:15 – 6:45)
**[On screen: Presenter webcam / closing slide]**

"To wrap up: this engine bridges the gap between raw text streams and quantitative institutional risk management. It turns unstructured market panic into concrete, explainable risk limits in milliseconds.

All datasets used here are fully public or synthetic, and the code is 100% reproducible out of the box.

Thank you so much for your time, and I look forward to your questions."
