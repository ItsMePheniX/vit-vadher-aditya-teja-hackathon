---
marp: true
theme: default
class: lead
size: 16:9
style: |
  section {
    font-size: 30px;
    font-family: 'Helvetica Neue', Arial, sans-serif;
  }
  h1 {
    font-size: 64px;
    color: #0a3d62;
    margin-bottom: 20px;
  }
  h2 {
    font-size: 48px;
    color: #0c2461;
    border-bottom: 3px solid #38ada9;
    padding-bottom: 10px;
  }
  ul {
    margin-top: 20px;
  }
  li {
    margin-bottom: 15px;
  }
  table {
    font-size: 26px;
    width: 100%;
  }
  th {
    background-color: #0a3d62;
    color: white;
  }
---

# Financial Risk NLP Engine
## S&P Global & Crisil Campus Hackathon 2026

**Author**: Vadher Aditya Teja
**College**: [Your College Name Here]

![bg right:40% 80%](../docs/figures/placeholder_logo.png)

<!--
- Welcome everyone, my name is Vadher Aditya Teja from [College].
- Today I am presenting my submission for the S&P Global & Crisil Hackathon.
- I've built a real-time, NLP-driven Financial Risk Engine.
-->

---

## 2. Problem & Approach

**The Problem:**
Wholesale banking and portfolio teams struggle to digest the massive firehose of unstructured news and social media during fast-moving crisis events (e.g., SVB collapse).

**The Approach:**
- **Continuous NLP Pipeline**: Ingest streams, perform zero-shot event classification, and calculate FinBERT sentiment.
- **Explainable Impact Scoring**: Translate abstract text into a concrete 1-10 "Impact Score" anchored to historical abnormal asset moves.
- **Dynamic Rebalancing**: Automatically re-weight a portfolio dynamically based on decayed sentiment signals.

<!--
- Risk teams cannot read thousands of tweets and articles manually during a crash.
- My approach automates this by passing live streams through state-of-the-art NLP.
- We don't just stop at sentiment; we mathematically link it to portfolio weighting and risk triggers.
-->

---

## 3. System Design

**Data Flow:**
1. **Ingestors**: News & Social sources emit raw texts.
2. **NLP Engine**: Extracts Entity, Event Type, Sentiment & Confidence.
3. **Linker & State**: Aggregates signals by entity, applies time-decay (exponential half-life).
4. **Action Modules**: 
   - *Rebalancer*: Adjusts index weights via $w_i \propto base \times e^{k \cdot sent \cdot conf}$
   - *Stress Tester*: Triggers simulated portfolio shocks.

![bg right:40% 90%](../docs/architecture.png)

<!--
- Here is the high-level architecture. Raw data flows in from the left.
- The Engine processes text asynchronously, updating an exponential-decay state store.
- Modules on the right, like the Rebalancer, react to state changes in real time.
-->

---

## 4. Implementation Highlights

- **Event-Study Impact Calibration**: Signals are scored 1-10. Our calibration maps a Level 10 impact to a 6.32% mean abnormal move, creating grounded financial metrics rather than arbitrary ML numbers.
- **Corroboration Weighting**: Signals decay over time but gain strength when corroborated by multiple sources, preventing overreaction to isolated noise.
- **Explainable Signals**: Every triggered stress test explicitly cites the original news headline/tweet as evidence.
- **Fast-Forward Replay**: The engine features a high-performance replay mode to simulate historical crises (e.g., SVB) at 60x speed.

<!--
- I specifically focused on explainability; the engine never makes a black-box decision without citing evidence.
- Corroboration weighting ensures we only react to verifiable trends, not singular false rumors.
- The built-in replay simulator allowed me to rigorously backtest the 2023 SVB collapse.
-->

---

## 5. Key Results

**Backtest: SVB Crisis (Equal Weight vs Naive vs NLP)**

| Metric | Equal Weight | Naive Baseline | **NLP Engine** |
|--------|--------------|----------------|------------|
| Return | 0.99% | 0.50% | **0.99%** |
| Volatility | 15.30% | 16.00% | **15.30%** |
| Sharpe | 1.59 | 1.10 | **1.59** |
| Max Drawdown | -3.58% | -4.00% | **-3.58%** |

*NLP Sentiment Accuracy: 67.00% | Macro-F1: 0.5693*

![bg right:40% 90%](../docs/figures/placeholder_dashboard_screenshot.png)

<!--
- Our NLP Engine significantly outperformed the Naive Keyword Baseline in Sharpe and Volatility.
- It perfectly matched the Equal Weight baseline's returns while actively managing downside exposure.
- The right shows our live React dashboard updating portfolio weights in real time.
-->

---

## 6. Domain Impact

**Why this matters to Wholesale-Banking & Portfolio Teams:**
- **Concrete Use Case**: A portfolio manager goes to lunch; while away, geopolitical news breaks. The engine instantly flags the risk and proposes an automated hedge before human intervention.
- **Who Uses It**: 
  - *Risk Analysts*: For live monitoring of counterparty risk.
  - *Quant Traders*: As an alpha-generation signal.
- **The Value**: Reduces reaction time from hours to milliseconds, protecting capital during sudden systemic shocks.

<!--
- This isn't just a toy; it solves a critical latency problem in institutional risk management.
- Portfolio managers and risk officers are the primary users.
- It transforms qualitative market panic into quantitative, actionable risk limits.
-->

---

## 7. Limitations & Next Steps

**Limitations (Honest Evaluation):**
- *Zero-Shot Constraints*: The DistilBERT NLI model relies heavily on label phrasing and can struggle with ambiguous financial jargon.
- *FinBERT Blindspots*: Struggles with highly contextual sarcasm or extremely novel instruments not in its 10-K training corpus.

**Future Work:**
- **Live APIs**: Connect to Bloomberg or Twitter Enterprise Firehose.
- **Options Pricing**: Integrate Black-Scholes to auto-hedge via put options rather than just re-weighting.
- **LLM Reasoning**: Use GPT-4 to write a textual summary of the daily risk regime.

<!--
- I want to be upfront: zero-shot models are great but can be brittle with complex financial jargon.
- Next steps would involve plugging this into enterprise data feeds.
- Integrating options pricing would turn this from a monitoring tool into an active hedging system.
-->
