# FinRisk NLP Engine - Demo Video Script

## Overview
- **Target Length**: 10 minutes (Full Walkthrough)
- **Time-Limited Length**: 5 minutes (Tight Cut highlighted with `[5-MIN CUT]`)
- **Theme**: Showcasing institutional risk management shifting from reactive humans to proactive, explainable ML.

---

## Part 1: Introduction (0:00 - 0:30) `[5-MIN CUT: 0:00 - 0:20]`
**[On-Screen]**: Title Slide: *Financial Risk NLP Engine*. Speaker camera on.
**Voiceover**: "Hi, I'm Aditya. Today I'm presenting a real-time NLP Risk Engine. When crises like the SVB collapse happen, risk teams cannot manually read the thousands of news articles and tweets flooding in per second. My system automatically ingests unstructured text, scores financial impact using zero-shot NLP, and dynamically stress-tests and rebalances portfolios in milliseconds."

## Part 2: Setup & Run (0:30 - 1:00) `[5-MIN CUT: 0:20 - 0:35]`
**[On-Screen]**: Screen recording of the terminal showing `README.md` quickstart commands.
**Voiceover**: "Let's run it from scratch. Using standard Python and Node, we set up the environment in seconds. We pre-warm the HuggingFace cache to ensure it runs completely offline, avoiding network unreliability during the pitch. With a single command `bash scripts/demo.sh`, both our backend FastAPI and React dashboard boot up."

## Part 3: End-to-End Walkthrough (1:00 - 4:00) `[5-MIN CUT: 0:35 - 2:00]`
**[On-Screen]**: Terminal window showing log streams of incoming tweets and news.
**Voiceover**: "Data flows in from two parallel sources: historical news and social media. Let's trace a single headline. Here, we see an article about an unexpected SVB asset sale."

**[On-Screen]**: Zoom in on terminal: `src.engine.nlp.classifier: NLI classifier ready`.
**Voiceover**: "The text hits our pipeline. First, FinBERT determines the sentiment. Then, a DistilBERT Zero-Shot classifier categorizes it as a 'Credit Event'. It doesn't stop there. Our rules engine assigns a 1-to-10 Impact Score based on the event type and sentiment confidence, corroborating it across multiple sources to ignore isolated noise."

**[On-Screen]**: Switch to the **Dashboard - Stress Test Panel**.
**Voiceover**: "This leads us to Module B: The Stress Tester. This module continuously watches the signal stream. When it detects a Level 7 or higher impact event—like the SVB run—it triggers a live portfolio shock. Notice how the dashboard immediately updates. We simulate a 3% shock on the Banking sector, recalculating the VaR instantly while explicitly citing the exact tweet that caused the shock for total explainability."

## Part 4: Module A & Live-vs-Replay Proof (4:00 - 8:00) `[5-MIN CUT: 2:00 - 4:00]`
**[On-Screen]**: Switch to the **Dashboard - Rebalancer View**.
**Voiceover**: "Now let's look at Module A: Dynamic Rebalancing. We aren't just measuring risk; we are taking action. The engine maintains a time-decayed state for each entity. As sentiment drops, the system proportionally dials down our index weight for that specific ticker, capping turnover to reflect realistic transaction costs."

**[On-Screen]**: Terminal: `python -m src.run --mode live`
**Voiceover**: "Right now, we are running in Replay mode, accelerating the SVB crisis at 60x speed to prove the engine handles historical crashes. But the architecture is fully asynchronous. By flipping one flag to `--mode live`, the engine seamlessly detaches from the CSV files and connects to live WebSocket streams without dropping a single frame."

## Part 5: Results and Impact (8:00 - 10:00) `[5-MIN CUT: 4:00 - 5:00]`
**[On-Screen]**: Presentation Slide: *Key Results (Backtest Table)*.
**Voiceover**: "We backtested this engine against an Equal-Weight portfolio and a Naive Baseline. During the SVB timeframe, our NLP engine preserved an annualized volatility of 15.3% while significantly outperforming the naive keyword-matching approach's drawdown metric. 

The domain impact is massive. Wholesale banking teams can deploy this to instantly hedge against geopolitical shocks while they are away from their desks, transforming qualitative market panic into quantitative, actionable risk limits. Thank you."
