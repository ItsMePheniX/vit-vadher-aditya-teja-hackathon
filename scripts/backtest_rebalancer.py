"""backtest_rebalancer.py — Backtest the NLP-driven rebalancer against equal-weight.

Usage:
    python scripts/backtest_rebalancer.py
"""
import sys
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure repo root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.ingest.news import NewsSource
from src.engine.ingest.social import SocialSource
from src.engine.nlp.pipeline import process
from src.engine.state import StateStore
from src.modules.rebalancer.engine import RebalancerEngine, UNIVERSE

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


async def load_items(news_path, tweets_path):
    items = []
    if news_path.exists():
        news_src = NewsSource(news_path)
        async for item in news_src.items():
            if item.ts.year >= 2000:
                items.append(item)
    
    if tweets_path.exists():
        social_src = SocialSource(tweets_path)
        async for item in social_src.items():
            if item.ts.year >= 2000:
                items.append(item)
    return items


async def main():
    print("Loading datasets...")
    data_dir = Path("data")
    prices_path = data_dir / "svb_prices_sample.csv"
    news_path = data_dir / "svb_news_sample.csv"
    tweets_path = data_dir / "svb_tweets_sample.csv"

    # 1. Load prices
    df_prices = pd.read_csv(prices_path, parse_dates=["date"])
    
    # Calculate daily returns close-to-close
    df_pivot = df_prices.pivot(index="date", columns="ticker", values="close")
    df_returns = df_pivot.pct_change().dropna(how="all")
    
    dates = sorted(df_returns.index)

    # 2. Gather items from sources
    items = await load_items(news_path, tweets_path)
            
    items.sort(key=lambda x: x.ts)
    print(f"Loaded {len(items)} items to process.")

    # 3. Simulate timeline day by day
    store = StateStore()
    engine = RebalancerEngine()
    
    portfolio_eq_val = 100.0
    portfolio_nlp_val = 100.0
    
    eq_returns = []
    nlp_returns = []
    
    # Initial weights
    prev_weights = engine.equal_weights()
    current_nlp_weights = prev_weights.copy()
    
    item_idx = 0
    total_items = len(items)
    
    # Transaction cost in basis points
    bps_cost = 10.0
    tc_rate = bps_cost / 10000.0

    print("Running backtest...")
    
    for i, date in enumerate(dates):
        # We process signals up to 16:00 (market close) of the current day.
        # Then we rebalance, and the new weights apply to the *next* day's return.
        # But wait, df_returns.loc[date] is the return from date-1 to date.
        # So at the start of 'date', we experience the return from 'date-1' to 'date'.
        
        # 1. Apply today's returns to our portfolios
        day_ret = df_returns.loc[date].fillna(0.0)
        
        # Equal weight return
        eq_ret = sum(day_ret.get(t, 0.0) * (1.0 / len(UNIVERSE)) for t in UNIVERSE)
        portfolio_eq_val *= (1 + eq_ret)
        eq_returns.append(eq_ret)
        
        # NLP return
        nlp_ret = sum(day_ret.get(t, 0.0) * current_nlp_weights.get(t, 1.0/len(UNIVERSE)) for t in UNIVERSE)
        portfolio_nlp_val *= (1 + nlp_ret)
        nlp_returns.append(nlp_ret)
        
        # 2. Process signals up to EOD today
        # EOD is 23:59:59 UTC of the 'date'
        eod_ts = pd.Timestamp(date).tz_localize("UTC") + pd.Timedelta(hours=23, minutes=59, seconds=59)
        
        while item_idx < total_items and items[item_idx].ts <= eod_ts:
            item = items[item_idx]
            signals = process(item)
            for sig in signals:
                if sig.entity in UNIVERSE:
                    store.ingest(sig)
            item_idx += 1
            
        store.prune()
        
        # 3. Compute new weights for tomorrow
        snap = store.snapshot()
        sentiment_map = {ent: float(entry.sentiment) for ent, entry in snap.items() if ent in UNIVERSE}
        
        new_weights, turnover = engine.compute(sentiment_map, current_nlp_weights)
        
        # 4. Apply transaction cost (paid today, deduct from portfolio value)
        tc = turnover * tc_rate
        portfolio_nlp_val *= (1 - tc)
        
        current_nlp_weights = new_weights
        
    # Calculate metrics
    eq_series = pd.Series(eq_returns, index=dates)
    nlp_series = pd.Series(nlp_returns, index=dates)
    
    def calc_metrics(series: pd.Series):
        ann_ret = (1 + series.mean()) ** 252 - 1
        ann_vol = series.std() * math.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum_ret = (1 + series).cumprod()
        drawdown = (cum_ret / cum_ret.cummax()) - 1
        max_dd = drawdown.min()
        return ann_ret, ann_vol, sharpe, max_dd

    eq_ann_ret, eq_vol, eq_sharpe, eq_mdd = calc_metrics(eq_series)
    nlp_ann_ret, nlp_vol, nlp_sharpe, nlp_mdd = calc_metrics(nlp_series)
    
    print("\n" + "="*50)
    print("BACKTEST RESULTS (vs Equal Weight)")
    print("="*50)
    print(f"{'Metric':<20} | {'Eq Weight':<12} | {'NLP Driven':<12}")
    print("-" * 50)
    print(f"{'Total Return':<20} | {portfolio_eq_val - 100:>11.2f}% | {portfolio_nlp_val - 100:>11.2f}%")
    print(f"{'Ann. Return':<20} | {eq_ann_ret*100:>11.2f}% | {nlp_ann_ret*100:>11.2f}%")
    print(f"{'Ann. Volatility':<20} | {eq_vol*100:>11.2f}% | {nlp_vol*100:>11.2f}%")
    print(f"{'Sharpe Ratio':<20} | {eq_sharpe:>11.2f}  | {nlp_sharpe:>11.2f} ")
    print(f"{'Max Drawdown':<20} | {eq_mdd*100:>11.2f}% | {nlp_mdd*100:>11.2f}%")
    print("="*50)

    # 5. Plot the equity curves
    plt.figure(figsize=(10, 6))
    plt.plot(dates, (1 + eq_series).cumprod() * 100, label="Equal Weight", color="gray", linestyle="--")
    plt.plot(dates, (1 + nlp_series).cumprod() * 100, label="NLP Driven", color="blue", linewidth=2)
    plt.title("Backtest: NLP Rebalancer vs Equal Weight (SVB Window)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(alpha=0.3)
    
    chart_path = data_dir / "backtest_results.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    print(f"Saved chart to {chart_path}")

if __name__ == "__main__":
    asyncio.run(main())
