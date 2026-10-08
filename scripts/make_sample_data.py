"""make_sample_data.py — generate or sample small data files for the demo.

What this does
--------------
1.  If full Kaggle datasets exist (downloaded via download_data.py), samples
    them to the SVB March 2023 window (2023-03-08 to 2023-03-17) and writes:
        data/svb_news_sample.csv
        data/svb_tweets_sample.csv

2.  Always (re)generates:
        data/svb_prices_sample.csv   via yfinance for the real OHLCV data

3.  If full Kaggle datasets are absent, the script keeps the existing
    synthetic sample files unchanged and prints a notice.

Usage:
    python scripts/make_sample_data.py [--force-synthetic]

The SVB window was chosen because it contains clear, dated financial events
across all five target banking tickers (JPM, BAC, C, GS, WFC).
See docs/ASSUMPTIONS.md for reasoning.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from datetime import date
from pathlib import Path

_REPO_ROOT  = Path(__file__).parent.parent
_DATA_DIR   = _REPO_ROOT / "data"
_SVB_START  = date(2023, 3, 8)
_SVB_END    = date(2023, 3, 17)

_FULL_NEWS_PATH   = _DATA_DIR / "all_the_news.csv"       # Kaggle "All the News"
_FULL_TWEET_PATH  = _DATA_DIR / "stock_tweets.csv"       # Kaggle stock tweets

_BANKING_KEYWORDS = {
    "svb", "silicon valley bank", "jpmorgan", "jp morgan", "jpm",
    "bank of america", "bofa", "bac", "citigroup", "citi", "citibank",
    "goldman sachs", "goldman", "wells fargo", "wfc", "signature bank",
    "fdic", "federal reserve", "bank run", "deposit", "banking crisis",
    "regional bank", "btfp", "bank term funding",
}

TICKERS = ["JPM", "BAC", "C", "GS", "WFC"]


# ── Prices via yfinance ────────────────────────────────────────────────────────

def generate_prices() -> None:
    out = _DATA_DIR / "svb_prices_sample.csv"
    print(f"Fetching real OHLCV prices from yfinance → {out}")
    try:
        import pandas as pd   # noqa: PLC0415
        import yfinance as yf  # noqa: PLC0415
    except ImportError:
        print("  yfinance / pandas not installed — skipping price generation.")
        return

    frames = []
    for ticker in TICKERS:
        raw = yf.download(
            ticker,
            start="2023-03-01",
            end="2023-03-18",
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            print(f"  WARNING: no data for {ticker}")
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        raw["ticker"] = ticker
        raw.index.name = "date"
        raw = raw.reset_index()[["date", "ticker", "Open", "High", "Low", "Close", "Volume"]]
        raw.columns = ["date", "ticker", "open", "high", "low", "close", "volume"]
        frames.append(raw)

    if not frames:
        print("  No price data retrieved — keeping existing file.")
        return

    df = pd.concat(frames).sort_values(["date", "ticker"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_csv(out, index=False)
    print(f"  Saved {len(df)} rows.")


# ── News sampling ──────────────────────────────────────────────────────────────

def _is_banking_row(title: str, body: str) -> bool:
    combined = (title + " " + body).lower()
    return any(kw in combined for kw in _BANKING_KEYWORDS)


def sample_news(force_synthetic: bool = False) -> None:
    out = _DATA_DIR / "svb_news_sample.csv"

    if force_synthetic or not _FULL_NEWS_PATH.exists():
        print(f"Full news dataset not found at {_FULL_NEWS_PATH}.")
        print("  Keeping existing synthetic sample. Run `make download` first.")
        return

    print(f"Sampling news from {_FULL_NEWS_PATH} → {out}")
    sampled: list[dict] = []

    with open(_FULL_NEWS_PATH, encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            # Detect date column
            date_val = (
                row.get("date") or row.get("publish_date") or row.get("published_at") or ""
            ).strip()[:10]
            if not date_val:
                continue
            try:
                row_date = date.fromisoformat(date_val)
            except ValueError:
                continue

            if not (_SVB_START <= row_date <= _SVB_END):
                continue

            title = row.get("title") or row.get("headline") or ""
            body  = row.get("content") or row.get("body") or row.get("text") or ""
            if not _is_banking_row(title, body):
                continue

            sampled.append({
                "id": i,
                "publish_date": date_val,
                "title": title[:500],
                "body": (body[:1000] if body else ""),
                "source": row.get("publication") or row.get("source") or "unknown",
                "url": row.get("url") or "",
            })

    if not sampled:
        print("  No matching rows found — keeping existing file.")
        return

    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "publish_date", "title", "body", "source", "url"])
        writer.writeheader()
        writer.writerows(sampled[:200])
    print(f"  Sampled {min(len(sampled), 200)} rows.")


# ── Tweet sampling ─────────────────────────────────────────────────────────────

def sample_tweets(force_synthetic: bool = False) -> None:
    out = _DATA_DIR / "svb_tweets_sample.csv"

    if force_synthetic or not _FULL_TWEET_PATH.exists():
        print(f"Full tweet dataset not found at {_FULL_TWEET_PATH}.")
        print("  Keeping existing synthetic sample. Run `make download` first.")
        return

    print(f"Sampling tweets from {_FULL_TWEET_PATH} → {out}")
    sampled: list[dict] = []

    with open(_FULL_TWEET_PATH, encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            date_val = (
                row.get("post_date") or row.get("created_at") or row.get("date") or ""
            ).strip()[:10]
            if not date_val:
                continue
            try:
                row_date = date.fromisoformat(date_val)
            except ValueError:
                continue

            if not (_SVB_START <= row_date <= _SVB_END):
                continue

            text = row.get("body") or row.get("tweet") or row.get("text") or ""
            ticker = row.get("ticker") or row.get("ticker_symbol") or ""

            # Only keep if mentions one of our tracked tickers or banking keywords
            if ticker.upper() not in TICKERS and not _is_banking_row(text, ""):
                continue

            sampled.append({
                "tweet_id": i,
                "writer": row.get("writer") or row.get("user") or "",
                "post_date": date_val,
                "body": text[:500],
                "ticker": ticker,
            })

    if not sampled:
        print("  No matching rows — keeping existing file.")
        return

    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["tweet_id", "writer", "post_date", "body", "ticker"])
        writer.writeheader()
        writer.writerows(sampled[:500])
    print(f"  Sampled {min(len(sampled), 500)} rows.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample data files for the demo")
    parser.add_argument(
        "--force-synthetic", action="store_true",
        help="Use synthetic files even if full datasets are present",
    )
    parser.add_argument(
        "--skip-prices", action="store_true",
        help="Skip yfinance price fetch (useful in CI without network)",
    )
    args = parser.parse_args()

    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    sample_news(force_synthetic=args.force_synthetic)
    sample_tweets(force_synthetic=args.force_synthetic)
    if not args.skip_prices:
        generate_prices()

    print("\nDone. Files in data/:")
    for f in sorted(_DATA_DIR.glob("svb_*.csv")):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
