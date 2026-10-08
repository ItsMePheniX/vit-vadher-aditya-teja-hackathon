"""event_study.py — price fetching + abnormal-return computation.

Abnormal Return (AR) Definition
--------------------------------
    AR(t) = R_stock(t) - R_spy(t)

where R = log(P_t / P_{t-1}).  SPY is the benchmark proxy for market return.

This follows the Market Model simplified to beta=1 (no regression), which is
appropriate for a short hackathon event window.  A full market-model regression
(estimating alpha and beta over a pre-event estimation window) would be more
precise — noted as future work in docs/ASSUMPTIONS.md.

Windows
-------
    AR1  = AR on the day after the event (t+1)
    AR3  = cumulative AR over t+1, t+2, t+3   (CAR[1,3])

Data Caching
------------
Prices are cached as CSVs in  data/prices_cache/<TICKER>_<START>_<END>.csv
so the full pipeline runs without network access after the first pull.

Leakage Guard
-------------
All returned columns are anchored to the EVENT DATE.  We never include any
price from *after* t+3.  The training script calls this function and then joins
features that are computed from text BEFORE the event.

    ⚠️  LEAKAGE WARNING: if you add any feature that requires knowing the price
    on the event day (P_t) to compute, you must log-transform it from P_{t-1}.
    The get_abnormal_returns() function only returns rows where BOTH the event
    date AND the return window exist in the price data — rows at the end of the
    price series are dropped to avoid look-ahead.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("data/prices_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_SPY = "SPY"
_MAX_WINDOW = 5      # days after event we need prices for
_EST_WINDOW = 0      # estimation window days BEFORE event (beta=1, no regression needed)


class EventReturn(NamedTuple):
    entity: str
    event_date: date
    event_type: str
    sentiment: float
    ar1: float        # abnormal return day+1
    ar3: float        # cumulative abnormal return days+1..+3
    abs_ar1: float    # |AR1| — regression target (1-day)
    abs_ar3: float    # |CAR3| — regression target (3-day)


# ── Price fetching (with local cache) ─────────────────────────────────────────

def _cache_path(ticker: str, start: str, end: str) -> Path:
    tag = f"{ticker}_{start}_{end}".replace("-", "")
    return _CACHE_DIR / f"{tag}.csv"


def _fetch(ticker: str, start: date, end: date) -> pd.Series:
    """Return daily log-returns for ticker between start and end (inclusive).

    Attempts local CSV cache first; falls back to yfinance.
    Returns an empty Series on any failure.
    """
    start_s = start.isoformat()
    end_s   = end.isoformat()
    cache   = _cache_path(ticker, start_s, end_s)

    if cache.exists():
        try:
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            series = df["log_ret"].dropna()
            logger.debug("Cache hit: %s", cache.name)
            return series
        except Exception as exc:
            logger.warning("Cache read failed (%s): %s", cache.name, exc)

    try:
        import yfinance as yf  # noqa: PLC0415
        # yfinance end date is exclusive, so add 1 day
        yf_end = (end + timedelta(days=1)).isoformat()
        raw = yf.download(
            ticker, start=start_s, end=yf_end,
            progress=False, auto_adjust=True,
        )
        if raw.empty:
            logger.warning("yfinance returned empty for %s %s→%s", ticker, start_s, end_s)
            return pd.Series(dtype=float)

        prices = raw["Close"].squeeze()
        log_ret = np.log(prices / prices.shift(1)).dropna()

        # Cache to CSV
        log_ret.to_frame("log_ret").to_csv(cache)
        logger.info("Cached prices for %s → %s", ticker, cache.name)
        return log_ret

    except Exception as exc:  # noqa: BLE001
        logger.error("yfinance failed for %s: %s", ticker, exc)
        return pd.Series(dtype=float)


# ── Abnormal return calculation ────────────────────────────────────────────────

def _compute_ar(
    stock_ret: pd.Series,
    spy_ret: pd.Series,
    event_date: date,
    window: int = 3,
) -> tuple[float | None, float | None]:
    """Compute AR1 and CAR3 for a single event.

    Leakage note: we use t+1 through t+window, which are all AFTER the event.
    These are the labels (y), not features (X).  Features come from before t.

    Returns (ar1, ar3) or (None, None) if insufficient data.
    """
    event_ts = pd.Timestamp(event_date)

    # Find positions in the index after the event date
    stock_dates = stock_ret.index[stock_ret.index > event_ts]
    spy_dates   = spy_ret.index[spy_ret.index > event_ts]

    if len(stock_dates) < window or len(spy_dates) < window:
        return None, None

    # t+1 abnormal return
    d1_stock = stock_ret.loc[stock_dates[:1]]
    d1_spy   = spy_ret.loc[spy_dates[:1]]
    if d1_stock.empty or d1_spy.empty:
        return None, None
    ar1 = float(d1_stock.iloc[0] - d1_spy.iloc[0])

    # CAR[1..3]: sum of AR over next `window` days
    stock_window = stock_ret.loc[stock_dates[:window]].values
    spy_window   = spy_ret.loc[spy_dates[:window]].values
    n = min(len(stock_window), len(spy_window))
    ar3 = float((stock_window[:n] - spy_window[:n]).sum())

    return ar1, ar3


# ── Public API ─────────────────────────────────────────────────────────────────

def get_abnormal_returns(events: list[dict]) -> pd.DataFrame:
    """Compute abnormal returns for a list of labeled events.

    Args:
        events: list of dicts with keys:
            entity      (str)   — ticker symbol
            event_date  (date)  — date of the news event
            event_type  (str)   — one of 8 event types
            sentiment   (float) — FinBERT score [-1,1]
            source_count (int)  — number of distinct sources that day
            velocity    (float) — signal count in 24h window before event

    Returns:
        DataFrame with columns: entity, event_date, event_type, sentiment,
        source_count, velocity, ar1, ar3, abs_ar1, abs_ar3.
        Rows where prices were unavailable are DROPPED and a warning logged.

    Leakage guarantee:
        - All returned features (sentiment, velocity, source_count) are
          from AT OR BEFORE the event_date.
        - ar1, ar3, abs_ar1, abs_ar3 are computed from prices AFTER the event.
        - No price on the event day itself is used as a feature.
    """
    if not events:
        return pd.DataFrame()

    # Compute global date range for price fetching
    all_dates = [e["event_date"] for e in events]
    fetch_start = min(all_dates) - timedelta(days=5)
    fetch_end   = max(all_dates) + timedelta(days=_MAX_WINDOW + 2)

    # Fetch SPY once for the full range
    spy_ret = _fetch(_SPY, fetch_start, fetch_end)
    if spy_ret.empty:
        logger.error("Cannot fetch SPY — no benchmark available. Generating synthetic returns.")
        # Synthetic SPY returns for offline operation
        rng = np.random.default_rng(42)
        date_range = pd.date_range(fetch_start, fetch_end, freq="B")
        spy_ret = pd.Series(rng.normal(0.0003, 0.008, len(date_range)), index=date_range)

    rows: list[dict] = []

    # Deduplicate per-entity fetches
    ticker_cache: dict[str, pd.Series] = {}
    for ev in events:
        t = ev["entity"]
        if t not in ticker_cache and t != "MARKET":
            ticker_cache[t] = _fetch(t, fetch_start, fetch_end)

    for ev in events:
        entity = ev["entity"]
        edate  = ev["event_date"]

        if entity == "MARKET":
            # Use SPY as proxy for market-wide events
            stock_ret = spy_ret
        else:
            stock_ret = ticker_cache.get(entity, pd.Series(dtype=float))
            if stock_ret.empty:
                logger.warning("No price data for %s on %s — skipping.", entity, edate)
                continue

        ar1, ar3 = _compute_ar(stock_ret, spy_ret, edate, window=3)
        if ar1 is None:
            logger.warning(
                "Insufficient post-event price data for %s on %s — skipping.", entity, edate,
            )
            continue

        rows.append({
            "entity":       entity,
            "event_date":   edate,
            "event_type":   ev["event_type"],
            "sentiment":    ev["sentiment"],
            "source_count": ev.get("source_count", 1),
            "velocity":     ev.get("velocity", 1.0),
            "ar1":          ar1,
            "ar3":          ar3,
            "abs_ar1":      abs(ar1),
            "abs_ar3":      abs(ar3),
        })

    df = pd.DataFrame(rows)
    logger.info(
        "Event study: %d events → %d with returns (%.0f%% coverage)",
        len(events), len(df), 100 * len(df) / max(len(events), 1),
    )
    return df


def build_events_from_csv(news_path: str | Path) -> list[dict]:
    """Read the news sample CSV and convert to event dicts.

    Applies the entity linker to extract tickers and the rule-based classifier
    to assign event types.  Sentiment is set to a placeholder 0.0 here —
    the training script will override it with real FinBERT scores if available.

    This function is intentionally offline-capable: it reads only the CSV,
    no models needed.

    Leakage note: source_count and velocity are computed from the same CSV
    over a ±0-day window — they reflect information available at event time.
    No future-date rows contribute to these counts.
    """
    df = pd.read_csv(news_path, parse_dates=["publish_date"])
    df = df.dropna(subset=["publish_date", "title"])
    df["event_date"] = df["publish_date"].dt.date

    from src.engine.linker import EntityLinker      # noqa: PLC0415
    from src.engine.classifier import classify      # noqa: PLC0415

    linker = EntityLinker()
    events: list[dict] = []

    # Compute per-day velocity (signals per day)
    daily_counts = df.groupby("event_date")["id"].count().to_dict()

    for _, row in df.iterrows():
        text = f"{row.get('title','')} {row.get('body','')}".strip()
        if not text:
            continue

        links = linker.link(text)
        entities = [lk.ticker for lk in links] if links else ["MARKET"]
        event_type = classify(text)

        # source_count: how many distinct sources covered this date
        date_df = df[df["event_date"] == row["event_date"]]
        source_count = date_df["source"].nunique() if "source" in date_df else 1

        velocity = float(daily_counts.get(row["event_date"], 1))

        for entity in entities:
            events.append({
                "entity":       entity,
                "event_date":   row["event_date"],
                "event_type":   event_type,
                "sentiment":    float(row.get("sentiment", 0.0)),
                "source_count": int(source_count),
                "velocity":     velocity,
            })

    logger.info("Built %d event rows from %s", len(events), news_path)
    return events
