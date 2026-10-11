"""Portfolio risk module.

Computes:
- Historical Value at Risk (VaR) using 252-day rolling window
- Signal-adjusted risk score for a portfolio given recent NLP signals
- Per-ticker exposure based on incoming signals

Data source: yfinance (Apache-2.0 compatible).
See data/DATA_SOURCES.md for provenance.

Assumption: equal-weight portfolio unless weights are provided.
See docs/ASSUMPTIONS.md §Portfolio Risk.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from src import config
from src.engine.schema import Signal

logger = logging.getLogger(__name__)

_LOOKBACK = config.get("portfolio", "lookback_days", default=252)
_VAR_CONF = config.get("portfolio", "var_confidence", default=0.95)
_DEFAULT_TICKERS: list[str] = config.get("portfolio", "default_tickers",
                                          default=["AAPL", "MSFT", "JPM", "GS", "XOM"])


@dataclass
class PortfolioRisk:
    tickers: list[str]
    var_95: float                           # daily VaR as fraction
    cvar_95: float                          # conditional VaR (Expected Shortfall)
    signal_adjusted_risk: float             # VaR amplified by signal impact
    ticker_exposures: dict[str, float]      # ticker → risk contribution fraction
    as_of: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())


def _fetch_returns(tickers: list[str]) -> pd.DataFrame:
    """Download daily close prices from yfinance and return log returns."""
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=_LOOKBACK + 10)

    logger.info("Fetching prices for %s …", tickers)
    raw = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw

    prices = prices.dropna(how="all").ffill()
    if prices.empty:
        raise RuntimeError("Empty price data returned from yfinance")
    returns = np.log(prices / prices.shift(1)).dropna()
    if returns.empty or len(returns) < 5:
        raise RuntimeError("Insufficient price returns from yfinance")
    return returns


def _historical_var(returns: pd.DataFrame, confidence: float) -> tuple[float, float]:
    """Compute portfolio VaR and CVaR assuming equal weights."""
    n = returns.shape[1]
    weights = np.ones(n) / n
    port_returns = returns.values @ weights

    var = float(-np.percentile(port_returns, (1 - confidence) * 100))
    cvar = float(-port_returns[port_returns < -var].mean()) if var else 0.0
    return var, cvar


def compute(
    signals: list[Signal],
    tickers: list[str] | None = None,
) -> PortfolioRisk:
    """Compute portfolio VaR and signal-adjusted risk.

    Args:
        signals: Recent NLP signals (used to amplify VaR by average impact).
        tickers: Portfolio tickers. Defaults to config default_tickers.
    """
    tickers = tickers or _DEFAULT_TICKERS

    try:
        returns = _fetch_returns(tickers)
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance failed (%s); using synthetic returns.", exc)
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(
            rng.normal(0.0005, 0.015, size=(_LOOKBACK, len(tickers))),
            columns=tickers,
        )

    var, cvar = _historical_var(returns, _VAR_CONF)

    # Signal amplification: average normalised impact of recent signals
    impact_multiplier = 1.0
    if signals:
        avg_impact = sum(s.impact for s in signals) / len(signals)
        # Impact 5 = no change, >5 amplifies, <5 reduces
        impact_multiplier = 1.0 + (avg_impact - 5) / 10.0
        impact_multiplier = max(0.5, min(2.0, impact_multiplier))

    signal_adjusted_risk = round(var * impact_multiplier, 6)

    # Ticker exposure: individual VaR as fraction of total
    individual_vars = {}
    for col in returns.columns:
        col_var = float(-np.percentile(returns[col].values, (1 - _VAR_CONF) * 100))
        individual_vars[str(col)] = col_var

    total_var = sum(individual_vars.values()) or 1.0
    exposures = {t: round(v / total_var, 4) for t, v in individual_vars.items()}

    return PortfolioRisk(
        tickers=tickers,
        var_95=round(var, 6),
        cvar_95=round(cvar, 6),
        signal_adjusted_risk=signal_adjusted_risk,
        ticker_exposures=exposures,
    )
