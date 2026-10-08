"""portfolio_builder.py — build a synthetic ~1000-position portfolio from SVB sample data.

Mapping Rules (documented, deterministic)
-----------------------------------------
The SVB news CSV has columns: id, publish_date, title, body, source, url.
The prices CSV has: date, ticker, open, high, low, close, volume.

We do NOT use real transaction data (none in the repo). Instead we generate a
reproducible synthetic portfolio seeded from the tickers and dates in the price
CSV, with the following deterministic mapping:

    ticker_hash = hash(ticker) using a fixed seed (numpy RNG 42)

    Position type assignment (by ticker_hash % 3):
        0 → Bond      (corporate bond, senior)
        1 → Loan      (bank loan, term)
        2 → Derivative (interest rate swap or equity option)

    For Bonds:
        face_value  = 1_000_000
        coupon_rate = 0.03 + (ticker_hash % 100) / 1000   (2.9–13.0%, realistic range)
        maturity_years = 1 + (ticker_hash % 9)             (1–10 years)
        ytm         = coupon_rate + (price_z_score * 0.005) (small yield offset from price)
        mod_duration = (1-(1+ytm)^-T)/ytm/((1+ytm)^1)  (standard formula)
        convexity   = analytical formula

    For Loans:
        notional    = 500_000
        pd          = 0.005 + (ticker_hash % 50) / 1000   (0.5%–5.5%)
        lgd         = 0.30 + (ticker_hash % 20) / 100     (30%–50%)
        market_value = notional * (1 - pd * lgd)          (expected-loss haircut)

    For Derivatives:
        notional    = 2_000_000
        delta       = -0.5 + (ticker_hash % 100) / 100    (-0.5 to +0.5)
        delta_dollar_value = notional * delta * (close/100)
        (approximates a delta-neutral book of options / swaps)

Each ticker in the price CSV generates N_PER_TICKER=15 positions, cycling through
the three asset types, for a total of ~#tickers × 15 ≈ 1000+ positions.

Output: data/portfolio.csv (and optionally data/portfolio.parquet if pyarrow available)

CSV columns (all numeric except asset_class, ticker, pos_id):
    pos_id, ticker, asset_class, face_value, coupon_rate, maturity_years, ytm,
    mod_duration, convexity, pd, lgd, delta, notional, market_value
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PRICES_PATH   = Path("data/svb_prices_sample.csv")
_TICKERS_YAML  = Path("src/engine/tickers.yaml")
_PORTFOLIO_CSV = Path("data/portfolio.csv")
_N_PER_TICKER  = 67    # 15 tickers × 67 = 1005 positions
_SEED          = 42


def _load_tickers() -> list[str]:
    """Load all ticker symbols from tickers.yaml; fall back to prices CSV."""
    try:
        import yaml                          # noqa: PLC0415
        raw = yaml.safe_load(_TICKERS_YAML.read_text())
        tickers = [entry["ticker"] for entry in raw.get("tickers", [])]
        if tickers:
            return sorted(tickers)
    except Exception:
        pass
    # Fallback: read from prices CSV
    try:
        prices = pd.read_csv(_PRICES_PATH)
        return sorted(prices["ticker"].unique().tolist())
    except Exception:
        return ["JPM", "BAC", "C", "GS", "WFC"]


def _ticker_int(ticker: str) -> int:
    """Deterministic integer hash for a ticker string (seeded, reproducible)."""
    h = int(hashlib.md5(ticker.encode()).hexdigest(), 16)
    return h % 10_000


def _bond_row(pos_id: str, ticker: str, th: int, rng: np.random.Generator) -> dict:
    """Build one bond position."""
    coupon   = 0.03 + (th % 70) / 1000        # 3.0% – 10.0%
    T        = 1 + (th % 9)                    # 1–9 years
    ytm      = coupon + rng.uniform(-0.005, 0.015)
    ytm      = max(0.001, ytm)
    face     = 1_000_000

    # Bond price (clean, per unit face)
    # P = coupon/ytm * (1 - (1+ytm)^-T) + face*(1+ytm)^-T
    pv_coupons = (coupon * face / ytm) * (1 - (1 + ytm) ** -T)
    pv_face    = face * (1 + ytm) ** -T
    price      = pv_coupons + pv_face

    # Modified duration (Macaulay / (1+ytm))
    mac_dur = sum(
        t * (coupon * face / (1 + ytm) ** t) for t in range(1, T + 1)
    ) + T * face / (1 + ytm) ** T
    mac_dur /= price
    mod_dur  = mac_dur / (1 + ytm)

    # Convexity
    conv_num = sum(
        t * (t + 1) * (coupon * face / (1 + ytm) ** (t + 2)) for t in range(1, T + 1)
    ) + T * (T + 1) * face / (1 + ytm) ** (T + 2)
    convexity = conv_num / price

    return {
        "pos_id":         pos_id,
        "ticker":         ticker,
        "asset_class":    "Bond",
        "face_value":     face,
        "coupon_rate":    round(coupon, 4),
        "maturity_years": T,
        "ytm":            round(ytm, 4),
        "mod_duration":   round(mod_dur, 4),
        "convexity":      round(convexity, 4),
        "pd":             0.0,
        "lgd":            0.0,
        "delta":          0.0,
        "notional":       0.0,
        "market_value":   round(price, 2),
    }


def _loan_row(pos_id: str, ticker: str, th: int, rng: np.random.Generator) -> dict:
    """Build one loan position (expected-loss haircut)."""
    notional = 500_000
    pd_val   = 0.005 + (th % 50) / 1000          # 0.5% – 5.5%
    lgd      = 0.30  + (th % 20) / 100            # 30%  – 50%
    el       = pd_val * lgd
    mv       = notional * (1 - el)
    return {
        "pos_id":         pos_id,
        "ticker":         ticker,
        "asset_class":    "Loan",
        "face_value":     notional,
        "coupon_rate":    0.0,
        "maturity_years": 0,
        "ytm":            0.0,
        "mod_duration":   0.0,
        "convexity":      0.0,
        "pd":             round(pd_val, 5),
        "lgd":            round(lgd, 4),
        "delta":          0.0,
        "notional":       notional,
        "market_value":   round(mv, 2),
    }


def _deriv_row(pos_id: str, ticker: str, th: int, close_price: float,
               rng: np.random.Generator) -> dict:
    """Build one derivative position (delta × notional approximation)."""
    notional = 2_000_000
    # Delta in (-0.5, 0.5): use th deterministically, ensure non-trivial value
    delta    = -0.5 + (th % 100) / 100       # -0.50 to +0.49
    # Dollar delta-value: notional × delta × (close/reference_price)
    ref_price = 100.0
    dv = notional * delta * (close_price / ref_price)
    return {
        "pos_id":         pos_id,
        "ticker":         ticker,
        "asset_class":    "Derivative",
        "face_value":     0.0,
        "coupon_rate":    0.0,
        "maturity_years": 0,
        "ytm":            0.0,
        "mod_duration":   0.0,
        "convexity":      0.0,
        "pd":             0.0,
        "lgd":            0.0,
        "delta":          round(delta, 4),
        "notional":       float(notional),
        "market_value":   round(dv, 2),
    }


def build_portfolio(
    prices_path: Path = _PRICES_PATH,
    n_per_ticker: int = _N_PER_TICKER,
    seed: int = _SEED,
    out_path: Path = _PORTFOLIO_CSV,
) -> pd.DataFrame:
    """Generate synthetic portfolio and save to CSV.

    Args:
        prices_path:  SVB prices CSV with columns [date, ticker, close, ...]
        n_per_ticker: how many positions to create per ticker (default 15)
        seed:         numpy RNG seed for reproducibility
        out_path:     where to write the output CSV

    Returns:
        DataFrame of positions (also written to out_path).

    Mapping summary:
        position_index % 3 == 0 → Bond
        position_index % 3 == 1 → Loan
        position_index % 3 == 2 → Derivative
    """
    rng = np.random.default_rng(seed)

    # Load tickers from tickers.yaml (falls back to prices CSV)
    tickers = _load_tickers()

    # Load close prices for tickers that appear in the prices CSV
    avg_close: dict[str, float] = {}
    if prices_path.exists():
        prices = pd.read_csv(prices_path)
        avg_close = prices.groupby("ticker")["close"].mean().to_dict()

    # Synthetic close price for any ticker not in price data
    # Use deterministic value: 50 + (ticker_hash % 200)
    def _close(t: str) -> float:
        return avg_close.get(t, 50.0 + (_ticker_int(t) % 200))

    rows: list[dict] = []
    pos_counter = 0

    for ticker in sorted(tickers):
        th    = _ticker_int(ticker)
        close = _close(ticker)

        for i in range(n_per_ticker):
            pos_counter += 1
            pos_id     = f"{ticker}_{pos_counter:05d}"
            asset_type = i % 3   # 0=Bond, 1=Loan, 2=Derivative

            if asset_type == 0:
                rows.append(_bond_row(pos_id, ticker, th + i, rng))
            elif asset_type == 1:
                rows.append(_loan_row(pos_id, ticker, th + i, rng))
            else:
                rows.append(_deriv_row(pos_id, ticker, th + i, close, rng))

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(
        "Portfolio built: %d positions (%d tickers × %d each) → %s",
        len(df), len(tickers), n_per_ticker, out_path,
    )

    # Also write parquet if pyarrow available
    parquet_path = out_path.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        logger.info("Parquet written: %s", parquet_path)
    except ImportError:
        logger.debug("pyarrow not installed; parquet skipped")

    return df


def load_portfolio(path: Path = _PORTFOLIO_CSV) -> pd.DataFrame:
    """Load portfolio from CSV (or parquet if available)."""
    parquet = path.with_suffix(".parquet")
    if parquet.exists():
        try:
            return pd.read_parquet(parquet)
        except Exception:
            pass
    if path.exists():
        return pd.read_csv(path)
    raise FileNotFoundError(
        f"Portfolio not found at {path}. Run `make stress-build` to generate it."
    )
