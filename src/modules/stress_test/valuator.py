"""valuator.py — apply scenario shocks to a portfolio and compute P&L.

Valuation Models
----------------

BONDS (modified duration + convexity):
    ΔP/P ≈ −ModDur × Δy + ½ × Convexity × (Δy)²
    where Δy = rate_bp / 10000  (parallel yield curve shift)
    Also absorbs credit spread widening:
        Δy_total = rate_bp/10000 + credit_spread_bp/10000
    Bond P&L = market_value × (−ModDur×Δy_total + 0.5×Convexity×Δy_total²)

    Reference: Fabozzi, "Fixed Income Mathematics" (public domain formula)

LOANS (PD/LGD expected-loss haircut):
    stress_pd   = pd * (1 + pd_stress_multiplier)   # stress multiplier from scenario
    stress_el   = stress_pd * lgd
    new_mv      = notional * (1 − stress_el)
    P&L         = new_mv − market_value              (negative = loss)

    PD stress multiplier is derived from credit_spread_bp:
        pd_stress_mult = credit_spread_bp / 10000 / 0.004   # normalised by 4bp/1% PD
    (UNVERIFIED approximation — typical IG spread = 4bp per 1% PD)

DERIVATIVES (delta-notional approximation):
    Equity derivatives:
        delta_pnl = market_value × equity_pct/100
    FX derivatives (where ticker ends in currency code or is FOREX):
        delta_pnl = notional × delta × fx_pct/100
    Rate derivatives:
        delta_pnl = −mod_duration × rate_bp/10000 × market_value
    Classification: we use delta sign and asset class heuristics.
    Default: treat as equity-linked when no other information.

EQUITY (if any equity positions appear):
    pnl = market_value × equity_pct/100
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.modules.stress_test.scenario_loader import Scenario

logger = logging.getLogger(__name__)

_PD_BP_SCALE = 0.004   # 4bp per 1% PD (UNVERIFIED approximation)


@dataclass
class AssetClassResult:
    """P&L breakdown for one asset class."""
    asset_class:   str
    n_positions:   int
    value_before:  float
    value_after:   float
    loss:          float          # positive = loss (value_before > value_after)
    loss_pct:      float          # loss / value_before


@dataclass
class StressResult:
    """Full output of one stress-test run."""
    signal_id:     str
    signal_entity: str
    event_type:    str
    scenario_name: str
    sentiment:     float
    impact:        int
    confidence:    float
    evidence:      list[str]
    shock_multiplier: float       # how scenario was scaled by sentiment

    value_before:  float
    value_after:   float
    loss:          float
    loss_pct:      float

    breakdown:     list[AssetClassResult]
    triggered_at:  str            # ISO8601 timestamp

    # Shock parameters actually applied
    equity_pct_applied:        float
    rate_bp_applied:           float
    credit_spread_bp_applied:  float
    hy_spread_bp_applied:      float
    fx_pct_applied:            float


# ── Core valuation functions ─────────────────────────────────────────────────

def _bond_pnl(row: pd.Series, dy: float) -> float:
    """ΔP ≈ −ModDur×Δy + ½×Convexity×Δy² scaled by market_value."""
    mv  = float(row["market_value"])
    md  = float(row["mod_duration"])
    cx  = float(row["convexity"])
    if mv == 0:
        return 0.0
    dp_over_p = -md * dy + 0.5 * cx * dy * dy
    return mv * dp_over_p


def _loan_pnl(row: pd.Series, pd_stress_mult: float) -> float:
    """New EL = stressed_pd × LGD; P&L = new_mv − old_mv."""
    notional = float(row["notional"])
    pd_val   = float(row["pd"])
    lgd      = float(row["lgd"])
    old_mv   = float(row["market_value"])
    if notional == 0:
        return 0.0
    stressed_pd = min(pd_val * (1 + pd_stress_mult), 1.0)
    new_el      = stressed_pd * lgd
    new_mv      = notional * (1 - new_el)
    return new_mv - old_mv


def _deriv_pnl(row: pd.Series, equity_pct: float, rate_bp: float, fx_pct: float) -> float:
    """Delta-notional P&L for derivatives."""
    mv     = float(row["market_value"])
    delta  = float(row["delta"])
    notional = float(row["notional"])
    md     = float(row["mod_duration"])

    if notional == 0:
        return 0.0

    # Heuristic classification based on delta sign and magnitude
    # Positive-delta large → equity-linked long
    # Near-zero delta + non-zero mod_duration → rate-linked
    # Negative-delta → short equity or receiving-fixed swap
    if abs(md) > 0.01:  # rate-linked
        return -md * (rate_bp / 10_000) * mv
    else:               # equity-linked (default)
        return mv * (equity_pct / 100)


# ── Main stress function ──────────────────────────────────────────────────────

def apply_stress(
    portfolio: pd.DataFrame,
    scenario: "Scenario",
    signal_id: str,
    signal_entity: str,
    event_type: str,
    sentiment: float,
    impact: int,
    confidence: float,
    evidence: list[str],
    triggered_at: str,
    sentiment_scale_factor: float = 0.5,
) -> StressResult:
    """Apply a stress scenario to the portfolio and return a StressResult.

    Args:
        portfolio:   DataFrame from portfolio_builder.load_portfolio()
        scenario:    Scenario namedtuple with shock parameters
        signal_*:    Signal metadata for the result object
        sentiment_scale_factor: scales scenario severity by |sentiment|
            multiplier = 1 + |sentiment| × sentiment_scale_factor
            (so high-conviction negative news amplifies the stress)

    Returns:
        StressResult with per-asset-class breakdown.
    """
    mult = 1.0 + abs(sentiment) * sentiment_scale_factor

    # Applied shocks
    eq_pct  = scenario.equity_pct  * mult
    rb_bp   = scenario.rate_bp     * mult
    cs_bp   = scenario.credit_spread_bp * mult
    hy_bp   = scenario.hy_spread_bp * mult
    fx_pct  = scenario.fx_pct      * mult

    # Total yield shift for bonds: rate + IG credit spread
    dy_bond = (rb_bp + cs_bp) / 10_000

    # PD stress multiplier for loans (from credit spread widening)
    # credit_spread_bp / 10000 / _PD_BP_SCALE = number of PD units of stress
    pd_mult = (cs_bp / 10_000) / _PD_BP_SCALE

    pnl_values: list[float] = []
    mv_before: list[float]  = []
    results_by_ac: dict[str, dict] = {}

    for ac in ["Bond", "Loan", "Derivative"]:
        sub = portfolio[portfolio["asset_class"] == ac]
        if sub.empty:
            continue
        before = float(sub["market_value"].sum())

        if ac == "Bond":
            pnls = sub.apply(lambda r: _bond_pnl(r, dy_bond), axis=1)
        elif ac == "Loan":
            pnls = sub.apply(lambda r: _loan_pnl(r, pd_mult), axis=1)
        else:
            pnls = sub.apply(lambda r: _deriv_pnl(r, eq_pct, rb_bp, fx_pct), axis=1)

        total_pnl = float(pnls.sum())
        after      = before + total_pnl
        loss       = -(total_pnl)   # positive loss = money lost
        loss_pct   = loss / before if before != 0 else 0.0

        results_by_ac[ac] = AssetClassResult(
            asset_class=ac,
            n_positions=len(sub),
            value_before=round(before, 2),
            value_after=round(after, 2),
            loss=round(loss, 2),
            loss_pct=round(loss_pct, 6),
        )
        pnl_values.append(total_pnl)
        mv_before.append(before)

    total_before = sum(mv_before)
    total_pnl    = sum(pnl_values)
    total_after  = total_before + total_pnl
    total_loss   = -total_pnl
    total_loss_pct = total_loss / total_before if total_before != 0 else 0.0

    logger.info(
        "Stress test: event=%s mult=%.2f loss=%.2f (%.2f%%) triggered_by=%s",
        event_type, mult, total_loss, total_loss_pct * 100, signal_id,
    )

    return StressResult(
        signal_id=signal_id,
        signal_entity=signal_entity,
        event_type=event_type,
        scenario_name=scenario.label,
        sentiment=round(sentiment, 4),
        impact=impact,
        confidence=round(confidence, 4),
        evidence=evidence,
        shock_multiplier=round(mult, 4),
        value_before=round(total_before, 2),
        value_after=round(total_after, 2),
        loss=round(total_loss, 2),
        loss_pct=round(total_loss_pct, 6),
        breakdown=list(results_by_ac.values()),
        triggered_at=triggered_at,
        equity_pct_applied=round(eq_pct, 4),
        rate_bp_applied=round(rb_bp, 4),
        credit_spread_bp_applied=round(cs_bp, 4),
        hy_spread_bp_applied=round(hy_bp, 4),
        fx_pct_applied=round(fx_pct, 4),
    )
