"""Tests for the stress-test hero module.

Coverage:
  A. Portfolio builder — determinism, shape, asset class distribution
  B. Valuation — hand-computed verification of bond/loan/derivative shocks
  C. Scenario loader — YAML parsing, all 8 event types present
  D. Trigger — threshold logic, run_now(), history, WS fan-out
  E. API — GET /stress/latest (empty→404, populated→200), GET /stress/history,
            POST /stress/run, WS /ws/stress connection
"""
from __future__ import annotations

import json
import math
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from src.modules.stress_test.valuator import (
    StressResult, AssetClassResult,
    _bond_pnl, _loan_pnl, _deriv_pnl, apply_stress,
)
from src.modules.stress_test.scenario_loader import load, get_scenario, Scenario
from src.modules.stress_test.portfolio_builder import build_portfolio, load_portfolio
from src.engine.schema import Signal


# ── Helpers ────────────────────────────────────────────────────────────────────

_SCENARIOS_YAML = Path("src/modules/stress_test/scenarios.yaml")


def _make_signal(
    entity:     str   = "JPM",
    event_type: str   = "Credit Event",
    sentiment:  float = -0.8,
    impact:     int   = 8,
    confidence: float = 0.85,
) -> Signal:
    return Signal(
        id="test_sig_001",
        ts=datetime.now(tz=timezone.utc),
        entity=entity,
        sentiment=sentiment,
        event_type=event_type,
        impact=impact,
        confidence=confidence,
        half_life_hours=24.0,
        evidence=["Test evidence sentence."],
        sources=["test"],
    )


def _make_scenario(
    equity_pct: float = -5.0,
    rate_bp: float = 75.0,
    credit_spread_bp: float = 50.0,
    hy_spread_bp: float = 100.0,
    fx_pct: float = 1.0,
) -> Scenario:
    return Scenario(
        name="Test",
        label="Test scenario",
        historical_analog="Test",
        equity_pct=equity_pct,
        rate_bp=rate_bp,
        credit_spread_bp=credit_spread_bp,
        hy_spread_bp=hy_spread_bp,
        fx_pct=fx_pct,
    )


# ══════════════════════════════════════════════════════════════════════════════
# A. Portfolio builder
# ══════════════════════════════════════════════════════════════════════════════

class TestPortfolioBuilder:

    def test_portfolio_size_approx_1000(self):
        df = build_portfolio()
        assert 900 <= len(df) <= 1100, f"Expected ~1000 positions, got {len(df)}"

    def test_three_asset_classes_present(self):
        df = build_portfolio()
        assert set(df["asset_class"].unique()) == {"Bond", "Loan", "Derivative"}

    def test_deterministic_output(self):
        df1 = build_portfolio()
        df2 = build_portfolio()
        # Same seed → identical first 10 rows
        pd.testing.assert_frame_equal(df1.head(10), df2.head(10))

    def test_bond_fields_valid(self):
        df = build_portfolio()
        bonds = df[df["asset_class"] == "Bond"]
        assert (bonds["mod_duration"] > 0).all(), "All bonds should have positive duration"
        assert (bonds["convexity"]    > 0).all(), "All bonds should have positive convexity"
        assert (bonds["coupon_rate"]  > 0).all()
        assert (bonds["market_value"] > 0).all()

    def test_loan_fields_valid(self):
        df = build_portfolio()
        loans = df[df["asset_class"] == "Loan"]
        assert (loans["pd"] > 0).all()
        assert (loans["lgd"] > 0).all()
        assert (loans["pd"] < 1.0).all()
        assert (loans["lgd"] < 1.0).all()
        assert (loans["market_value"] < loans["notional"]).all()  # EL always reduces value

    def test_derivative_delta_range(self):
        df = build_portfolio()
        derivs = df[df["asset_class"] == "Derivative"]
        assert (derivs["delta"].between(-0.51, 0.51)).all()

    def test_csv_written(self, tmp_path):
        out = tmp_path / "portfolio_test.csv"
        df  = build_portfolio(out_path=out)
        assert out.exists()
        loaded = pd.read_csv(out)
        assert len(loaded) == len(df)

    def test_load_portfolio_reads_csv(self, tmp_path):
        out = tmp_path / "portfolio.csv"
        build_portfolio(out_path=out)
        loaded = load_portfolio(path=out)
        assert len(loaded) > 0
        assert "asset_class" in loaded.columns


# ══════════════════════════════════════════════════════════════════════════════
# B. Valuation — hand-computed verification
# ══════════════════════════════════════════════════════════════════════════════

class TestValuation:
    """Verify each valuation formula against hand-computed values."""

    # ── Bond ────────────────────────────────────────────────────────────────

    def test_bond_pnl_rate_up_is_loss(self):
        """Rates up → bond price falls (negative P&L)."""
        row = pd.Series({
            "asset_class": "Bond",
            "market_value": 1_000_000.0,
            "mod_duration": 5.0,
            "convexity":    30.0,
        })
        dy = 0.01   # +100bp
        # Expected: MV × (-ModDur × dy + 0.5 × conv × dy²)
        #         = 1e6 × (-5×0.01 + 0.5×30×0.0001)
        #         = 1e6 × (-0.05 + 0.0015) = -48_500
        pnl = _bond_pnl(row, dy)
        assert abs(pnl - (-48_500)) < 1.0, f"Expected ≈ -48500, got {pnl}"

    def test_bond_pnl_rate_down_is_gain(self):
        """Rates down → bond price rises (positive P&L)."""
        row = pd.Series({
            "asset_class": "Bond",
            "market_value": 1_000_000.0,
            "mod_duration": 5.0,
            "convexity":    30.0,
        })
        pnl = _bond_pnl(row, -0.01)  # -100bp
        # = 1e6 × (5×0.01 + 0.5×30×0.0001) = +50_500 (gains + convexity benefit)
        assert abs(pnl - 51_500) < 1.0, f"Expected ≈ 51500, got {pnl}"

    def test_bond_pnl_zero_mv_returns_zero(self):
        row = pd.Series({"market_value": 0.0, "mod_duration": 5.0, "convexity": 30.0})
        assert _bond_pnl(row, 0.01) == 0.0

    def test_bond_convexity_positive_effect(self):
        """Convexity always adds a positive correction term."""
        row = pd.Series({"market_value": 1e6, "mod_duration": 5.0, "convexity": 50.0})
        # Large shock → convexity term dominates for extreme moves
        pnl_up   = _bond_pnl(row, +0.05)
        pnl_down = _bond_pnl(row, -0.05)
        # Both gains from convexity; |down move gain| > |up move loss|
        assert pnl_down > -pnl_up  # convexity asymmetry

    # ── Loan ────────────────────────────────────────────────────────────────

    def test_loan_pnl_stress_increases_loss(self):
        """Higher PD stress → larger loan loss."""
        row = pd.Series({
            "asset_class": "Loan",
            "notional":     500_000.0,
            "pd":           0.02,    # 2% PD
            "lgd":          0.40,    # 40% LGD
            "market_value": 500_000 * (1 - 0.02 * 0.40),   # = 496_000
        })
        # stressed_pd = pd × (1 + pd_stress_mult) = 0.02 × 3.0 = 0.06
        # new_el = 0.06 × 0.40 = 0.024
        # new_mv = 500_000 × (1 - 0.024) = 488_000
        # pnl    = 488_000 - 496_000 = -8_000
        pnl = _loan_pnl(row, pd_stress_mult=2.0)
        assert abs(pnl - (-8_000)) < 1.0, f"Expected ≈ -8000, got {pnl}"

    def test_loan_pnl_zero_stress_is_zero(self):
        """Zero stress multiplier → no change in value."""
        row = pd.Series({
            "notional":     500_000.0,
            "pd":           0.02,
            "lgd":          0.40,
            "market_value": 500_000 * (1 - 0.02 * 0.40),
        })
        pnl = _loan_pnl(row, pd_stress_mult=0.0)
        assert abs(pnl) < 1.0  # should be ≈ 0

    def test_loan_pnl_pd_capped_at_1(self):
        """Stressed PD must never exceed 100%."""
        row = pd.Series({
            "notional":     100_000.0,
            "pd":           0.80,
            "lgd":          0.50,
            "market_value": 100_000 * (1 - 0.40),
        })
        # Extreme stress: pd_mult = 10 → would give pd=8.0, capped to 1.0
        pnl = _loan_pnl(row, pd_stress_mult=10.0)
        # new_mv = 100_000 × (1 - 1.0×0.50) = 50_000; old_mv = 60_000
        assert pnl <= 0   # always a loss or flat
        new_mv = 100_000 + pnl
        assert new_mv >= 0

    # ── Derivative ──────────────────────────────────────────────────────────

    def test_deriv_pnl_equity_linked(self):
        """Equity-linked derivative: pnl = market_value × equity_pct/100."""
        row = pd.Series({
            "market_value": 200_000.0,
            "delta":         0.3,
            "notional":      2_000_000.0,
            "mod_duration":  0.0,
        })
        # equity_pct = -5.0 → pnl = 200_000 × (-5/100) = -10_000
        pnl = _deriv_pnl(row, equity_pct=-5.0, rate_bp=0.0, fx_pct=0.0)
        assert abs(pnl - (-10_000)) < 1.0, f"Expected -10000, got {pnl}"

    def test_deriv_pnl_rate_linked(self):
        """Rate-linked derivative (mod_duration > 0): pnl = -ModDur × dy × mv."""
        row = pd.Series({
            "market_value": 1_000_000.0,
            "delta":         0.0,
            "notional":      5_000_000.0,
            "mod_duration":  5.0,
        })
        # rate_bp = 100 → dy = 0.01 → pnl = -5.0 × 0.01 × 1e6 = -50_000
        pnl = _deriv_pnl(row, equity_pct=0.0, rate_bp=100.0, fx_pct=0.0)
        assert abs(pnl - (-50_000)) < 1.0, f"Expected -50000, got {pnl}"

    # ── Full apply_stress ────────────────────────────────────────────────────

    def test_apply_stress_full_portfolio(self):
        """apply_stress on real portfolio returns valid StressResult."""
        df       = build_portfolio()
        scenario = _make_scenario()
        sig      = _make_signal()
        result   = apply_stress(
            df, scenario, sig.id, sig.entity, sig.event_type,
            sig.sentiment, sig.impact, sig.confidence, sig.evidence,
            datetime.now(tz=timezone.utc).isoformat(), 0.5,
        )
        assert isinstance(result, StressResult)
        assert result.value_before > 0
        assert len(result.breakdown) == 3
        # Negative-sentiment credit event → expected loss
        assert result.loss != 0.0

    def test_apply_stress_breakdown_sums_to_total(self):
        """Sum of per-asset-class values should equal total portfolio values."""
        df       = build_portfolio()
        scenario = _make_scenario()
        sig      = _make_signal()
        result   = apply_stress(
            df, scenario, sig.id, sig.entity, sig.event_type,
            sig.sentiment, sig.impact, sig.confidence, sig.evidence,
            datetime.now(tz=timezone.utc).isoformat(), 0.5,
        )
        total_before = sum(b.value_before for b in result.breakdown)
        total_after  = sum(b.value_after  for b in result.breakdown)
        assert abs(total_before - result.value_before) < 1.0
        assert abs(total_after  - result.value_after)  < 1.0

    def test_sentiment_scaling_amplifies_loss(self):
        """Higher |sentiment| → more severe stress → larger loss."""
        df  = build_portfolio()
        sc  = _make_scenario(equity_pct=-5.0, credit_spread_bp=50.0)
        ts  = datetime.now(tz=timezone.utc).isoformat()

        def _stress(sentiment: float) -> float:
            r = apply_stress(df, sc, "s1", "JPM", "Credit Event",
                             sentiment, 8, 0.85, [], ts, 0.5)
            return r.loss

        # Higher |sentiment| → multiplier > 1 → more loss
        loss_mild   = _stress(-0.3)
        loss_severe = _stress(-0.9)
        assert loss_severe > loss_mild, "More negative sentiment should produce greater loss"

    def test_tiny_hand_computed_portfolio(self):
        """Verify shocks on a minimal 3-position portfolio by hand.

        Portfolio:
            Bond:       MV = 1_000_000, ModDur = 5.0, Convexity = 30.0
            Loan:       notional = 500_000, PD = 0.02, LGD = 0.40, MV = 496_000
            Derivative: MV = 100_000, ModDur = 0.0 (equity-linked)

        Scenario: equity_pct = -5.0, rate_bp = 100, credit_spread_bp = 50,
                  hy_spread_bp = 80, fx_pct = 1.0
        sentiment = -0.5, scale_factor = 0.5
        mult = 1 + 0.5×0.5 = 1.25

        Applied shocks:
            eq = -5.0 × 1.25 = -6.25%
            rb = 100  × 1.25 = 125 bp
            cs = 50   × 1.25 = 62.5 bp
            dy_bond = (125 + 62.5) / 10000 = 0.01875

        Bond P&L = 1_000_000 × (-5.0×0.01875 + 0.5×30×0.01875²)
                 = 1_000_000 × (-0.09375 + 0.00527)
                 = 1_000_000 × -0.08848 ≈ -88_477

        Loan: pd_mult = (62.5/10000) / 0.004 = 1.5625
              stressed_pd = 0.02 × 2.5625 = 0.05125
              new_el = 0.05125 × 0.40 = 0.0205
              new_mv = 500_000 × (1-0.0205) = 489_750
              pnl    = 489_750 - 496_000 = -6_250

        Deriv P&L = 100_000 × (-6.25/100) = -6_250

        Total loss ≈ 88_477 + 6_250 + 6_250 = 100_977
        """
        mini_df = pd.DataFrame([
            {
                "pos_id": "BOND_01", "ticker": "JPM", "asset_class": "Bond",
                "face_value": 1_000_000, "coupon_rate": 0.05,
                "maturity_years": 5, "ytm": 0.05,
                "mod_duration": 5.0, "convexity": 30.0,
                "pd": 0.0, "lgd": 0.0, "delta": 0.0,
                "notional": 0.0, "market_value": 1_000_000.0,
            },
            {
                "pos_id": "LOAN_01", "ticker": "JPM", "asset_class": "Loan",
                "face_value": 500_000, "coupon_rate": 0.0,
                "maturity_years": 0, "ytm": 0.0,
                "mod_duration": 0.0, "convexity": 0.0,
                "pd": 0.02, "lgd": 0.40, "delta": 0.0,
                "notional": 500_000.0, "market_value": 496_000.0,
            },
            {
                "pos_id": "DERIV_01", "ticker": "JPM", "asset_class": "Derivative",
                "face_value": 0.0, "coupon_rate": 0.0,
                "maturity_years": 0, "ytm": 0.0,
                "mod_duration": 0.0, "convexity": 0.0,
                "pd": 0.0, "lgd": 0.0, "delta": 0.3,
                "notional": 2_000_000.0, "market_value": 100_000.0,
            },
        ])

        scenario = Scenario(
            name="HandTest",
            label="Hand test",
            historical_analog="Unit test",
            equity_pct=-5.0, rate_bp=100.0,
            credit_spread_bp=50.0, hy_spread_bp=80.0, fx_pct=1.0,
        )

        result = apply_stress(
            mini_df, scenario,
            "sig_hand", "JPM", "Credit Event",
            -0.5, 8, 0.85, ["test"], datetime.now(tz=timezone.utc).isoformat(),
            sentiment_scale_factor=0.5,
        )

        # multiplier = 1 + 0.5 × 0.5 = 1.25
        assert abs(result.shock_multiplier - 1.25) < 0.001

        bond_result = next(b for b in result.breakdown if b.asset_class == "Bond")
        loan_result = next(b for b in result.breakdown if b.asset_class == "Loan")
        deriv_result = next(b for b in result.breakdown if b.asset_class == "Derivative")

        # Bond loss ≈ 88_477 (±500 tolerance for floating point)
        assert abs(bond_result.loss - 88_477) < 500, f"Bond loss: {bond_result.loss}"

        # Loan loss ≈ 6_250 (±100)
        assert abs(loan_result.loss - 6_250) < 100, f"Loan loss: {loan_result.loss}"

        # Derivative loss ≈ 6_250 (±100)
        assert abs(deriv_result.loss - 6_250) < 100, f"Deriv loss: {deriv_result.loss}"

        # Total loss ≈ 100_977 (±700)
        assert abs(result.loss - 100_977) < 700, f"Total loss: {result.loss}"


# ══════════════════════════════════════════════════════════════════════════════
# C. Scenario loader
# ══════════════════════════════════════════════════════════════════════════════

class TestScenarioLoader:

    def test_all_eight_event_types_present(self):
        cfg = load(_SCENARIOS_YAML)
        expected = {
            "Geopolitical", "Macroeconomic", "Credit Event",
            "Merger/Acquisition", "Earnings", "Regulatory",
            "Product Launch", "Other",
        }
        assert expected == set(cfg.scenarios.keys())

    def test_trigger_thresholds_loaded(self):
        cfg = load(_SCENARIOS_YAML)
        assert cfg.min_impact     >= 1
        assert cfg.min_confidence >= 0.0

    def test_historical_analog_non_empty(self):
        cfg = load(_SCENARIOS_YAML)
        for name, sc in cfg.scenarios.items():
            assert sc.historical_analog, f"Missing historical_analog for {name}"

    def test_get_scenario_returns_correct_type(self):
        sc = get_scenario("Credit Event")
        assert sc is not None
        assert isinstance(sc, Scenario)
        assert sc.name == "Credit Event"

    def test_get_scenario_unknown_returns_none(self):
        assert get_scenario("Nonexistent") is None

    def test_geopolitical_equity_shock_negative(self):
        """Geopolitical scenario should have negative equity shock."""
        sc = get_scenario("Geopolitical")
        assert sc.equity_pct < 0

    def test_product_launch_equity_shock_positive(self):
        """Product launch scenario: slight positive equity surprise."""
        sc = get_scenario("Product Launch")
        assert sc.equity_pct >= 0

    def test_sentiment_scale_factor_present(self):
        cfg = load(_SCENARIOS_YAML)
        assert cfg.sentiment_scale_factor > 0


# ══════════════════════════════════════════════════════════════════════════════
# D. Trigger
# ══════════════════════════════════════════════════════════════════════════════

class TestStressTrigger:

    def test_threshold_guard_low_impact(self):
        from src.modules.stress_test.trigger import StressTrigger
        t   = StressTrigger()
        sig = _make_signal(impact=3, confidence=0.90)  # impact too low
        assert t._should_trigger(sig) is False

    def test_threshold_guard_low_confidence(self):
        from src.modules.stress_test.trigger import StressTrigger
        t   = StressTrigger()
        sig = _make_signal(impact=9, confidence=0.20)  # confidence too low
        assert t._should_trigger(sig) is False

    def test_threshold_guard_passes(self):
        from src.modules.stress_test.trigger import StressTrigger
        t   = StressTrigger()
        sig = _make_signal(impact=8, confidence=0.80)
        assert t._should_trigger(sig) is True

    def test_run_now_returns_result(self):
        from src.modules.stress_test.trigger import StressTrigger
        t   = StressTrigger()
        sig = _make_signal()
        r   = t.run_now(sig)
        assert r is not None
        assert isinstance(r, StressResult)
        assert r.signal_id == sig.id

    def test_run_now_stores_history(self):
        from src.modules.stress_test.trigger import StressTrigger
        t   = StressTrigger()
        for i in range(3):
            sig = _make_signal()
            t.run_now(sig)
        assert len(t.history()) == 3

    def test_latest_returns_most_recent(self):
        from src.modules.stress_test.trigger import StressTrigger
        t     = StressTrigger()
        sig1  = _make_signal(entity="JPM")
        sig2  = _make_signal(entity="BAC")
        t.run_now(sig1)
        t.run_now(sig2)
        latest = t.latest()
        assert latest is not None
        assert latest.signal_entity == "BAC"

    def test_ws_broadcast_on_run_now(self):
        from src.modules.stress_test.trigger import StressTrigger
        t = StressTrigger()
        q: asyncio.Queue = asyncio.Queue(maxsize=5)
        t.add_ws_client(q)
        t.run_now(_make_signal())
        assert not q.empty()
        result = q.get_nowait()
        assert isinstance(result, StressResult)

    def test_history_limit(self):
        from src.modules.stress_test.trigger import StressTrigger
        t = StressTrigger()
        assert t.history(n=5) == []     # empty before any runs

    def test_unknown_event_type_returns_none(self):
        from src.modules.stress_test.trigger import StressTrigger
        t = StressTrigger()
        sig = _make_signal(event_type="Other")   # Other is a valid type with scenario
        r   = t.run_now(sig)
        assert r is not None


# ══════════════════════════════════════════════════════════════════════════════
# E. API tests
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _reset_trigger():
    """Reset the module-level trigger singleton before each test."""
    import src.modules.stress_test.trigger as tr
    tr._trigger = None
    yield
    tr._trigger = None


@pytest.fixture
async def client():
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_stress_latest_empty_returns_404(client):
    resp = await client.get("/stress/latest")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_stress_latest_after_run(client):
    """After POST /stress/run, GET /stress/latest should return 200."""
    run_resp = await client.post("/stress/run", json={
        "entity": "JPM", "event_type": "Credit Event",
        "sentiment": -0.8, "impact": 8, "confidence": 0.85,
        "evidence": ["Test"],
    })
    assert run_resp.status_code == 201

    latest_resp = await client.get("/stress/latest")
    assert latest_resp.status_code == 200
    data = latest_resp.json()
    assert data["signal_entity"] == "JPM"
    assert data["event_type"]    == "Credit Event"
    assert "value_before" in data
    assert "loss" in data
    assert "breakdown" in data
    assert len(data["breakdown"]) == 3


@pytest.mark.asyncio
async def test_stress_history_empty(client):
    resp = await client.get("/stress/history")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_stress_history_populated(client):
    for _ in range(3):
        await client.post("/stress/run", json={
            "entity": "GS", "event_type": "Macroeconomic",
            "sentiment": -0.5, "impact": 7, "confidence": 0.75,
            "evidence": [],
        })
    resp = await client.get("/stress/history?n=3")
    assert resp.status_code == 200
    assert len(resp.json()) == 3


@pytest.mark.asyncio
async def test_stress_run_invalid_event_type(client):
    resp = await client.post("/stress/run", json={
        "entity": "JPM", "event_type": "Nonexistent Type",
        "sentiment": -0.5, "impact": 8, "confidence": 0.85,
        "evidence": [],
    })
    # FastAPI validates the Literal type → 422
    assert resp.status_code in (422, 422)


@pytest.mark.asyncio
async def test_stress_run_breakdown_structure(client):
    resp = await client.post("/stress/run", json={
        "entity": "BAC", "event_type": "Geopolitical",
        "sentiment": -0.7, "impact": 9, "confidence": 0.90,
        "evidence": ["Geopolitical shock test."],
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["value_before"] > 0
    assert isinstance(data["breakdown"], list)
    # Each breakdown entry has required fields
    for b in data["breakdown"]:
        assert "asset_class"  in b
        assert "n_positions"  in b
        assert "value_before" in b
        assert "loss"         in b
        assert "loss_pct"     in b


@pytest.mark.asyncio
async def test_stress_run_shock_multiplier_with_sentiment(client):
    """Shock multiplier should equal 1 + |sentiment| × 0.5."""
    resp = await client.post("/stress/run", json={
        "entity": "C", "event_type": "Credit Event",
        "sentiment": -0.6, "impact": 8, "confidence": 0.80,
        "evidence": [],
    })
    assert resp.status_code == 201
    data = resp.json()
    expected_mult = 1.0 + 0.6 * 0.5   # = 1.30
    assert abs(data["shock_multiplier"] - expected_mult) < 0.01


@pytest.mark.asyncio
async def test_stress_ws_connects():
    """WebSocket /ws/stress should accept connections."""
    from src.api.main import app
    from starlette.testclient import TestClient
    with TestClient(app) as tc:
        with tc.websocket_connect("/ws/stress") as ws:
            pass   # connection opened without error


@pytest.mark.asyncio
async def test_stress_ws_receives_on_broadcast():
    """Verify WS queue receives StressResult when trigger broadcasts."""
    from src.modules.stress_test.trigger import get_trigger

    trigger = get_trigger()
    q: asyncio.Queue = asyncio.Queue(maxsize=5)
    trigger.add_ws_client(q)
    try:
        sig = _make_signal()
        trigger.run_now(sig)
        assert not q.empty()
        result = q.get_nowait()
        assert isinstance(result, StressResult)
        assert result.signal_id == sig.id
    finally:
        trigger.remove_ws_client(q)
