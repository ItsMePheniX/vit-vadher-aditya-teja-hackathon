"""test_rebalancer.py — Unit tests for the Rebalancer engine."""

from datetime import datetime, timezone

from src.modules.rebalancer.engine import RebalancerEngine


def test_equal_weights():
    engine = RebalancerEngine()
    eq = engine.equal_weights()
    assert len(eq) == 15
    assert all(abs(w - 1 / 15) < 1e-6 for w in eq.values())


def test_compute_no_sentiment():
    engine = RebalancerEngine()
    empty_map = {}
    w, turn = engine.compute(empty_map, prev_weights=None)
    assert len(w) == 15
    assert all(abs(w[t] - 1 / 15) < 1e-6 for t in w)
    assert turn == 0.0


def test_compute_with_sentiment():
    engine = RebalancerEngine()
    # If one stock has positive sentiment, it should get a higher weight
    s_map = {"JPM": 0.5, "AAPL": -0.5}
    w, turn = engine.compute(s_map, prev_weights=None)
    
    assert w["JPM"] > 1 / 15
    assert w["AAPL"] < 1 / 15
    
    # Check normalization
    assert abs(sum(w.values()) - 1.0) < 1e-6
    
    # Check min/max clipping (assuming default config bounds: 0.02 to 0.20)
    assert all(0.0199 <= weight <= 0.2001 for weight in w.values())


def test_turnover_cap():
    engine = RebalancerEngine()
    # Force a very extreme sentiment to cause large turnover
    s_map = {"JPM": 5.0, "BAC": 5.0} 
    prev = engine.equal_weights()
    
    # Run compute
    w, turn = engine.compute(s_map, prev_weights=prev)
    
    # Total turnover is limited by config (default 0.30)
    assert turn <= 0.30 + 1e-6
    
    # Re-check normalization
    assert abs(sum(w.values()) - 1.0) < 1e-6
