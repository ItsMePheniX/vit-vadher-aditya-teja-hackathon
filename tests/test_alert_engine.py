"""Tests for the alert engine."""
from __future__ import annotations

from datetime import datetime, timezone

from src.engine.schema import Signal
from src.modules.alert_engine import evaluate, evaluate_batch

_BASE = {
    "id": "01JBXYZ",
    "ts": datetime.now(tz=timezone.utc),
    "entity": "AAPL",
    "sentiment": -0.8,
    "event_type": "Earnings",
    "impact": 8,
    "confidence": 0.9,
    "half_life_hours": 12.0,
    "evidence": [],
    "sources": [],
}


def _make(**overrides) -> Signal:  # type: ignore[no-untyped-def]
    return Signal(**{**_BASE, **overrides})


def test_high_severity_alert() -> None:
    signal = _make(sentiment=-0.8, impact=8, confidence=0.9)
    alert = evaluate(signal)
    assert alert is not None
    assert alert.severity == "HIGH"


def test_no_alert_low_confidence() -> None:
    signal = _make(confidence=0.3)
    alert = evaluate(signal)
    assert alert is None


def test_medium_alert_impact_only() -> None:
    # Impact high but sentiment neutral
    signal = _make(sentiment=0.1, impact=8, confidence=0.9)
    alert = evaluate(signal)
    assert alert is not None
    assert alert.severity == "MEDIUM"


def test_no_alert_below_threshold() -> None:
    signal = _make(sentiment=0.5, impact=3, confidence=0.9)
    alert = evaluate(signal)
    assert alert is None


def test_evaluate_batch() -> None:
    signals = [
        _make(sentiment=-0.8, impact=8, confidence=0.9),
        _make(sentiment=0.5, impact=3, confidence=0.9),
        _make(sentiment=-0.6, impact=7, confidence=0.7),
    ]
    alerts = evaluate_batch(signals)
    assert len(alerts) == 2
